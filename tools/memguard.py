#!/usr/bin/env python3
"""memguard: run one heavy, RESUMABLE command under the ICDE27 memory policy.

  python3 tools/memguard.py [--cap-gb 60] [--cooldown 60] [--max-restarts 6] [--log logs/memguard.log] -- <command ...>

* Admission: before (re)starting, waits while logs/MEMORY_PRESSURE or logs/MEMORY_COOLDOWN exist
  (written by tools/memwatch.py, the project-wide 200 GB watchdog).
* Per-job cap: if this command's process tree footprint reaches --cap-gb, the tree is stopped.
* Host cap: pause admission and stop this job at --system-cap-gb, regardless of project share.
* If the job is stopped (by this cap or by memwatch), it is restarted after a cooldown that doubles
  each time (restart slowly), up to --max-restarts. Exit code = command's final exit code.
* The command MUST checkpoint progress so a restart resumes rather than recomputes.
* Set MEMGUARD_SHRINK=1 in the environment of restarted attempts so scripts can lower batch sizes /
  worker counts (MEMGUARD_ATTEMPT gives the attempt number).
"""
import argparse, ctypes, ctypes.util, errno, os, shlex, signal, subprocess, sys, time

from memwatch import sys_used_bytes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, "logs")
PRESSURE, COOLDOWN = os.path.join(LOGDIR, "MEMORY_PRESSURE"), os.path.join(LOGDIR, "MEMORY_COOLDOWN")
STATUS = os.path.join(LOGDIR, "memwatch_status.json")
GIB = 2**30

def watchdog_alive():
    """memwatch rewrites its status file every few seconds; a stale file means it is not running."""
    try: return time.time() - os.path.getmtime(STATUS) < 90
    except OSError: return False
libc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib", use_errno=True)
libc.proc_listchildpids.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
libc.proc_listchildpids.restype = ctypes.c_int
_buf = ctypes.create_string_buffer(512)

def footprint(pid):
    ctypes.set_errno(0)
    if libc.proc_pid_rusage(pid, 2, _buf) != 0:
        error = ctypes.get_errno()
        if error == errno.ESRCH:
            return 0
        raise OSError(error, f"cannot measure memory for job process {pid}")
    return int.from_bytes(_buf.raw[72:80], "little")


def child_pids(pid):
    size = 32
    while True:
        children = (ctypes.c_int * size)()
        ctypes.set_errno(0)
        count = libc.proc_listchildpids(pid, children, ctypes.sizeof(children))
        error = ctypes.get_errno()
        if count < 0 or (count == 0 and error not in (0, errno.ESRCH)):
            raise OSError(error, f"cannot inspect children of job process {pid}")
        if count < size:
            return children[:max(0, count)]
        size *= 2


def tree_bytes(root):
    """Measure the launched job and its descendants without listing unrelated processes."""
    total, pending, seen = 0, [root], set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        pending.extend(child_pids(pid))
        total += footprint(pid)
    return total

def kill_group(p):
    """SIGTERM the job's process group, SIGKILL after 15 s. Zombies give EPERM on macOS: treat as gone."""
    for sig, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        try: os.killpg(p.pid, sig)
        except (ProcessLookupError, PermissionError): return
        for _ in range(int(wait * 10)):
            if p.poll() is not None:
                try: os.killpg(p.pid, 0)
                except (ProcessLookupError, PermissionError): return
            time.sleep(0.1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-gb", type=float, default=60)
    ap.add_argument("--system-cap-gb", type=float, default=490)
    ap.add_argument("--poll", type=float, default=2)
    ap.add_argument("--cooldown", type=float, default=60)
    ap.add_argument("--max-restarts", type=int, default=6)
    ap.add_argument("--max-admission-wait", type=float, default=3 * 3600)
    ap.add_argument("--log", default=os.path.join(LOGDIR, "memguard.log"))
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd: ap.error("no command given")
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    logf = open(a.log, "a")
    def log(m):
        s = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] memguard[{os.getpid()}]: {m}"
        print(s, file=sys.stderr, flush=True); logf.write(s + "\n"); logf.flush()
    attempt, cooldown, rc = 0, a.cooldown, 1
    disk_critical = os.path.join(LOGDIR, "DISK_CRITICAL")
    while True:
        waited = 0
        if not watchdog_alive():
            log("project memory accounting unavailable; start tools/ensure_memwatch.sh before retrying")
            sys.exit(75)
        while True:
            try:
                host_over = sys_used_bytes() >= a.system_cap_gb * GIB
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                log(f"host memory check failed: {error}")
                sys.exit(75)
            if not watchdog_alive():
                log("project memory accounting stopped while waiting")
                sys.exit(75)
            if not (host_over or any(os.path.exists(f) for f in (PRESSURE, COOLDOWN, disk_critical))):
                break
            if waited == 0: log("host or project memory pressure/cooldown or critical disk space; waiting for admission")
            time.sleep(10); waited += 10
            if waited > a.max_admission_wait: log("admission wait exceeded; giving up"); sys.exit(75)
        attempt += 1
        env = dict(os.environ, MEMGUARD_ATTEMPT=str(attempt), MEMGUARD_SHRINK="1" if attempt > 1 else "0")
        log(f"attempt {attempt}: {shlex.join(cmd)}")
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, env=env)
        tripped, peak = False, 0
        while p.poll() is None:
            time.sleep(a.poll)
            if p.poll() is not None:
                break
            try:
                b = tree_bytes(p.pid)
                host_bytes = sys_used_bytes()
                if not watchdog_alive():
                    raise OSError("project memory accounting is stale")
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                log(f"job memory check failed: {error}; stopping")
                kill_group(p)
                p.wait()
                sys.exit(75)
            peak = max(peak, b)
            if b >= a.cap_gb * GIB or host_bytes >= a.system_cap_gb * GIB:
                log(f"memory cap hit: job {b/GIB:.1f}/{a.cap_gb:g} GiB, host {host_bytes/GIB:.1f}/{a.system_cap_gb:g} GiB; stopping")
                kill_group(p); tripped = True; break
        p.wait(); rc = p.returncode
        stopped_externally = rc in (-signal.SIGTERM, -signal.SIGKILL, 128 + 9, 128 + 15) and (os.path.exists(COOLDOWN) or os.path.exists(PRESSURE))
        log(f"attempt {attempt} ended rc={rc} peak={peak/2**30:.1f} GB" + (" (stopped by memwatch)" if stopped_externally else ""))
        if not (tripped or stopped_externally): break
        if attempt > a.max_restarts: log("max restarts exceeded; giving up"); rc = 137; break
        log(f"slow restart: sleeping {cooldown:.0f}s"); time.sleep(cooldown); cooldown = min(cooldown * 2, 1800)
    sys.exit(rc if rc >= 0 else 128 - rc)

if __name__ == "__main__":
    main()
