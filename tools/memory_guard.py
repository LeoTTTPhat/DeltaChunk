#!/usr/bin/env python3
"""Serial, fail-closed process-tree RSS guard with resumable slow retries.

200 GB means decimal bytes. Sampling cannot enforce a kernel hard memory limit.
Run every research experiment through this supervisor; never detach children.

Stop rules (2026-10-04 user rule added): the supervised tree reaches its RSS limit (never above
200 GB), OR whole-machine RAM use reaches 490 GB, OR immediately available host RAM falls below the
reserve. Machine use is measured conservatively as physical minus free and speculative pages, so
file cache counts as used and the stop comes no later than Activity Monitor's "Memory Used"
reaching 490 GB. After a stop the guard cools down (longer each time), waits until machine use is
back under 90% of the 490 GB ceiling, and restarts slowly: one worker, half the batch scale, added
pacing.

Lock (2026-10-04): waiters queue first-come-first-served through ticket files in
runs/lock_queue/, so one project's back-to-back invocations cannot starve another's.
Since 2026-10-05 --wait-lock bounds the wait without queue progress (see acquire_lock).

--start-if-fits (user rule 2026-10-26, opt-in): the FIRST launch may also proceed when current
machine use plus this job's own cap (--limit-gb) stays at or under the host ceiling, instead of
waiting for the 90% line. Every stop rule is unchanged, and a restart after a stop still waits for
the 90% line.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GB = 10**9
PROJECT_CEILING_GB = 200  # user rule: a project never exceeds 200 GB RSS
HOST_USED_CEILING_GB = 490  # user rule 2026-10-04: stop when whole-machine RAM use reaches 490 GB
RECOVER_FRACTION = 0.9  # restart only once machine use is back below 90% of the host ceiling
QUEUE_DIR = ROOT / 'runs' / 'lock_queue'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def process_table():
    out = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,pgid=,rss='], text=True, stderr=subprocess.PIPE)
    rows = {}
    for line in out.splitlines():
        pid, parent, group, rss = map(int, line.split())
        rows[pid] = (parent, group, rss * 1024)
    if not rows:
        raise RuntimeError('Process telemetry returned no rows')
    return rows


def tree_pids(rows, pid):
    selected = {p for p, (_, group, _) in rows.items() if group == pid}
    selected.add(pid)
    while True:
        new = {p for p, (parent, _, _) in rows.items() if parent in selected}
        if new.issubset(selected):
            return selected
        selected |= new


def tree_rss(rows, pid):
    return sum(rows[p][2] for p in tree_pids(rows, pid) if p in rows)


def parse_vm_stat(raw, total):
    """Return (available, used, detail) from macOS vm_stat output.

    available = free + speculative pages (inactive, purgeable and compressed pages are NOT counted
    as free). used = physical - available, i.e. file cache counts as used. detail also carries the
    Activity Monitor breakdown (app + wired + compressed = "Memory Used") for the record only."""
    page = int(re.search(r'page size of (\d+) bytes', raw).group(1))
    counts = {k.strip(): int(v) for k, v in re.findall(r'^([^:\n]+):\s*(\d+)\.', raw, re.M)}
    available = page * (counts['Pages free'] + counts.get('Pages speculative', 0))
    app = page * max(0, counts.get('Anonymous pages', 0) - counts.get('Pages purgeable', 0))
    wired = page * counts.get('Pages wired down', 0)
    compressed = page * counts.get('Pages occupied by compressor', 0)
    detail = {'activity_monitor_used_bytes': app + wired + compressed, 'app_bytes': app, 'wired_bytes': wired,
              'compressed_bytes': compressed,
              'cached_files_bytes': page * (counts.get('File-backed pages', 0) + counts.get('Pages purgeable', 0))}
    return available, total - available, detail


def host_memory(detail=False):
    if sys.platform == 'darwin':
        total = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True, stderr=subprocess.PIPE))
        raw = subprocess.check_output(['vm_stat'], text=True, stderr=subprocess.PIPE)
        available, used, extra = parse_vm_stat(raw, total)
    else:
        counts = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, val = line.split(':', 1)
            counts[key] = int(val.split()[0]) * 1024
        total, available = counts['MemTotal'], counts['MemAvailable']
        used, extra = total - available, {}
    return (total, available, used, extra) if detail else (total, available, used)


def stop_reason(rss, available, limit, reserve, host_used=0, host_limit=None):
    if rss >= limit:
        return 'project_rss_limit'
    if host_limit is not None and host_used >= host_limit:
        return 'host_used_limit'
    if available < reserve:
        return 'host_memory_reserve'
    return None


def recovered(available, used, reserve, host_limit):
    return available >= reserve * 1.25 and used <= host_limit * RECOVER_FRACTION


def fits(available, used, reserve, host_limit, limit):
    """--start-if-fits: the job's own cap fits under the host ceiling on top of current use."""
    return available >= reserve * 1.25 and used + limit <= host_limit


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def pid_is_guard(pid):
    try:
        cmd = subprocess.check_output(['ps', '-p', str(pid), '-o', 'command='], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, OSError):
        return False
    return 'memory_guard' in cmd


def live_tickets(queue_dir, verified, alive=pid_alive, is_guard=pid_is_guard):
    """Sorted tickets of waiting guards; tickets of dead or reused PIDs are removed."""
    out = []
    for t in sorted(queue_dir.glob('*_*')):
        if t.suffix == '.tmp':
            continue
        try:
            pid = int(t.name.split('_')[1])
        except (IndexError, ValueError):
            continue
        if pid == os.getpid():
            out.append(t)
            continue
        ok = alive(pid)
        if ok and t.name not in verified:
            ok = is_guard(pid)
            if ok:
                verified.add(t.name)
        if ok:
            out.append(t)
        else:
            try:
                t.unlink()
            except OSError:
                pass
    return out


#: --priority (2026-10-06, user: FastLane takes priority over the other projects): a priority ticket's
#: name is stamped this far in the past, so it sorts ahead of every normal ticket while priority
#: tickets keep first-come-first-served order among themselves. The running holder is never preempted.
PRIORITY_OFFSET_NS = 10 ** 18
#: --priority-level 2 (2026-10-06, user: FastLane before SnapMerge, both above the rest) sorts ahead of
#: level 1 in the same way; level L is stamped PRIORITY_LEVEL_OFFSET_NS[L] in the past.
PRIORITY_LEVEL_OFFSET_NS = {0: 0, 1: PRIORITY_OFFSET_NS, 2: 15 * 10 ** 17, 3: 175 * 10 ** 16}


def acquire_lock(lock, wait_s, queue_dir=QUEUE_DIR, info=None, alive=pid_alive, is_guard=pid_is_guard,
                 priority=False):
    """First-come-first-served wait for the experiment lock. Returns seconds waited or None.

    wait_s bounds the wait WITHOUT QUEUE PROGRESS (2026-10-05): the deadline restarts whenever a
    waiter ahead of this one leaves the queue (it got the lock or gave up). A guard at the head of a
    long FIFO queue therefore times out only when one holder keeps the lock for wait_s, not merely
    because several 85-minute slots were ahead of it (two invocations lost their turn that way on
    2026-10-04/05). wait_s = 0 still fails fast."""
    queue_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.time_ns() - PRIORITY_LEVEL_OFFSET_NS[int(priority)]
    ticket = queue_dir / ('%020d_%d' % (stamp, os.getpid()))
    ticket.write_text(json.dumps(info or {}) + '\n')
    t0 = time.monotonic()
    deadline = t0 + max(0.0, wait_s)
    verified = set()
    ahead_prev = None
    try:
        while True:
            tickets = live_tickets(queue_dir, verified, alive=alive, is_guard=is_guard)
            if not tickets or tickets[0] == ticket:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return time.monotonic() - t0
                except BlockingIOError:
                    pass
                pause = 0.02
                ahead = 0
            else:
                pause = 0.2
                ahead = tickets.index(ticket) if ticket in tickets else len(tickets)
            if ahead_prev is not None and ahead < ahead_prev and wait_s > 0:
                deadline = time.monotonic() + wait_s
            ahead_prev = ahead
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(pause, max(0.01, deadline - time.monotonic())))
    finally:
        try:
            ticket.unlink()
        except OSError:
            pass


def terminate_tree(proc):
    try:
        descendants = tree_pids(process_table(), proc.pid)
    except Exception:
        descendants = {proc.pid}
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):  # macOS: EPERM for an emptied group
        pass
    # Also cover workers which created their own process groups.
    for pid in descendants - {proc.pid}:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):  # macOS: EPERM for an emptied group
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):  # macOS: EPERM for an emptied group
        pass
    for pid in descendants - {proc.pid}:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):  # macOS: EPERM for an emptied group
            pass
    proc.wait()


def wait_for_recovery(reserve, host_limit, max_seconds, state, state_path, fit_limit=None):
    """Three consecutive safe readings, 5 s apart, within max_seconds. Returns True, False or None
    (telemetry lost). With fit_limit, a reading where that cap fits under the ceiling is also safe."""
    safe = 0
    t_end = time.monotonic() + max_seconds
    while True:
        try:
            _, available, used = host_memory()
        except Exception:
            return None
        state['last_host_used_bytes'] = used
        ok = recovered(available, used, reserve, host_limit) or (
            fit_limit is not None and fits(available, used, reserve, host_limit, fit_limit))
        safe = safe + 1 if ok else 0
        if safe >= 3:
            return True
        if time.monotonic() >= t_end:
            atomic_json(state_path, state)
            return False
        time.sleep(5)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--limit-gb', type=float, default=180, help='RSS stop threshold, never above 200 decimal GB')
    p.add_argument('--host-used-limit-gb', type=float, default=HOST_USED_CEILING_GB,
                   help='stop when whole-machine RAM use reaches this, never above 490 decimal GB')
    p.add_argument('--reserve-gb', type=float, default=32, help='minimum immediately available host RAM')
    p.add_argument('--interval', type=float, default=0.5)
    p.add_argument('--cooldown', type=float, default=60, help='seconds before the first restart; doubles per restart')
    p.add_argument('--recover-wait', type=float, default=900, help='max seconds to wait for host memory to recover')
    p.add_argument('--max-restarts', type=int, default=2)
    p.add_argument('--state', type=Path, default=ROOT / 'runs' / 'memory_guard.json')
    p.add_argument('--wait-lock', type=float, default=0, help='seconds to wait for the project-wide experiment lock without queue progress; restarts whenever a waiter ahead leaves (0 = fail fast with exit 75)')
    p.add_argument('--priority', action='store_true',
                   help='queue ahead of normal waiters (FIFO among priority waiters); never preempts the holder')
    p.add_argument('--priority-level', type=int, default=0, choices=(0, 1, 2, 3),
                   help='higher levels queue ahead of lower ones (--priority = level 1)')
    p.add_argument('--start-if-fits', action='store_true',
                   help='first launch may proceed when machine use + --limit-gb <= the host ceiling '
                        '(user rule 2026-10-26); restarts after a stop still wait for the 90%% line')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.command and a.command[0] == '--':
        a.command = a.command[1:]
    if (not a.command or not (0 < a.limit_gb <= PROJECT_CEILING_GB) or not (0 < a.host_used_limit_gb <= HOST_USED_CEILING_GB)
            or a.reserve_gb < 0 or not 0.05 <= a.interval <= 2 or a.cooldown < 0 or a.recover_wait < 0
            or a.max_restarts < 0 or a.wait_lock < 0):
        p.error('Require command, 0<limit<=200 GB, 0<host-used-limit<=490 GB, reserve>=0, 0.05<=interval<=2, '
                'cooldown>=0, recover-wait>=0, restarts>=0')
    (ROOT / 'runs').mkdir(exist_ok=True)
    lock = open(ROOT / 'runs' / 'experiment.lock', 'a')
    waited = acquire_lock(lock, a.wait_lock, info={'pid': os.getpid(), 'state': str(a.state), 'command': a.command,
                                                   'queued_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                                   'priority': max(int(a.priority), a.priority_level)},
                          priority=max(int(a.priority), a.priority_level))
    if waited is None:
        p.exit(75, 'Another guarded experiment is running; run projects serially.\n')
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    host_limit = a.host_used_limit_gb * GB
    state = {'command': a.command, 'cwd': str(Path.cwd()), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'status': 'preflight', 'attempts': [], 'limit_bytes': int(a.limit_gb * GB), 'reserve_bytes': int(a.reserve_gb * GB),
             'host_used_limit_bytes': int(host_limit), 'lock_wait_seconds': round(waited, 1)}
    atomic_json(a.state, state)
    try:
        process_table()
        total, available, used, detail = host_memory(detail=True)
    except Exception as e:
        state.update(status='telemetry_unavailable', error=str(e))
        atomic_json(a.state, state)
        p.exit(78, 'Memory telemetry unavailable. No job started; use an environment with ps/sysctl access.\n')
    limit = min(a.limit_gb * GB, total * 0.65)
    reserve = max(a.reserve_gb * GB, total * 0.08)
    state.update(physical_bytes=total, effective_limit_bytes=int(limit), effective_reserve_bytes=int(reserve),
                 host_at_start=dict(detail, used_bytes=used, available_bytes=available))
    try:
        for attempt in range(a.max_restarts + 1):
            try:
                _, available, used = host_memory()
            except Exception as e:
                state.update(status='telemetry_unavailable', error=str(e)); atomic_json(a.state, state); return 78
            fit_limit = limit if (a.start_if_fits and attempt == 0) else None
            state['start_rule'] = 'start_if_fits' if fit_limit is not None else 'recovered'
            if not (recovered(available, used, reserve, host_limit)
                    or (fit_limit is not None and fits(available, used, reserve, host_limit, fit_limit))):
                # Never launch onto a pressured machine: wait (bounded) for it to calm down first.
                state.update(status='waiting_for_memory', last_host_used_bytes=used); atomic_json(a.state, state)
                ok = wait_for_recovery(reserve, host_limit, a.recover_wait, state, a.state, fit_limit)
                if ok is None:
                    state.update(status='telemetry_unavailable'); atomic_json(a.state, state); return 78
                if not ok:
                    state.update(status='memory_not_recovered'); atomic_json(a.state, state); return 75
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
                       MEMGUARD_WORKERS='1', MEMGUARD_BATCH_SCALE=str(0.5 ** attempt), MEMGUARD_RESTART=str(attempt),
                       MEMGUARD_DELAY_SECONDS=str(0.02 * (2 ** attempt - 1)))
            record = {'attempt': attempt, 'batch_scale': 0.5 ** attempt, 'peak_tree_rss_bytes': 0, 'peak_host_used_bytes': 0}
            state['attempts'].append(record)
            proc = subprocess.Popen(a.command, env=env, start_new_session=True)
            state.update(status='running', child_pid=proc.pid)
            atomic_json(a.state, state)
            reason = None
            try:
                while proc.poll() is None:
                    try:
                        rss = tree_rss(process_table(), proc.pid)
                        _, available, used = host_memory()
                    except Exception as e:
                        reason = 'telemetry_unavailable'; record['error'] = str(e); break
                    record['peak_tree_rss_bytes'] = max(record['peak_tree_rss_bytes'], rss)
                    record['peak_host_used_bytes'] = max(record['peak_host_used_bytes'], used)
                    record['last_available_bytes'] = available
                    atomic_json(a.state, state)
                    reason = stop_reason(rss, available, limit, reserve, used, host_limit)
                    if reason:
                        break
                    time.sleep(a.interval)
            except KeyboardInterrupt:
                reason = 'interrupted'
            if reason:
                terminate_tree(proc)
                record.update(stop_reason=reason, returncode=proc.returncode)
                state.update(status='stopped', reason=reason)
                atomic_json(a.state, state)
                if reason in ('telemetry_unavailable', 'interrupted') or attempt == a.max_restarts:
                    return 130 if reason == 'interrupted' else 75
                state['status'] = 'cooldown'; atomic_json(a.state, state)
                time.sleep(a.cooldown * (2 ** attempt))
                ok = wait_for_recovery(reserve, host_limit, a.recover_wait, state, a.state)
                if ok is None:
                    state.update(status='telemetry_unavailable'); atomic_json(a.state, state); return 78
                if not ok:
                    state.update(status='memory_not_recovered'); atomic_json(a.state, state); return 75
                continue
            record['returncode'] = proc.wait()
            # A successful worker must not leave detached work behind.
            try:
                surviving = tree_pids(process_table(), proc.pid) - {proc.pid}
            except Exception:
                surviving = set()
            if surviving:
                terminate_tree(proc)
                state.update(status='failed', reason='orphan_workers'); atomic_json(a.state, state); return 70
            state.update(status='completed' if proc.returncode == 0 else 'failed', returncode=proc.returncode,
                         finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            atomic_json(a.state, state)
            return proc.returncode
    except KeyboardInterrupt:
        state.update(status='stopped', reason='interrupted'); atomic_json(a.state, state)
        return 130
    return 75

if __name__ == '__main__':
    sys.exit(main())
