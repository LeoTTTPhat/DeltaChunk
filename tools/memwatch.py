#!/usr/bin/env python3
"""memwatch: project-wide memory watchdog for the ICDE27 project (macOS).

Every --interval seconds it finds all "project processes":
  * processes whose current working directory is inside PROJECT_ROOT,
  * their descendants (even if they chdir elsewhere),
  (Ollama's own server/runners are shared with other projects and are NOT counted or touched.)
It measures each process's physical footprint (proc_pid_rusage, includes Metal/GPU
unified-memory allocations; falls back to RSS) and sums them.

Policy (defaults):
  total >= --soft-gb (180)  -> write logs/MEMORY_PRESSURE (memguard will not START new jobs)
  total >= --cap-gb  (200)  -> stop the largest killable job group (SIGTERM, then SIGKILL after 15 s),
                               write logs/MEMORY_COOLDOWN (jobs restart only after it expires,
                               cooldown doubles on repeated trips = "restart slowly")
  system used >= --sys-cap-gb (470) AND project >= --sys-min-project-gb (20) -> same kill action (machine safety,
                               512 GB box shared with other projects; we only act when our share is material).
                               It must hold for two consecutive samples, and after such a kill the next one waits
                               --sys-grace (60 s) so freed memory and brief spikes from other projects can settle;
                               at or above --sys-hard-gb (490) it acts at once regardless of project share.
  system used >= --sys-warn-gb (430) -> warning event only
Never kills Claude/Codex agent processes, terminals, or anything outside the project.
Events are appended to logs/memwatch_events.jsonl; status every minute to logs/memwatch.log.
"""
import argparse, ctypes, ctypes.util, json, os, signal, subprocess, sys, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(PROJECT_ROOT, "logs")
PRESSURE = os.path.join(LOGDIR, "MEMORY_PRESSURE")
COOLDOWN = os.path.join(LOGDIR, "MEMORY_COOLDOWN")
EVENTS = os.path.join(LOGDIR, "memwatch_events.jsonl")
STATUS = os.path.join(LOGDIR, "memwatch_status.json")
DISK_PRESSURE = os.path.join(LOGDIR, "DISK_PRESSURE")
DISK_CRITICAL = os.path.join(LOGDIR, "DISK_CRITICAL")
import shutil
import re
# never kill (they are agents, terminals, app helpers, the Ollama server itself, or this watchdog)
PROTECTED_RE = re.compile(r"/Claude\.app/|(^|/)claude( |$)|(^|/)codex( |$)|memwatch\.py|memguard\.py|shell-snapshots|"
                          r"Ollama\.app/Contents/MacOS/Ollama|ollama serve|/Terminal\.app/|iTerm|tmux|^-|^login|launchd")
# Ollama is a SHARED service used by other projects: its server and runners are never project members (2026-11-09 fix:
# counting them made the watchdog stop runners that belonged to another project). Our own llama-servers are started from
# inside this folder and are members by cwd.

libc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
_buf_vp = ctypes.create_string_buffer(2352)
_buf_ru = ctypes.create_string_buffer(512)

def cwd_of(pid):
    n = libc.proc_pidinfo(pid, 9, ctypes.c_uint64(0), _buf_vp, 2352)  # PROC_PIDVNODEPATHINFO
    if n <= 0: return None
    return _buf_vp.raw[152:152 + 1024].split(b"\0", 1)[0].decode("utf-8", "replace")

def footprint(pid):
    if libc.proc_pid_rusage(pid, 2, _buf_ru) != 0:  # RUSAGE_INFO_V2
        return None
    return int.from_bytes(_buf_ru.raw[72:80], "little")  # ri_phys_footprint

def procs():
    out = subprocess.run(["ps", "-axww", "-o", "pid=,ppid=,rss=,command="], capture_output=True, text=True).stdout
    P = {}
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4: continue
        pid, ppid, rss = int(parts[0]), int(parts[1]), int(parts[2]) * 1024
        P[pid] = {"pid": pid, "ppid": ppid, "rss": rss, "comm": parts[3]}
    return P

def sys_used_bytes():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page, v = 16384, {}
    for line in out.splitlines():
        if "page size of" in line: page = int(line.split("page size of")[1].split()[0])
        elif ":" in line:
            k, x = line.split(":", 1)
            try: v[k.strip()] = int(x.strip().rstrip("."))
            except ValueError: pass
    required = ("Anonymous pages", "Pages purgeable", "Pages wired down", "Pages occupied by compressor")
    if any(k not in v for k in required):
        raise ValueError("vm_stat did not report required memory counters")
    app = max(0, v["Anonymous pages"] - v["Pages purgeable"])
    return (app + v["Pages wired down"] + v["Pages occupied by compressor"]) * page

def is_protected(p):
    return bool(PROTECTED_RE.search(p["comm"])) or p["pid"] in (os.getpid(), 1)

def classify(P):
    member = {}
    for pid, p in P.items():
        c = cwd_of(pid)
        p["cwd"] = c
        member[pid] = bool(c and (c == PROJECT_ROOT or c.startswith(PROJECT_ROOT + "/")))
    # descendants of members are members
    changed = True
    while changed:
        changed = False
        for pid, p in P.items():
            if not member[pid] and member.get(p["ppid"]) and p["ppid"] != 1 and not re.search(r"ollama serve|Ollama\.app/Contents/MacOS/Ollama", p["comm"]):
                member[pid] = True; changed = True
    for pid, p in P.items():
        if member[pid]:
            f = footprint(pid)
            p["mem"] = f if f is not None else p["rss"]
    return {pid for pid, m in member.items() if m}

def kill_root(pid, P, members):
    cur = pid
    while True:
        par = P.get(cur, {}).get("ppid")
        if par in members and par in P and not is_protected(P[par]): cur = par
        else: return cur

def log_event(ev):
    ev["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(EVENTS, "a") as f: f.write(json.dumps(ev) + "\n")
    print(json.dumps(ev), flush=True)

def group_tree(root, P):
    kids = {}
    for pid, p in P.items(): kids.setdefault(p["ppid"], []).append(pid)
    out, st = [], [root]
    while st:
        x = st.pop(); out.append(x); st.extend(kids.get(x, []))
    return out

def terminate(pids):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try: os.kill(pid, sig)
            except (ProcessLookupError, PermissionError): pass
        if sig == signal.SIGTERM:
            for _ in range(30):
                time.sleep(0.5)
                if not any(alive(p) for p in pids): return
def alive(pid):
    try: os.kill(pid, 0); return True
    except (ProcessLookupError, PermissionError): return False

def system_trip(sysu, total, now, state, a, G=2 ** 30):
    if sysu >= a.sys_hard_gb * G: return True
    over = sysu >= a.sys_cap_gb * G and total >= a.sys_min_project_gb * G
    state["hits"] = state.get("hits", 0) + 1 if over else 0
    if not over: return False
    last = state.get("last_kill")
    return state["hits"] >= 2 and (last is None or now - last >= a.sys_grace)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-gb", type=float, default=200)
    ap.add_argument("--soft-gb", type=float, default=180)
    ap.add_argument("--sys-cap-gb", type=float, default=470)
    ap.add_argument("--sys-warn-gb", type=float, default=430)
    ap.add_argument("--sys-min-project-gb", type=float, default=20)
    ap.add_argument("--sys-hard-gb", type=float, default=490)
    ap.add_argument("--sys-grace", type=float, default=60)
    ap.add_argument("--interval", type=float, default=5)
    ap.add_argument("--disk-warn-gb", type=float, default=25)
    ap.add_argument("--disk-crit-gb", type=float, default=12)
    ap.add_argument("--cooldown", type=float, default=120)
    ap.add_argument("--once", action="store_true", help="print accounting once and exit (no kills)")
    a = ap.parse_args()
    os.makedirs(LOGDIR, exist_ok=True)
    if not a.once:
        for f in (PRESSURE, COOLDOWN):  # flags left by a previous (dead) watchdog are stale
            if os.path.exists(f): os.remove(f)
        open(os.path.join(LOGDIR, "memwatch.pid"), "w").write(str(os.getpid()))
        log_event({"event": "start", "pid": os.getpid(), "cap_gb": a.cap_gb, "soft_gb": a.soft_gb, "sys_cap_gb": a.sys_cap_gb, "sys_warn_gb": a.sys_warn_gb, "sys_min_project_gb": a.sys_min_project_gb, "sys_hard_gb": a.sys_hard_gb, "sys_grace": a.sys_grace})
    cooldown, last_status, last_trip, last_warn = a.cooldown, 0, 0, 0
    sysstate = {}
    while True:
        P = procs(); members = classify(P)
        total = sum(P[m].get("mem", 0) for m in members)
        groups = {}
        for m in members:
            if is_protected(P[m]): continue
            r = kill_root(m, P, members); groups.setdefault(r, 0); groups[r] += P[m].get("mem", 0)
        top = sorted(groups.items(), key=lambda kv: -kv[1])[:8]
        sysu = sys_used_bytes()
        G = 2 ** 30
        free_gb = shutil.disk_usage(PROJECT_ROOT).free / G
        st = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "project_gb": round(total / G, 2), "system_gb": round(sysu / G, 2),
              "disk_free_gb": round(free_gb, 1),
              "cap_gb": a.cap_gb, "soft_gb": a.soft_gb,
              "top_groups": [{"root": r, "comm": P[r]["comm"][:100], "gb": round(b / G, 2)} for r, b in top]}
        if a.once:
            print(json.dumps(st, indent=1)); return
        with open(STATUS, "w") as f: json.dump(st, f, indent=1)
        if total >= a.soft_gb * G:
            if not os.path.exists(PRESSURE): log_event({"event": "pressure_on", **st})
            open(PRESSURE, "w").write(json.dumps(st))
        elif os.path.exists(PRESSURE) and total < (a.soft_gb - 10) * G:
            os.remove(PRESSURE); log_event({"event": "pressure_off", "project_gb": st["project_gb"]})
        if sysu >= a.sys_warn_gb * G and time.time() - last_warn > 600:
            last_warn = time.time(); log_event({"event": "system_warn", "project_gb": st["project_gb"], "system_gb": st["system_gb"]})
        proj_over = total >= a.cap_gb * G
        sys_over = system_trip(sysu, total, time.time(), sysstate, a, G)
        if (proj_over or sys_over) and top:
            root, b = top[0]
            pids = [pid for pid in group_tree(root, P) if pid in members and not is_protected(P[pid])]
            if time.time() - last_trip < 3600: cooldown = min(cooldown * 2, 3600)
            else: cooldown = a.cooldown
            last_trip = time.time()
            if not proj_over: sysstate.update(last_kill=time.time(), hits=0)
            log_event({"event": "kill", "reason": "project_cap" if proj_over else "system_cap",
                       "root": root, "comm": P[root]["comm"], "cwd": P[root].get("cwd"), "group_gb": round(b / G, 2),
                       "pids": pids, "cooldown_s": cooldown, **{k: st[k] for k in ("project_gb", "system_gb")}})
            open(COOLDOWN, "w").write(str(time.time() + cooldown))
            terminate(pids)
        if os.path.exists(COOLDOWN):
            try:
                if time.time() > float(open(COOLDOWN).read().strip() or 0): os.remove(COOLDOWN); log_event({"event": "cooldown_over"})
            except (ValueError, FileNotFoundError): pass
        for flag, thr, name in ((DISK_PRESSURE, a.disk_warn_gb, "disk_pressure"), (DISK_CRITICAL, a.disk_crit_gb, "disk_critical")):
            if free_gb < thr:
                if not os.path.exists(flag): log_event({"event": name + "_on", "disk_free_gb": round(free_gb, 1)})
                open(flag, "w").write(str(round(free_gb, 1)))
            elif os.path.exists(flag) and free_gb > thr + 5:
                os.remove(flag); log_event({"event": name + "_off", "disk_free_gb": round(free_gb, 1)})
        if time.time() - last_status > 60:
            last_status = time.time()
            with open(os.path.join(LOGDIR, "memwatch.log"), "a") as f:
                f.write(f"{st['ts']} project={st['project_gb']}GB system={st['system_gb']}GB disk_free={st['disk_free_gb']}GB top={[(g['comm'][-30:], g['gb']) for g in st['top_groups'][:3]]}\n")
        time.sleep(a.interval)

if __name__ == "__main__":
    main()
