"""EditStream-RAG git streams (K8S: kubernetes/website content/en/docs/**/*.md; CPY: cpython Doc/**/*.rst).

For each hash-sampled file with >= MIN_COMMITS changes on main's FIRST-PARENT chain (following renames; merge commits
diffed against their first parent, so a merged PR is one main-branch state change), emit every version's text in
first-parent order. Text-identical consecutive versions are dropped and counted. Every consecutive pair is a real
main-branch transition (older state is an ancestor of the newer one); a sample is verified with merge-base.
(Build session 1 used `--no-merges`, which interleaved concurrent side-branch commits: review finding B4-G1.)

usage: python -m deltachunk.data.git_extract {k8s|cpy} [--n 300] [--procs 8]
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time
from multiprocessing import Pool

import pandas as pd

P3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SPECS = {
    "k8s": dict(repo="data/raw/k8s", prefix="content/en/docs/", ext=".md", salt="editstream-2026-K8S"),
    "cpy": dict(repo="data/raw/cpython", prefix="Doc/", ext=".rst", salt="editstream-2026-CPY"),
}
MIN_COMMITS = 20


def salted(salt: str, key: str) -> str:
    return hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()


ENV = dict(os.environ)   # partial clone: rename detection on merge diffs may lazily fetch a few old blobs from GitHub


def git(repo: str, *args: str, inp: bytes | None = None, check: bool = True) -> bytes:
    return subprocess.run(["git", "-C", repo, *args], input=inp, capture_output=True, check=check, env=ENV).stdout


def is_ancestor(repo: str, a: str, b: str) -> bool:
    return subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", a, b], capture_output=True, env=ENV).returncode == 0


def file_history(repo: str, path: str):
    """[(sha, ct, path_at_commit)] oldest first, following renames, main's first-parent chain only (merges diffed
    against their first parent)."""
    out = git(repo, "log", "--first-parent", "--diff-merges=first-parent", "--follow", "--format=@@%H %ct",
              "--name-only", "--", path).decode()
    hist, cur = [], None
    for line in out.splitlines():
        if line.startswith("@@"):
            sha, ct = line[2:].split()
            cur = [sha, int(ct), None]; hist.append(cur)
        elif line.strip() and cur is not None and cur[2] is None:
            cur[2] = line.strip()
    return [tuple(h) for h in reversed(hist) if h[2]]


def read_blobs(repo: str, specs: list[str]) -> list[str | None]:
    """git cat-file --batch for 'sha:path' specs; None for missing (deleted at that commit)."""
    out = git(repo, "cat-file", "--batch", inp=("\n".join(specs) + "\n").encode())
    res, i = [], 0
    for _ in specs:
        nl = out.index(b"\n", i); header = out[i:nl].decode(); i = nl + 1
        if header.endswith("missing") or header.endswith("ambiguous"):
            res.append(None); continue
        size = int(header.split()[2]); res.append(out[i:i + size].decode("utf-8", errors="replace")); i += size + 1
    return res


def process(args):
    repo, path = args
    try:
        hist = file_history(repo, path)
    except subprocess.CalledProcessError:
        return path, None, -1
    if len(hist) < MIN_COMMITS:
        return path, None, len(hist)
    texts = read_blobs(repo, [f"{sha}:{p}" for sha, _, p in hist])
    rows, prev, dropped = [], None, 0
    import random
    rng = random.Random(salted("anc", path))
    pairs = list(zip(hist, hist[1:]))
    anc_bad = sum(not is_ancestor(repo, x[0], y[0]) for x, y in rng.sample(pairs, min(5, len(pairs))))
    for (sha, ct, p), t in zip(hist, texts):
        if t is None:
            continue  # deletion commit
        if t == prev:
            dropped += 1; continue
        rows.append(dict(doc_id=path, rev_idx=len(rows), rev_id=sha, ts=ct, path_at_rev=p, text=t)); prev = t
    return path, (rows, dropped, anc_bad, min(5, len(pairs))), len(hist)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stream", choices=list(SPECS))
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args(); s = SPECS[a.stream]; repo = os.path.join(P3, s["repo"])
    t0 = time.time()
    files = [f for f in git(repo, "ls-tree", "-r", "--name-only", "HEAD", "--", s["prefix"]).decode().splitlines() if f.endswith(s["ext"])]
    files.sort(key=lambda f: salted(s["salt"], f))
    chosen, stats, all_rows = [], dict(candidates=len(files), examined=0, too_few_commits=0, identical_dropped=0,
                                       ancestry_checked=0, ancestry_violations=0, history="first-parent"), []
    with Pool(a.procs) as pool:
        for path, res, ncommits in pool.imap(process, [(repo, f) for f in files], chunksize=4):
            stats["examined"] += 1
            stats["log_errors"] = stats.get("log_errors", 0) + (ncommits == -1)
            if res is None or len(res[0]) < MIN_COMMITS:
                stats["too_few_commits"] += 1; continue
            rows, dropped, anc_bad, anc_n = res
            if len(chosen) < a.n:
                chosen.append(path); all_rows.extend(rows); stats["identical_dropped"] += dropped
                stats["ancestry_checked"] += anc_n; stats["ancestry_violations"] += anc_bad
            if len(chosen) >= a.n:
                pool.terminate(); break
    df = pd.DataFrame(all_rows)
    out = os.path.join(P3, "data", "streams", f"{a.stream}.parquet"); os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    tags = {}
    if a.stream == "cpy":
        with open(os.path.join(P3, "results", "cpy_release_tags.json")) as f:
            tags = json.load(f)
    stats.update(docs=len(chosen), versions=len(df), head=git(repo, "rev-parse", "HEAD").decode().strip(),
                 seconds=round(time.time() - t0, 1), procs=a.procs, min_commits=MIN_COMMITS, salt=s["salt"], n_release_tags=len(tags))
    json.dump(stats, open(os.path.join(P3, "results", f"e0_{a.stream}_stats.json"), "w"), indent=1)
    if tags:
        json.dump(tags, open(os.path.join(P3, "data", "streams", "cpy_tags.json"), "w"))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
