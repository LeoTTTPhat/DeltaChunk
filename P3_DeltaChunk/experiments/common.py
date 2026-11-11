"""Shared helpers for the experiment drivers: stream loading, run metadata, calibration lookup."""
from __future__ import annotations

import json, os, platform, re, subprocess, sys, time

P3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(P3, "src"))
RESULTS = os.path.join(P3, "results")
RAW = os.path.join(RESULTS, "raw")
STREAMS = {"wiki": "wiki.parquet", "k8s": "k8s.parquet", "cpy": "cpy.parquet"}
THREADS = int(os.environ.get("P3_THREADS", "8"))
# Thread control (binding <= 10 threads/job): the `tokenizers` Rayon pool defaults to one thread per core. Set before
# tokenizers is imported; multi-process drivers call worker_threads() in each worker (1 thread per worker).
os.environ.setdefault("RAYON_NUM_THREADS", str(THREADS))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
# E-phase bug fix (runner session 1): memguard sets MEMGUARD_SHRINK="0" on the first attempt and "1" on retries, but
# e1.py and deltachunk.embed test the variable for truthiness, so every memguarded run shrank (E1 at procs // 2, the
# embedder at batch 64 // 4 = 16). Normalise "0" to unset here (src/ is frozen; checkpoint hashes stay valid).
if os.environ.get("MEMGUARD_SHRINK") == "0":
    del os.environ["MEMGUARD_SHRINK"]


def worker_threads():
    """Pool initializer helper: single-threaded tokenizers/OMP/torch inside a worker process."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"; os.environ["RAYON_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(1)


def code_hash(*extra) -> str:
    """Hash of src/**/*.py + the given extra objects (driver config): checkpoint directories are keyed by it."""
    import glob, hashlib
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(P3, "src", "**", "*.py"), recursive=True)):
        h.update(f.encode()); h.update(open(f, "rb").read())
    h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def checkpoint_dir(path: str, cfg_hash: str) -> str:
    """Create `path` for resumable per-document checkpoints. If it holds checkpoints from another code/config hash,
    move them aside (path.stale-<hash>) so stale and new rows are never mixed (review finding B4-m: resume keying)."""
    os.makedirs(path, exist_ok=True)
    f = os.path.join(path, "_config_hash")
    old = open(f).read().strip() if os.path.exists(f) else None
    has_ckpt = any(x.endswith(".parquet") for x in os.listdir(path))
    if has_ckpt and old != cfg_hash:
        os.rename(path, f"{path}.stale-{old or 'unknown'}-{int(time.time())}"); os.makedirs(path)
    open(f, "w").write(cfg_hash)
    return path


def load_stream(name: str, docs: int | None = None, columns=None):
    """Versions sorted by (doc_id, rev_idx). `ts` is made monotone within each document (cummax over rev_idx; the raw
    value is kept in `ts_raw`): git commit times and wiki timestamps can go backwards, and every timestamp-ordered
    replay / sync-interval coalescing must keep each document's revision order (review finding B4-T1).
    Empty (whitespace-only) versions are dropped (wiki revision-deleted text and blanking are indistinguishable in the
    dump), then text-identical consecutive versions are dropped again; counts are in df.attrs["dropped"]."""
    import pandas as pd
    df = pd.read_parquet(os.path.join(P3, "data", "streams", STREAMS[name]), columns=columns)
    df = df.sort_values(["doc_id", "rev_idx"])
    dropped = {}
    if "text" in df.columns:
        empty = df["text"].str.strip().eq("")
        dropped["empty"] = int(empty.sum()); df = df[~empty]
        same = df["text"].eq(df.groupby("doc_id", sort=False)["text"].shift())
        dropped["identical_after_empty"] = int(same.sum()); df = df[~same]
    if "ts" in df.columns:
        df["ts_raw"] = df["ts"]; df["ts"] = df.groupby("doc_id", sort=False)["ts"].cummax()
    df.attrs["dropped"] = dropped
    if docs:
        keep = sorted(df.doc_id.unique(), key=lambda d: _h(d))[:docs]
        df = df[df.doc_id.isin(set(keep))]
    return df


def _h(s: str) -> str:
    import hashlib
    return hashlib.sha256(f"editstream-order:{s}".encode()).hexdigest()


def doc_groups(df):
    """[(doc_id, [dict(rev_idx, ts, text)])] in salted-hash doc order (WIKI-100 = first 100)."""
    out = []
    for d, g in df.groupby("doc_id", sort=False):
        out.append((d, g[["rev_idx", "ts", "text"]].to_dict("records")))
    out.sort(key=lambda x: _h(str(x[0])))
    return out


def release_ts(stream: str) -> list[int] | None:
    if stream != "cpy":
        return None
    tags = json.load(open(os.path.join(P3, "data", "streams", "cpy_tags.json")))
    return sorted(ts for n, ts in tags.items() if re.fullmatch(r"v3\.\d+\.\d+", n))


def calibration(stream: str | None = None) -> dict:
    """{T: {chunker: params}}. Calibration is per stream (review finding B4-C2: pooled calibration left chunkers up to
    1.6x apart in size within a stream); `stream` selects that stream's parameters."""
    f = os.path.join(RESULTS, "calibration.json")
    if not os.path.exists(f):
        return {}
    d = json.load(open(f))
    if d.get("per_stream"):
        if stream is None:
            raise ValueError("calibration is per stream: pass the stream")
        return {T: P[stream] for T, P in d["params"].items() if stream in P}
    return d["params"]


def load_snapshot() -> dict:
    try:
        ps = subprocess.run(["ps", "-Ao", "pcpu,rss,comm", "-r"], capture_output=True, text=True).stdout.splitlines()[1:11]
    except Exception:
        ps = []
    mw = {}
    try:
        mw = json.load(open(os.path.join(os.path.dirname(P3), "logs", "memwatch_status.json")))
        mw = {k: mw[k] for k in ("ts", "project_gb", "system_gb") if k in mw}
    except Exception:
        pass
    return {"top10": [" ".join(x.split()[:2]) + " " + os.path.basename(x.split()[-1]) for x in ps if x.split()], "memwatch": mw}


def run_meta(**extra) -> dict:
    import importlib.metadata as md
    vers = {}
    for p in ("numpy", "pandas", "langchain-text-splitters", "kara-toolkit", "steadlith", "fastcdc", "tokenizers",
              "sentence-transformers", "transformers", "torch", "hnswlib", "faiss-cpu"):
        try: vers[p] = md.version(p)
        except Exception: pass
    thr = dict(P3_THREADS=THREADS, RAYON_NUM_THREADS=os.environ.get("RAYON_NUM_THREADS"),
               TOKENIZERS_PARALLELISM=os.environ.get("TOKENIZERS_PARALLELISM"), OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"))
    if "torch" in sys.modules:
        thr["torch_num_threads"] = sys.modules["torch"].get_num_threads()
    return dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.machine(), python=platform.python_version(),
                versions=vers, threads=thr, load=load_snapshot(), **extra)


# ------------------------------------------------------------------------------------------------------------------
# E-phase bug fix (runner session 1, Nov 9 ~17:50), applied here and not in src/ because the E1/E3/E8 checkpoints are
# keyed by a hash of src/ (feature freeze). Steadlith rejects a version whose whitespace-delimited "word" has a token
# count of 0; the bge tokenizer gives 0 tokens to words made only of characters it drops (U+200E LEFT-TO-RIGHT MARK,
# control characters such as U+0003). 4 of 1,000 WIKI documents (31493, 9594, 828348, 1089984) failed in E1 for every
# chunker because of it, and 828348's final version would abort E4's STL corpus. Fix: when Steadlith raises that
# error, the version is re-chunked with the token counter max(1, ntok(word)) (such a word counts as one token); the
# number of such versions is counted in SteadlithAdapter.zero_token_fallbacks. Behaviour is unchanged for every
# version without zero-token words.
def _patch_steadlith_zero_token_words():
    from deltachunk.chunkers import stateful as _st
    if getattr(_st.SteadlithAdapter.chunk, "_zero_tok_patched", False):
        return
    orig = _st.SteadlithAdapter.chunk

    def safe_ntok(w):
        return max(1, _st.ntok(w))

    def chunk(self, v):
        try:
            return orig(self, v)
        except ValueError as e:
            if "positive count" not in str(e):
                raise
        from steadlith.chunk import CDCParams, chunk_text
        _st.SteadlithAdapter.zero_token_fallbacks += 1
        p = CDCParams(min_tokens=self.min_tokens, max_tokens=self.max_tokens, snap_to_boundaries=self.snap)
        text = v.text
        try:
            res = chunk_text(text, p, token_counter=safe_ntok, tokenizer_id="bge-small-en-v1.5")
        except ValueError:   # a word longer than max_tokens as well: same split fallback as the adapter
            text = self._split_long_words(text); _st.SteadlithAdapter.fallbacks += 1
            res = chunk_text(text, p, token_counter=safe_ntok, tokenizer_id="bge-small-en-v1.5")
        out = [_st.Chunk(c.page_content, c.metadata.get("start_offset", -1), c.metadata.get("end_offset", -1)) for c in res.chunks]
        return _st.enforce_max(text, out)
    chunk._zero_tok_patched = True
    _st.SteadlithAdapter.chunk = chunk
    _st.SteadlithAdapter.zero_token_fallbacks = 0


_patch_steadlith_zero_token_words()


# E-phase bug fix (runner session 2, Nov 9 ~20:15; same reason for living here). LangChain's RecursiveCharacterTextSplitter
# with chunk_overlap=0 still repeats a zero-length split at the start of the next chunk: `_merge_splits` keeps popping
# the current pieces only `while total > chunk_overlap`, and a piece with 0 bge tokens (U+200E) leaves total = 0. WIKI
# doc 31493 at T=320 (REC, DIFF[REC]; 5 versions) produced the pieces '...References\n\n\u200e' and
# '\u200e\n\nzh-min-nan:LEGO', and `locate` (no-overlap search from the previous end) raised, so E1 had no rows for this
# document. Fix: when a piece is not found in order, a prefix that repeats the previous piece's tail and has 0 tokens
# is dropped (with the whitespace after it) and the rest is located; the count is kept in base.locate_zero_tok_dups.
# Pieces are otherwise located exactly as before (the error path only).
def _patch_locate_zero_token_dup():
    from deltachunk.chunkers import base as _b
    if getattr(_b.locate, "_zero_tok_patched", False):
        return

    def locate(text, pieces, base=0, overlap=False):
        out, pos = [], 0
        for p in pieces:
            i = text.find(p, pos)
            if i < 0 and not overlap and pos > 0:
                for k in range(min(len(p), pos), 0, -1):
                    if text[pos - k:pos] == p[:k] and _b.ntok(p[:k]) == 0:
                        q = p[k:].lstrip()
                        j = text.find(q, pos) if q else -1
                        if j >= 0 and not text[pos:j].strip():
                            p, i = q, j; _b.locate_zero_tok_dups += 1
                        break
            if i < 0:
                raise ValueError(f"locate: piece not found in order at {pos}: {p[:60]!r}")
            out.append(_b.Chunk(p, base + i, base + i + len(p))); pos = i + 1 if overlap else i + len(p)
        return out
    locate._zero_tok_patched = True
    _b.locate = locate
    _b.locate_zero_tok_dups = 0


_patch_locate_zero_token_dup()


def cap_procs(n: int) -> int:
    """Runner session 1: an operator cap on worker processes for already-queued multi-process steps (E1/E2/E3/E8),
    read at start from env P3_MAX_PROCS or the file logs/ephase/MAX_PROCS (one integer). Machine-wide memory pressure
    (watchdog kills at 470 GB with 1-hour cooldowns) made the 8-process peaks (~10-15 GB) too large. Not part of any
    checkpoint hash; results do not depend on the number of processes."""
    cap = os.environ.get("P3_MAX_PROCS")
    f = os.path.join(P3, "logs", "ephase", "MAX_PROCS")
    if cap is None and os.path.exists(f):
        cap = open(f).read().strip()
    try:
        return max(1, min(n, int(cap))) if cap else n
    except ValueError:
        return n
