"""Round 2 (C4): band-only ablation of guarded CDC at matched mean size (count replay, no embeddings).

Arms (all on the same canonical unit layer and text as E1; S1 counts per revision):
  REC                 LangChain recursive splitter (reference)
  REC-e1              REC at E1's nominal T=128 parameter (fidelity check against results/e1_edits.parquet)
  GCDC-wide           sentence-unit threshold CDC, min guard T_min, max 480 tokens (E1's configuration)
  GCDC-narrow         same with max = 2 x min
  GCDC-wide-p10       GCDC-wide with sparse candidate cuts (unit-hash threshold p = 0.10 instead of 0.25)
  GCDC-narrow-p10     GCDC-narrow with p = 0.10 (more cuts forced by the max guard, as in Steadlith)
  BCDC-wide           byte FastCDC, min/max = avg/4 and 4 x avg (E1's configuration). A narrow-band BCDC is not run:
                      fastcdc 1.7.0 requires max >= 1,024 bytes, so max = 2 x min cannot reach ~135-token means.
  STL-narrow          Steadlith cdc-rabin, max = 2 x min, default masks (8/6 bits)       (E1's STL)
  STL+snap-narrow     same with boundary snapping                                         (E1's STL+snap)
  STL-narrow-m6       max = 2 x min, primary/backup masks 6/4 bits
  STL-wide-m6         max = 16 x min (capped at 480 tokens, so the ratio falls below 16 when min > 30), masks 6/4 bits
                      (a wide band needs a denser mask for the same mean)
  STL+snap-wide-m6    same with boundary snapping
  STL-wide            max = 16 x min (capped at 480), default masks (reported with its realized size)
Each size-parameterized arm is calibrated per stream to two target means (110 and 165 tokens) on the same 10%
calibration slice and bisection procedure as experiments/calibrate.py, so the common size of results/round2 (134/136/137
tokens) is bracketed and costs are interpolated with the matched-size rule (log-log between bracketing achieved sizes).
Steps:  calibrate -> results/round2/band_calibration.json
        replay    -> per-document checkpoints results/raw/r2_band/<stream>/, results/round2/band_edits.parquet
        analyze   -> results/round2/band_ablation.csv (+ paired document bootstrap against REC)
usage: PYTHONPATH=src .venv/bin/python experiments/r2_band.py calibrate|replay|analyze|all [--streams ...] [--docs N]
       [--procs 8] [--tag ""]
"""
import argparse, glob, hashlib, json, os, time, traceback
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from common import RAW, RESULTS, cap_procs, calibration, checkpoint_dir, code_hash, doc_groups, load_stream, run_meta, worker_threads

from deltachunk.chunkers import ByteCDC, GuardedCDC, Recursive, Version
from deltachunk.chunkers.base import Chunk, enforce_max
from deltachunk.text import ntok, ntok_many

OUT = os.path.join(RESULTS, "round2")
TARGETS = (110, 165)
COMMON = {"wiki": 134, "k8s": 136, "cpy": 137}     # results/round2/matched_size_spreads.csv (common size)
T_MAX = 480


class Steadlith:
    """Steadlith cdc-rabin with explicit band and mask bits (same text, token counter and long-word fallback as the
    E1 adapter; zero-token words count as one token, as in E1)."""

    def __init__(self, min_tokens, max_tokens, snap=False, primary=8, backup=6):
        self.min_tokens, self.max_tokens, self.snap, self.primary, self.backup = min_tokens, max_tokens, snap, primary, backup

    def chunk(self, v):
        from steadlith.chunk import CDCParams, chunk_text
        from deltachunk.chunkers.stateful import SteadlithAdapter
        if not v.text.strip():
            return []
        p = CDCParams(min_tokens=self.min_tokens, max_tokens=self.max_tokens, snap_to_boundaries=self.snap,
                      primary_mask_bits=self.primary, backup_mask_bits=self.backup)
        counter = lambda w: max(1, ntok(w))
        text = v.text
        try:
            res = chunk_text(text, p, token_counter=counter, tokenizer_id="bge-small-en-v1.5")
        except ValueError:
            text = SteadlithAdapter._split_long_words(self, text)
            res = chunk_text(text, p, token_counter=counter, tokenizer_id="bge-small-en-v1.5")
        out = [Chunk(c.page_content, c.metadata.get("start_offset", -1), c.metadata.get("end_offset", -1)) for c in res.chunks]
        return enforce_max(text, out)


def build(arm, x):
    """Chunker for `arm` at scale parameter x (the calibrated quantity)."""
    if arm in ("REC", "REC-e1"):
        return Recursive(int(x), 0)
    if arm == "GCDC-wide":
        return GuardedCDC(int(x), 0.25, T_MAX)
    if arm == "GCDC-narrow":
        return GuardedCDC(int(x), 0.25, min(T_MAX, 2 * int(x)))
    if arm == "GCDC-wide-p10":
        return GuardedCDC(int(x), 0.10, T_MAX)
    if arm == "GCDC-narrow-p10":
        return GuardedCDC(int(x), 0.10, min(T_MAX, 2 * int(x)))
    if arm == "BCDC-wide":
        return ByteCDC(int(x))
    lo = int(x)
    if arm in ("STL-narrow", "STL+snap-narrow"):
        return Steadlith(lo, min(T_MAX, 2 * lo), arm.startswith("STL+snap"))
    if arm == "STL-narrow-m6":
        return Steadlith(lo, min(T_MAX, 2 * lo), False, 6, 4)
    if arm in ("STL-wide-m6", "STL+snap-wide-m6"):
        return Steadlith(lo, min(T_MAX, 16 * lo), arm.startswith("STL+snap"), 6, 4)
    if arm == "STL-wide":
        return Steadlith(lo, min(T_MAX, 16 * lo), False)
    raise KeyError(arm)


SPACE = {"REC": (40, 900), "GCDC-wide": (5, 450), "GCDC-narrow": (5, 240), "GCDC-wide-p10": (5, 450),
         "GCDC-narrow-p10": (5, 240), "BCDC-wide": (256, 6000), "STL-narrow": (10, 240), "STL+snap-narrow": (10, 240),
         "STL-narrow-m6": (10, 240), "STL-wide-m6": (2, 120), "STL+snap-wide-m6": (2, 120), "STL-wide": (2, 120)}
ARMS = list(SPACE) + ["REC-e1"]


def band_ratio(arm, x):
    c = build(arm, x)
    if isinstance(c, GuardedCDC):
        return round(c.T_max / c.T_min, 2)
    if isinstance(c, ByteCDC):
        return round(c.max_size / c.min_size, 2)
    if isinstance(c, Steadlith):
        return round(c.max_tokens / c.min_tokens, 2)
    return None


def mean_tok(arm, x, versions):
    ch = build(arm, x); toks = []
    for v in versions:
        toks += ntok_many([c.text for c in ch.chunk(v)])
    return float(np.mean(toks)) if toks else 0.0


def bisect(arm, target, versions):
    lo, hi = SPACE[arm]; best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        m = mean_tok(arm, mid, versions)
        if best is None or abs(m - target) < abs(best[1] - target):
            best = (mid, m)
        if m < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def calibrate(a):
    f = os.path.join(OUT, f"band_calibration{a.tag}.json")
    res = json.load(open(f)) if os.path.exists(f) else {"params": {}, "slice_mean": {}}
    for s in a.streams.split(","):
        groups = doc_groups(load_stream(s))
        k = max(1, int(round(0.1 * len(groups))))
        versions = [Version(g[1][-1]["text"]) for g in groups[:k]]
        P = res["params"].setdefault(s, {}); M = res["slice_mean"].setdefault(s, {})
        for arm in SPACE:
            for t in TARGETS:
                key = f"{arm}@{t}"
                if key in P:
                    continue
                x, m = bisect(arm, t, versions)
                P[key] = int(x); M[key] = round(m, 1)
                print("calibrate", s, key, x, round(m, 1), flush=True)
                json.dump(res, open(f + ".tmp", "w"), indent=1); os.replace(f + ".tmp", f)
        P["REC-e1@128"] = int(calibration(s)["128"]["REC"]["T"]); M["REC-e1@128"] = None
    res["meta"] = run_meta(targets=TARGETS, slice_frac=0.1)
    json.dump(res, open(f + ".tmp", "w"), indent=1); os.replace(f + ".tmp", f)


CFG = {}


def process(args):
    doc, vers = args
    f = os.path.join(CFG["odir"], hashlib.sha1(str(doc).encode()).hexdigest()[:16] + ".parquet")
    if os.path.exists(f):
        return "skip"
    try:
        V = [Version(v["text"]) for v in vers]
        rows = []
        for key, x in CFG["configs"].items():
            arm = key.rsplit("@", 1)[0]
            ch = build(arm, x); prev = None
            for i, v in enumerate(V):
                cs = ch.chunk(v); keys = [c.key for c in cs]; toks = ntok_many([c.text for c in cs])
                if prev is not None:
                    ins = Counter(keys) - Counter(prev); tk = dict(zip(keys, toks))
                    rows.append(dict(stream=CFG["stream"], doc_id=str(doc), config=key, rev_new=int(vers[i]["rev_idx"]),
                                     ins=sum(ins.values()), dele=sum((Counter(prev) - Counter(keys)).values()),
                                     toks=sum(tk[k] * c for k, c in ins.items()), tok_new=sum(toks), n_new=len(cs),
                                     doc_tok_new=sum(u.tok for u in v.units)))
                prev = keys
        tmp = f + ".tmp"
        pd.DataFrame(rows).to_parquet(tmp, index=False); os.replace(tmp, f)
        return "ok"
    except Exception:
        return "ERROR " + str(doc) + " " + traceback.format_exc()[-800:]


def init(c):
    worker_threads(); CFG.update(c)


def replay(a):
    cal = json.load(open(os.path.join(OUT, f"band_calibration{a.tag}.json")))
    frames, errors, t0 = [], [], time.time()
    procs = a.procs
    if os.environ.get("MEMGUARD_SHRINK"):
        procs = max(1, procs // 2)
    procs = min(8, cap_procs(procs))
    for s in a.streams.split(","):
        configs = dict(sorted(cal["params"][s].items()))
        odir = checkpoint_dir(os.path.join(RAW, f"r2_band{a.tag}", s), code_hash("r2_band", s, a.docs, configs))
        groups = doc_groups(load_stream(s, a.docs))
        groups.sort(key=lambda g: -sum(len(v["text"]) for v in g[1]))     # longest histories first (load balance)
        with Pool(procs, initializer=init, initargs=(dict(stream=s, configs=configs, odir=odir),)) as pool:
            for k, m in enumerate(pool.imap_unordered(process, groups, chunksize=1)):
                if m.startswith("ERROR"):
                    errors.append(m); print(m[:300], flush=True)
                if (k + 1) % 50 == 0:
                    print(s, k + 1, "/", len(groups), "documents", round(time.time() - t0), "s", flush=True)
        frames += [pd.read_parquet(x) for x in glob.glob(os.path.join(odir, "*.parquet"))]
        print(s, "done", len(groups), "documents", len(errors), "errors", round(time.time() - t0), "s", flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} documents failed; rerun to resume: {errors[0][:300]}")
    df = pd.concat(frames, ignore_index=True)
    df["toks_frac"] = df.toks / df.doc_tok_new.clip(lower=1)
    df["chunk_tok_mean"] = df.tok_new / df.n_new.clip(lower=1)
    df.to_parquet(os.path.join(OUT, f"band_edits{a.tag}.parquet"), index=False)
    json.dump(run_meta(seconds=round(time.time() - t0, 1), procs=procs, rows=len(df), streams=a.streams, docs=a.docs),
              open(os.path.join(OUT, f"band_edits{a.tag}.meta.json"), "w"), indent=1)


def analyze(a):
    df = pd.read_parquet(os.path.join(OUT, f"band_edits{a.tag}.parquet"))
    cal = json.load(open(os.path.join(OUT, f"band_calibration{a.tag}.json")))
    df["arm"] = df.config.str.rsplit("@", n=1).str[0]
    rng = np.random.default_rng(20261111)
    rows, fidelity = [], []
    for s, f in df.groupby("stream"):
        doc = f.groupby(["doc_id", "config"]).agg(edits=("toks_frac", "size"), cost=("toks_frac", "sum"),
                                                  size=("chunk_tok_mean", "sum")).reset_index()
        ids = pd.Index(sorted(doc.doc_id.unique()))
        draws = rng.integers(len(ids), size=(2000, len(ids)))
        pts = {}
        for cfg, g in doc.groupby("config"):
            g = g.set_index("doc_id").reindex(ids)
            e, c, z = g.edits.to_numpy(), g.cost.to_numpy(), g["size"].to_numpy()
            pts[cfg] = dict(achieved=z.sum() / e.sum(), obs=c.sum() / e.sum(), boot=c[draws].sum(1) / e[draws].sum(1))
        size = COMMON.get(s)

        def interp(arm):
            cs = [k for k in pts if k.rsplit("@", 1)[0] == arm]
            P = sorted((pts[k]["achieved"], np.r_[pts[k]["obs"], pts[k]["boot"]], k) for k in cs)
            if len(P) < 2 or not (P[0][0] <= size <= P[-1][0]):
                return None, P
            w = (np.log(size) - np.log(P[0][0])) / (np.log(P[-1][0]) - np.log(P[0][0]))
            return np.exp((1 - w) * np.log(np.clip(P[0][1], 1e-9, None)) + w * np.log(np.clip(P[-1][1], 1e-9, None))), P
        ref, _ = interp("REC")
        for arm in [x for x in SPACE]:
            y, P = interp(arm)
            row = dict(stream=s, arm=arm, common_size=size, supported=y is not None,
                       achieved=" / ".join(f"{p[0]:.1f}" for p in P),
                       params=" / ".join(str(cal["params"][s][p[2]]) for p in P),
                       band_ratio=" / ".join(str(band_ratio(arm, cal["params"][s][p[2]])) for p in P),
                       cost_at_targets=" / ".join(f"{100 * p[1][0]:.2f}" for p in P))
            if y is not None:
                row.update(cost=float(y[0]), ci=[float(v) for v in np.percentile(y[1:], [2.5, 97.5])])
                if ref is not None and arm != "REC":
                    r = y / ref - 1; d = y - ref
                    row.update(rel_vs_rec=float(r[0]), rel_ci=[float(v) for v in np.percentile(r[1:], [2.5, 97.5])],
                               diff_pp=100 * float(d[0]), diff_ci=[100 * float(v) for v in np.percentile(d[1:], [2.5, 97.5])])
            rows.append(row)
        # fidelity: REC at E1's T=128 parameter must reproduce E1's per-edit counts
        e1 = pd.read_parquet(os.path.join(RESULTS, "e1_edits.parquet"), columns=["stream", "doc_id", "chunker", "T", "rev_new", "ins", "toks"],
                             filters=[("interval", "==", "revision"), ("chunker", "==", "REC"), ("T", "==", 128), ("stream", "==", s)])
        mine = f[f.arm.eq("REC-e1")].merge(e1, on=["doc_id", "rev_new"], suffixes=("", "_e1"))
        fidelity.append(dict(stream=s, edits=len(mine), ins_equal=float((mine.ins == mine.ins_e1).mean()) if len(mine) else None,
                             toks_equal=float((mine.toks == mine.toks_e1).mean()) if len(mine) else None))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, f"band_ablation{a.tag}.csv"), index=False)
    json.dump(fidelity, open(os.path.join(OUT, f"band_fidelity{a.tag}.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(t.to_string()); print(fidelity)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("step", choices=["calibrate", "replay", "analyze", "all"])
    ap.add_argument("--streams", default="wiki,k8s,cpy"); ap.add_argument("--docs", type=int)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("P3_THREADS", "8"))); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for step in (["calibrate", "replay", "analyze"] if a.step == "all" else [a.step]):
        dict(calibrate=calibrate, replay=replay, analyze=analyze)[step](a)


if __name__ == "__main__":
    main()
