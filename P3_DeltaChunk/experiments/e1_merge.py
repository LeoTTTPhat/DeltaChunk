"""E1 merge: concatenate per-document checkpoints, compute global-scope S2/S3 counts in timestamp order (per-revision,
T=200), export big-ann-benchmarks runbooks, write results/e1_edits<tag>.parquet and a per-chunker summary CSV.
usage: python experiments/e1_merge.py [--tag _smoke] [--runbooks REC,SentCDC,DIFF[REC],KARA]
"""
import argparse, glob, json, os
from collections import Counter

import numpy as np
import pandas as pd
from common import RAW, RESULTS, run_meta

from deltachunk.delta import Runbook


def global_scopes(deltas: pd.DataFrame, runbook_name: str | None = None, search_every: int = 1000):
    """deltas of one (stream, chunker): rows with ins_keys/ins_toks/del_keys in global (ts, doc, rev) order, where ts is
    monotone within each document (common.load_stream). Asserts that each document's revisions are replayed in order
    and that no key's live count goes negative (review finding B4-T1)."""
    live, seen, last_rev = Counter(), set(), {}
    rb = Runbook(runbook_name, search_every=search_every) if runbook_name else None
    out = []
    for r in deltas.itertuples(index=False):
        if r.doc_id in last_rev and r.rev_idx <= last_rev[r.doc_id]:
            raise AssertionError(f"replay out of revision order: {r.doc_id} {r.rev_idx} after {last_rev[r.doc_id]}")
        last_rev[r.doc_id] = r.rev_idx
        added, removed = [], []
        tok = dict(zip(r.ins_keys, r.ins_toks))
        for k in r.ins_keys:
            if live[k] == 0:
                added.append(k)
            live[k] += 1
        for k in r.del_keys:
            if live[k] <= 0:
                raise AssertionError(f"delete of a key that is not live: {r.doc_id} rev {r.rev_idx}")
            live[k] -= 1
            if live[k] == 0:
                del live[k]; removed.append(k)
        added = list(dict.fromkeys(added))
        emb3 = [k for k in added if k not in seen]; seen.update(added)
        out.append((r.doc_id, r.rev_idx, len(added), sum(tok[k] for k in added), len(removed), len(emb3), sum(tok[k] for k in emb3)))
        if rb:
            rb.edit([str(k) for k in added], [str(k) for k in removed], origin=(r.doc_id, int(r.rev_idx)))
    return pd.DataFrame(out, columns=["doc_id", "rev_new", "S2_ins", "S2_toks", "S2_del", "S3_ins", "S3_toks"]), rb


def matched_size(summ: pd.DataFrame) -> pd.DataFrame:
    """Size-matched comparison (review finding B4-C2): per stream and chunker, interpolate the per-revision cost
    fraction against the achieved mean chunk size over the T sweep (log-log, linear between the measured points, no
    extrapolation) and read it at common sizes: every size reachable by all size-parameterized chunkers of the stream
    (multiples of 8 tokens between the largest per-chunker minimum and the smallest per-chunker maximum)."""
    rows = []
    sub = summ[summ.interval == "revision"]
    for s, g in sub.groupby("stream"):
        curves = {}
        for ch, h in g.groupby("chunker"):
            h = h.sort_values("chunk_tok_mean")
            if len(h) >= 2:
                curves[ch] = (np.log(h.chunk_tok_mean.values), np.log(h.toks_frac_mean.clip(lower=1e-9).values))
        if not curves:
            continue
        # chunkers whose achieved size barely moves with T (plateau, e.g. SentCDC/STRUCT-M on WIKI) do not define the
        # common range; they are interpolated only where their own range overlaps it (else absent = not size-matchable)
        wide = {c: v for c, v in curves.items() if v[0].max() - v[0].min() > np.log(1.1)} or curves
        lo = max(np.exp(x.min()) for x, _ in wide.values()); hi = min(np.exp(x.max()) for x, _ in wide.values())
        sizes = sorted(set([round(lo), round(hi)] + list(range(int(np.ceil(lo / 8) * 8), int(hi) + 1, 8)))) if hi >= lo else []
        for ch, (x, y) in curves.items():
            for z in sizes:
                if not (np.exp(x.min()) * 0.98 <= z <= np.exp(x.max()) * 1.02):
                    continue
                rows.append(dict(stream=s, chunker=ch, size=z, toks_frac_interp=float(np.exp(np.interp(np.log(z), x, y))),
                                 common_lo=round(lo, 1), common_hi=round(hi, 1)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="")
    ap.add_argument("--runbooks", default="REC,SentCDC,DIFF[REC],KARA"); a = ap.parse_args()
    base = os.path.join(RAW, f"e1{a.tag}")
    counts, scopes = [], []
    for sdir in sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d) and ".stale" not in d):
        stream = os.path.basename(sdir)
        c = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(sdir, "*.counts.parquet"))], ignore_index=True)
        counts.append(c)
        d = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(sdir, "*.delta.parquet"))], ignore_index=True)
        for ch, g in d.groupby("chunker"):
            g = g.sort_values(["ts", "doc_id", "rev_idx"], kind="stable")
            rbname = f"editstream-{stream}-{ch}" if ch in a.runbooks.split(",") else None
            sc, rb = global_scopes(g, rbname)
            sc["stream"], sc["chunker"], sc["interval"], sc["T"] = stream, ch, "revision", 200
            # the first version of each document is initial indexing, not an edit
            scopes.append(sc)
            if rb:
                os.makedirs(os.path.join(RESULTS, "runbooks"), exist_ok=True)
                fn = os.path.join(RESULTS, "runbooks", f"{stream}_{ch.replace('[', '_').replace(']', '')}{a.tag}")
                open(fn + ".yaml", "w").write(rb.to_yaml())
                # point id -> (64-bit chunk key, doc, revision that inserted it); the chunk text is reproduced by
                # re-chunking that (doc, revision) with the chunker's calibrated parameters (results/calibration.json)
                pd.DataFrame(rb.points, columns=["id", "key64", "doc_id", "rev_idx"]).to_parquet(fn + ".points.parquet", index=False)
    C = pd.concat(counts, ignore_index=True)
    S = pd.concat(scopes, ignore_index=True)
    C = C.merge(S, on=["stream", "chunker", "interval", "T", "doc_id", "rev_new"], how="left")
    # primary cross-chunker fraction: re-embedded tokens / tokens of the new DOCUMENT version (chunker-independent
    # denominator; review finding B4-FRAC: the chunk-token sum includes FIXO's overlap). Bian et al.'s workload
    # fraction (denominator = embedding-input tokens of the new version) is kept as toks_frac_workload (T-rep only).
    C["toks_frac"] = C["toks"] / C["doc_tok_new"].clip(lower=1)
    C["toks_frac_workload"] = C["toks"] / C["tok_new"].clip(lower=1)
    C["chunk_tok_mean"] = C["tok_new"] / C["n_new"].clip(lower=1)
    out = os.path.join(RESULTS, f"e1_edits{a.tag}.parquet")
    C.to_parquet(out, index=False)
    # summary: per stream x chunker x interval x T (edits with edit_class != ws/noop included; ws reported separately)
    g = C.groupby(["stream", "chunker", "T", "interval"])
    summ = g.agg(edits=("ins", "size"), ins_mean=("ins", "mean"), ins_p95=("ins", lambda x: np.percentile(x, 95)),
                 del_mean=("dele", "mean"), toks_mean=("toks", "mean"), toks_frac_mean=("toks_frac", "mean"),
                 toks_frac_workload_mean=("toks_frac_workload", "mean"), chunk_tok_mean=("chunk_tok_mean", "mean"),
                 n_new_mean=("n_new", "mean"), tok_new_mean=("tok_new", "mean"),
                 S2_ins_mean=("S2_ins", "mean"), S3_ins_mean=("S3_ins", "mean"), S2_toks_mean=("S2_toks", "mean"),
                 chunk_ms_mean=("chunk_ms", "mean"), chunk_total_ms_mean=("chunk_total_ms", "mean")).reset_index()
    summ.to_csv(os.path.join(RESULTS, f"e1_summary{a.tag}.csv"), index=False)
    matched_size(summ).to_csv(os.path.join(RESULTS, f"e1_matched_size{a.tag}.csv"), index=False)
    json.dump(run_meta(rows=len(C)), open(os.path.join(RESULTS, f"e1_edits{a.tag}.meta.json"), "w"), indent=1)
    print(summ[(summ["T"] == 200) & (summ.interval == "revision")].round(3).to_string())


if __name__ == "__main__":
    main()
