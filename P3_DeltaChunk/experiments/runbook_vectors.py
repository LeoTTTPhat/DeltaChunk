"""Regenerate the vectors of an EditStream-RAG runbook deterministically from its point file (artifact rule: the runbooks
ship as YAML + point files; the vectors are rebuilt by this one command instead of being stored).

For every point id, results/runbooks/<stream>_<chunker>.points.parquet gives the 64-bit chunk key and the (document,
revision) that first inserted it. The script replays the chunker over each document's history with the parameters
e1.py used for the runbooks (T=200 calibration, results/calibration.json; stateful chunkers get the full revision
sequence, stateless ones chunk only the needed revisions), recovers the chunk text whose key matches, encodes it with
BAAI/bge-small-en-v1.5 (fp16 on MPS, unit-normalized; same Embedder and content cache as the experiments) and writes
  <out>/<stream>_<chunker>.fbin   big-ann-benchmarks float32 format: uint32 n, uint32 d, then n x d float32 (id order)
  <out>/<stream>_<chunker>.check.json   points, documents, keys not reproduced (must be 0), seconds
usage: PYTHONPATH=src python experiments/runbook_vectors.py --stream wiki --chunker REC [--docs N] [--out data/runbook_vectors]
       (chunker as in the file name: REC, SentCDC, DIFF_REC, KARA). --docs N restricts to the points of the first N
       documents of the point file (a reproducibility check; the .fbin then holds only those ids, in id order).
"""
import argparse, json, os, time

import numpy as np
import pandas as pd
from common import P3, RESULTS, calibration, doc_groups, load_stream

from deltachunk.chunkers import Version, make


def k64(hexkey: str) -> int:   # identical to e1.k64
    return int(hexkey[:16], 16) - (1 << 63)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stream", required=True); ap.add_argument("--chunker", required=True)
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument("--docs", type=int)
    selection.add_argument("--first-points", type=int, help="only regenerate the first N contiguous point IDs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", choices=["cpu", "mps"])
    ap.add_argument("--out", default=os.path.join(P3, "data", "runbook_vectors")); a = ap.parse_args(); t0 = time.time()
    name = {"DIFF_REC": "DIFF[REC]", "DIFF_STRUCT-M": "DIFF[STRUCT-M]"}.get(a.chunker, a.chunker)
    P = pd.read_parquet(os.path.join(RESULTS, "runbooks", f"{a.stream}_{a.chunker}{a.tag}.points.parquet"))
    P["doc_id"] = P.doc_id.astype(str)
    if a.first_points is not None:
        P = P.sort_values("id").head(a.first_points)
        if a.first_points <= 0 or P.id.tolist() != list(range(a.first_points)):
            ap.error("--first-points requires an existing contiguous prefix")
    if a.docs:
        keep = list(dict.fromkeys(P.doc_id))[:a.docs]; P = P[P.doc_id.isin(set(keep))]
    need = {d: g for d, g in P.groupby("doc_id")}
    params = calibration(a.stream).get("200", {}).get(name, {})   # e1.py: cal.get(name, {}) at T=200
    text_of = {}
    for d, vers in doc_groups(load_stream(a.stream)):
        d = str(d)
        if d not in need:
            continue
        g = need[d]; revs = set(int(r) for r in g.rev_idx); want = set(int(k) for k in g.key64)
        ch = make(name, params); st = None
        for v in vers:
            r = int(v["rev_idx"])
            if ch.stateful:
                cs, st = ch.update(Version(v["text"]), st)
            elif r in revs:
                cs = ch.chunk(Version(v["text"]))
            else:
                continue
            if r in revs:
                for c in cs:
                    k = k64(c.key)
                    if k in want:
                        text_of.setdefault((d, k), c.text)
    texts, missing = [], 0
    for r in P.sort_values("id").itertuples(index=False):
        t = text_of.get((r.doc_id, int(r.key64)))
        if t is None:
            missing += 1; t = ""
        texts.append(t)
    from deltachunk.embed import Embedder
    embedder = Embedder(device=a.device)
    X = embedder.encode(texts).astype(np.float32)
    os.makedirs(a.out, exist_ok=True)
    suffix = ""
    if a.docs:
        suffix = f"_docs{a.docs}"
    elif a.first_points:
        suffix = f"_points{a.first_points}"
    if a.device:
        suffix += f"_{a.device}"
    fn = os.path.join(a.out, f"{a.stream}_{a.chunker}{a.tag}{suffix}")
    with open(fn + ".fbin", "wb") as f:
        np.array([X.shape[0], X.shape[1]], dtype=np.uint32).tofile(f); X.tofile(f)
    chk = dict(stream=a.stream, chunker=name, points=len(P), documents=len(need), keys_not_reproduced=missing,
               params=params, dim=int(X.shape[1]), seconds=round(time.time() - t0, 1),
               device=embedder.device, model=embedder.model_name, file=os.path.relpath(fn + ".fbin", P3))
    json.dump(chk, open(fn + ".check.json", "w"), indent=1); print(chk)
    if missing:
        raise SystemExit(f"{missing} keys not reproduced")


if __name__ == "__main__":
    main()
