"""Size-supported cost curves with a paired document bootstrap at fixed size weights."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS

ALIGNED = ("BCDC", "DIFF[REC]", "DIFF[STRUCT-M]", "GCDC", "KARA", "REC", "STRUCT-M", "SentCDC")
TARGETS = {"wiki": (134, 144), "k8s": (136, 176), "cpy": (137, 200)}
COLS = ["stream", "doc_id", "chunker", "T", "toks_frac", "chunk_tok_mean"]


def interpolate(size, achieved, costs):
    if size < achieved[0] or size > achieved[-1]:
        return None
    right = np.clip(np.searchsorted(achieved, size), 1, len(achieved) - 1)
    left = right - 1
    w = (np.log(size) - np.log(achieved[left])) / (np.log(achieved[right]) - np.log(achieved[left]))
    return np.exp((1 - w) * np.log(costs[left].clip(1e-9)) + w * np.log(costs[right].clip(1e-9)))


def main():
    base = Path(RESULTS)
    raw = pd.read_parquet(base / "e1_edits.parquet", columns=COLS, filters=[("interval", "==", "revision")])
    doc = raw.groupby(["stream", "doc_id", "chunker", "T"], observed=True).agg(
        edits=("toks_frac", "size"), total_cost=("toks_frac", "sum"), total_size=("chunk_tok_mean", "sum")).reset_index()
    del raw
    curve_rows, spread_rows = [], []
    rng = np.random.default_rng(20261110)
    for stream, targets in TARGETS.items():
        frame = doc[doc.stream.eq(stream)]
        ids = pd.Index(frame.doc_id.unique()).sort_values()
        draws = rng.integers(len(ids), size=(2000, len(ids)))
        points = {}
        for chunker in (*ALIGNED, "FIX", "STL"):
            series = []
            for nominal, group in frame[frame.chunker.eq(chunker)].groupby("T"):
                group = group.set_index("doc_id").reindex(ids)
                assert group.edits.notna().all() and group.edits.gt(0).all()
                edits = group.edits.to_numpy()
                costs = group.total_cost.to_numpy()
                achieved = group.total_size.sum() / edits.sum()
                weighted = costs[draws].sum(axis=1) / edits[draws].sum(axis=1)
                observed = costs.sum() / edits.sum()
                series.append((achieved, np.r_[observed, weighted]))
                lo, hi = np.quantile(weighted, [0.025, 0.975])
                curve_rows.append(dict(stream=stream, chunker=chunker, nominal=nominal, achieved=achieved,
                                       cost=observed, ci_lo=lo, ci_hi=hi, documents=len(ids)))
            points[chunker] = sorted(series, key=lambda x: x[0])
        for target in targets:
            estimates = {}
            for chunker, values in points.items():
                sizes = np.array([x for x, _ in values])
                if len(np.unique(sizes)) < 2:
                    continue
                pred = interpolate(target, sizes, np.array([c for _, c in values]))
                if pred is not None:
                    estimates[chunker] = pred
            family = {c: y for c, y in estimates.items() if c in ALIGNED}
            ys = np.array(list(family.values()))
            spread = ys.max(axis=0) - ys.min(axis=0)
            spread_ci = np.quantile(spread[1:], [0.025, 0.975])
            spread_rows.append(dict(stream=stream, size=target, family=len(family),
                                    excluded=",".join(sorted(set(ALIGNED) - family.keys())),
                                    low=ys[:, 0].min(), high=ys[:, 0].max(), spread=spread[0],
                                    spread_ci_lo=spread_ci[0], spread_ci_hi=spread_ci[1],
                                    fixed=estimates["FIX"][0], steadlith=estimates["STL"][0],
                                    rec=estimates["REC"][0], documents=len(ids)))
    pd.DataFrame(curve_rows).to_csv(base / "review_size_curves.csv", index=False)
    table = pd.DataFrame(spread_rows)
    table.to_csv(base / "review_size_spreads.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
