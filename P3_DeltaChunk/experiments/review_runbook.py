"""Check a real update trace and generate exact neighbors for a search prefix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from common import P3, RESULTS


def replay(book, n_points, n_searches):
    live = set()
    snapshots = []
    next_id = max_live = searches = prefix_size = 0
    for step, event in book.items():
        if step == "max_pts":
            continue
        op = event["operation"]
        if op == "search":
            searches += 1
            if searches <= n_searches:
                snapshots.append((step, np.array(sorted(live), dtype=np.int64)))
                prefix_size = next_id
            continue
        start, end = event["start"], event["end"]
        assert 0 <= start < end <= n_points
        span = range(start, end)
        if op == "insert":
            assert start == next_id and live.isdisjoint(span)
            live.update(span)
            next_id = end
            max_live = max(max_live, len(live))
        elif op == "delete":
            assert live.issuperset(span)
            live.difference_update(span)
        else:
            raise ValueError(f"unknown operation {op}")
    assert next_id == n_points and max_live == book["max_pts"]
    return snapshots, prefix_size, searches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--searches", type=int, default=2)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--out", type=Path, default=Path("data/runbook_vectors"))
    args = parser.parse_args()
    assert args.searches > 0 and args.queries > 0
    source = Path(RESULTS) / "runbooks"
    with (source / "wiki_REC.yaml").open() as file:
        book = yaml.load(file, Loader=yaml.CSafeLoader)["editstream-wiki-REC"]
    points = pd.read_parquet(source / "wiki_REC.points.parquet", columns=["id"]).id.to_numpy()
    np.testing.assert_array_equal(points, np.arange(len(points)))
    snapshots, prefix, searches = replay(book, len(points), args.searches)
    assert len(snapshots) == args.searches
    vectors = args.out / f"wiki_REC_points{prefix}_{args.device}.fbin"
    check = vectors.with_suffix(".check.json")
    if not (vectors.exists() and check.exists()):
        subprocess.run([sys.executable, str(Path(P3) / "experiments" / "runbook_vectors.py"),
                        "--stream", "wiki", "--chunker", "REC", "--first-points", str(prefix),
                        "--device", args.device, "--out", str(args.out)], check=True)
    metadata = json.loads(check.read_text())
    assert metadata["points"] == prefix and metadata["keys_not_reproduced"] == 0
    assert metadata["device"] == args.device
    with vectors.open("rb") as file:
        n, d = np.fromfile(file, dtype=np.uint32, count=2)
        assert n == prefix
        matrix = np.fromfile(file, dtype=np.float32).reshape(int(n), int(d))
    with (Path(RESULTS) / "qa" / "questions_wiki.jsonl").open() as file:
        queries = sorted(map(json.loads, file), key=lambda q: q["qid"])[:args.queries]
    from deltachunk.embed import Embedder
    embedder = Embedder(device=args.device)
    query_matrix = embedder.encode_queries([q["question"] for q in queries]).astype(np.float32)
    versions = json.loads((Path(RESULTS) / "data_versions.json").read_text())
    rows = []
    for step, ids in snapshots:
        assert 0 < len(ids) and ids[-1] < prefix
        scores = query_matrix @ matrix[ids].T
        for q, score in zip(queries, scores):
            nearest = np.lexsort((ids, -score))[:10]
            rows.append(dict(step=step, qid=q["qid"], live_points=len(ids), ids=ids[nearest].tolist(),
                             scores=[round(float(x), 7) for x in score[nearest]]))
    result = dict(stream="wiki", chunker="REC", point_ids=len(points), max_live=book["max_pts"],
                  all_search_steps=searches, regenerated_points=prefix, questions=len(queries),
                  model=embedder.model_name, model_snapshot=versions["model_snapshots"][embedder.model_name],
                  device=embedder.device, search_dtype="float32", results=rows)
    out = Path(RESULTS) / "review_runbook_neighbors.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(points)} point IDs, {searches} search markers, {prefix} regenerated vectors, {len(rows)} exact query results")


if __name__ == "__main__":
    main()
