"""E0 / Table T1: EditStream-RAG statistics per stream from the chunker-independent edit columns of E1 (per-revision,
T=200, REC rows) plus the extraction filter statistics.
usage: python experiments/e0_t1.py [--tag _smoke]
"""
import argparse, json, os

import numpy as np
import pandas as pd
from common import RESULTS, load_stream, run_meta


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); a = ap.parse_args()
    E = pd.read_parquet(os.path.join(RESULTS, f"e1_edits{a.tag}.parquet"),
                        columns=["stream", "doc_id", "chunker", "T", "interval", "ts", "edit_class", "n_regions", "l_old", "l_new", "doc_tok_new"])
    E = E[(E.chunker == "REC") & (E["T"] == 200) & (E.interval == "revision")]
    filt = {}
    for f in ("e0_filter_stats.json", "e0_k8s_stats.json", "e0_cpy_stats.json"):
        p = os.path.join(RESULTS, f)
        if os.path.exists(p):
            filt[f] = json.load(open(p))
    rows = []
    for s, g in E.groupby("stream"):
        span_days = max(1.0, (g.ts.max() - g.ts.min()) / 86400)
        per_doc_days = g.groupby("doc_id").ts.agg(lambda x: max(1.0, (x.max() - x.min()) / 86400))
        cls = g.edit_class.value_counts(normalize=True)
        rows.append(dict(stream=s, docs=g.doc_id.nunique(), edits=len(g), versions=len(g) + g.doc_id.nunique(),
                         share_ws=cls.get("ws", 0.0), share_structure=cls.get("structure", 0.0), share_single=cls.get("single", 0.0),
                         share_multi=cls.get("multi", 0.0), share_noop=cls.get("noop", 0.0),
                         l_old_p50=float(g.l_old.median()), l_new_p50=float(g.l_new.median()), l_new_p95=float(np.percentile(g.l_new, 95)),
                         doc_tok_deciles=[int(x) for x in np.percentile(g.groupby("doc_id").doc_tok_new.last(), np.arange(10, 100, 10))],
                         doc_tok_mean=float(g.groupby("doc_id").doc_tok_new.last().mean()),
                         edits_per_doc_day=float((g.groupby("doc_id").size() / per_doc_days).median()),
                         time_span_days=round(span_days, 1)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, f"t1_streams{a.tag}.csv"), index=False)
    json.dump(dict(filters=filt, meta=run_meta()), open(os.path.join(RESULTS, f"t1_streams{a.tag}.meta.json"), "w"), indent=1, default=str)
    print(df.round(3).T.to_string())


if __name__ == "__main__":
    main()
