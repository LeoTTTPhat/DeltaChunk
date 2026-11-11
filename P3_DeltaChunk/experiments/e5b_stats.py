"""E5b paired statistics (runner session 4): each late-chunking maintenance policy vs EXACT on the same edit-proximal
questions (recall@1024): mean with document-clustered bootstrap 95% CI, paired difference with clustered bootstrap CI,
exact McNemar p (Holm across policies), Newcombe paired 90% interval and n_required for the 2 pp TOST margin (same
rules as stats_quality.py), MDE = 2.80 x clustered-bootstrap SE. Cost columns (forward tokens and vectors written, also
relative to EXACT) come from e5b_policies.json.
usage: python experiments/e5b_stats.py [--tag ""] [--qtag _prox]
"""
import argparse, json, os

import numpy as np
import pandas as pd
from common import RESULTS, run_meta
from scipy.stats import binomtest
from stats_quality import MARGIN, clustered, n_required, newcombe_paired


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); ap.add_argument("--qtag", default="_prox")
    a = ap.parse_args()
    D = pd.read_csv(os.path.join(RESULTS, f"e5b_policies{a.tag}.csv"))
    J = json.load(open(os.path.join(RESULTS, f"e5b_policies{a.tag}.json")))
    qdoc = {}
    for l in open(os.path.join(RESULTS, "qa", f"questions_wiki{a.qtag}.jsonl")):
        q = json.loads(l); qdoc[q["qid"]] = str(q["doc_id"])
    D["cluster"] = D.qid.map(qdoc).fillna(D.qid)
    ref = D[D.policy == "EXACT"].set_index("qid").recall1024
    ex = J["cost"]["EXACT"]
    rows = []
    for p, h in D.groupby("policy", sort=False):
        m, ci, _ = clustered(h.recall1024, h.cluster)
        c = J["cost"].get(p, {})
        row = dict(policy=p, n=len(h), docs=h.cluster.nunique(), recall1024=m, ci_lo=ci[0], ci_hi=ci[1],
                   forward_tokens=c.get("forward_tokens"), forward_rel=c.get("forward_tokens", np.nan) / ex["forward_tokens"],
                   vectors=c.get("vectors"), vectors_rel=c.get("vectors", np.nan) / ex["vectors"])
        if p != "EXACT":
            x = h.set_index("qid").recall1024; common = x.index.intersection(ref.index)
            d = x[common] - ref[common]; cl = h.set_index("qid").cluster[common]
            dm, dci, boots = clustered(d, cl)
            b, cc = int(((x[common] == 1) & (ref[common] == 0)).sum()), int(((x[common] == 0) & (ref[common] == 1)).sum())
            aa = int(((x[common] == 1) & (ref[common] == 1)).sum()); dd = int(((x[common] == 0) & (ref[common] == 0)).sum())
            lo, hi = newcombe_paired(aa, b, cc, dd); n = len(common); nreq = n_required((b + cc) / max(1, n))
            row.update(diff=dm, diff_ci_lo=dci[0], diff_ci_hi=dci[1], discordant=b + cc,
                       mcnemar_p=binomtest(b, b + cc, 0.5).pvalue if b + cc else 1.0, tost90_lo=lo, tost90_hi=hi,
                       n_required=nreq, tost_equivalent=bool(lo > -MARGIN and hi < MARGIN and (b + cc) > 0 and n >= nreq),
                       mde_boot=float(2.8016 * boots.std(ddof=1)))
        rows.append(row)
    R = pd.DataFrame(rows)
    idx = R.dropna(subset=["mcnemar_p"]).sort_values("mcnemar_p").index; prev = 0
    for i, j in enumerate(idx):
        prev = max(prev, min(1.0, (len(idx) - i) * R.loc[j, "mcnemar_p"])); R.loc[j, "p_holm"] = prev
    R.to_csv(os.path.join(RESULTS, f"e5b_stats{a.tag}.csv"), index=False)
    json.dump(run_meta(ref="EXACT", metric="recall1024", margin=MARGIN), open(os.path.join(RESULTS, f"e5b_stats{a.tag}.meta.json"), "w"), indent=1)
    print(R.round(4).to_string())


if __name__ == "__main__":
    main()
