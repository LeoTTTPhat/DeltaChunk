"""Freshness subset statistics (runner session 4): stale vs maintained index, paired on the same questions.

E4: REC rows of e4_quality{tag}.csv (maintained = final version) + REC-stale20 rows of e4_quality{tag}_stale.csv (index
of the version 20 retained revisions before the final one) -> e4_quality{tag}_fresh.csv (all questions) and
e4_quality{tag}_fresh_prox.csv (edit-proximal questions only: evidence changed in the last 20 edits), then
`stats_quality.py e4 --tag ..._fresh[_prox]` (paired vs REC: clustered bootstrap CI, McNemar, TOST, MDE).
E9: the same for the answers (REC from e9_answers{tag}.csv, REC-stale20 from e9_answers{tag}_stale.csv; contains).
Skips an experiment whose inputs are missing.
usage: python experiments/fresh_stats.py [--tag ""]
"""
import argparse, os, subprocess, sys

import pandas as pd
from common import RESULTS

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); a = ap.parse_args(); t = a.tag
    for exp, cfg, main_f, stale_f in (("e4", "chunker", f"e4_quality{t}.csv", f"e4_quality{t}_stale.csv"),
                                       ("e9", "config", f"e9_answers{t}.csv", f"e9_answers{t}_stale.csv")):
        fm, fs = os.path.join(RESULTS, main_f), os.path.join(RESULTS, stale_f)
        if not (os.path.exists(fm) and os.path.exists(fs)):
            print(exp, "freshness: inputs missing, skipped"); continue
        M = pd.read_csv(fm); S = pd.read_csv(fs)
        D = pd.concat([M[M[cfg] == "REC"], S[S[cfg] == "REC-stale20"]], ignore_index=True)
        common = set(D[D[cfg] == "REC"].qid) & set(D[D[cfg] == "REC-stale20"].qid)
        D = D[D.qid.isin(common)]
        base = "e4_quality" if exp == "e4" else "e9_answers"
        for sub, X in (("_fresh", D), ("_fresh_prox", D[D.proximal.astype(str).str.lower() == "true"])):
            if X.empty:
                continue
            X.to_csv(os.path.join(RESULTS, f"{base}{t}{sub}.csv"), index=False)
            print(exp, sub, "questions:", X.qid.nunique(), flush=True)
            subprocess.run([sys.executable, os.path.join(HERE, "stats_quality.py"), exp, "--tag", f"{t}{sub}"], check=True)


if __name__ == "__main__":
    main()
