"""E2 (b): out-of-domain INS predictor from chunker-agnostic edit statistics (design.md §3.3).

Inputs per edit (from E1, per revision, T=200): l_old, l_new, n_regions, rel_pos, doc_tok_new, units_new; per stream:
the empirical block (paragraph) token-length distribution and mean unit tokens of final versions. No measured counts of
the target stream are used.
  FIX      : (1 - rel_pos) * doc_tok_new / T + 1         (offset-defined: every later window shifts)
  DOC      : doc_tok_new / T                              (all chunks; T = the stream's configured REC target)
  SENTWIN  : l_new                                        (one vector per new unit)
  SentCDC  : sum over regions of the Monte Carlo table E[INS | l_old, l_new, r, lambda, p_par]
  REC, STRUCT-M : coupling simulation of two greedy packings (predict.greedy_ins_mc) on synthetic documents with the
             stream's block lengths; the edited block changes length by (l_new - l_old) * mean unit tokens
Edits without unit-identity regions (ws/reflow-only) are predicted 0 for every chunker. Per-stream calibration and
SentCDC table (the stream's empirical paragraph-end share p_par, l up to 16 then scaled). Seeds: sha256(doc:rev).
One free multiplicative factor per chunker is fitted on WIKI (measured / predicted mean INS) and applied unchanged
to K8S and CPY. Reported: per stream x chunker predicted vs measured mean INS, MdAPE, Kendall tau of the ranking.
usage: python experiments/e2_predict.py [--tag ""] [--sample 1500]
"""
import argparse, hashlib, json, os

import numpy as np
import pandas as pd
from common import RESULTS, calibration, doc_groups, load_stream, run_meta
from scipy.stats import kendalltau

from deltachunk.chunkers import Version
from deltachunk.predict import greedy_ins_mc, sentcdc_ins_mc

CHUNKERS = ["FIX", "DOC", "SENTWIN", "SentCDC", "REC", "STRUCT-M"]
NONTRIVIAL = ["SentCDC", "REC", "STRUCT-M"]


def stream_stats(stream, ndocs=100):
    blocks, units, par = [], [], []
    for d, vers in doc_groups(load_stream(stream, ndocs)):
        v = Version(vers[-1]["text"]); cur = 0
        for u in v.units:
            cur += u.tok; units.append(u.tok); par.append(bool(u.par_end))
            if u.par_end:
                blocks.append(cur); cur = 0
    return np.array(blocks), float(np.mean(units)), float(np.mean(par))


def seed_of(doc, rev) -> int:
    return int(hashlib.sha256(f"{doc}:{rev}".encode()).hexdigest()[:8], 16)


L_MAX = 16


def sentcdc_table(r, lam, p_par):
    """E[INS | l_old, l_new] for one region, l <= L_MAX (Monte Carlo); larger l scaled linearly from the edge."""
    t = {}
    for lo in range(0, L_MAX + 1):
        for ln in range(0, L_MAX + 1):
            if lo or ln:
                t[(lo, ln)] = sentcdc_ins_mc(200, 100, lo, ln, r, lam, p_par, samples=200, seed=lo * 101 + ln)[0]
    return t


def sentcdc_pred(t, lo, ln):
    if lo == 0 and ln == 0:
        return 0.0
    m = max(lo, ln)
    if m <= L_MAX:
        return t[(lo, ln)]
    f = L_MAX / m
    return t[(min(L_MAX, round(lo * f)), min(L_MAX, round(ln * f)))] / f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); ap.add_argument("--sample", type=int, default=1500)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    E = pd.read_parquet(os.path.join(RESULTS, f"e1_edits{a.tag}.parquet"))
    E = E[(E["T"] == 200) & (E.interval == "revision") & (E.chunker.isin(CHUNKERS))]
    per = []
    cfg = {}
    for s, g in E.groupby("stream"):
        cal = calibration(s).get("200", {})
        r = cal.get("SentCDC", {}).get("r", 3); lam = cal.get("SentCDC", {}).get("lam", 0.5)
        Tm = {c: cal.get(c, {}).get("T", 200) for c in ("REC", "STRUCT-M", "FIX")}
        blocks, unit_tok, p_par = stream_stats(s)
        table = sentcdc_table(r, lam, p_par)
        cfg[s] = dict(r=r, lam=lam, T=Tm, p_par=p_par, unit_tok=unit_tok, mean_block_tok=float(blocks.mean()))
        edits = g[g.chunker == "REC"].sample(min(a.sample, (g.chunker == "REC").sum()), random_state=0)
        meas = g.set_index(["chunker", "doc_id", "rev_new"]).ins
        for e in edits.itertuples():
            for c in CHUNKERS:
                if e.n_regions == 0:
                    p = 0.0        # no unit-identity change (ws/reflow-only edit): no chunk can change
                elif c == "FIX":
                    p = (1 - max(0.0, e.rel_pos)) * e.doc_tok_new / Tm["FIX"] + 1
                elif c == "DOC":
                    p = e.doc_tok_new / Tm["REC"]     # chunker-agnostic: document tokens / configured T
                elif c == "SENTWIN":
                    p = e.l_new
                elif c == "SentCDC":
                    p = sum(sentcdc_pred(table, b - a_, b2 - a2) for a_, b, a2, b2 in json.loads(e.regions))
                else:
                    n_par = max(1, int(e.doc_tok_new / max(1.0, blocks.mean())))
                    delta = np.array([(e.l_new - e.l_old) * unit_tok]) if e.l_new != e.l_old else np.array([unit_tok * 0.5])
                    p = greedy_ins_mc(blocks, n_par, max(0.0, e.rel_pos), "modify", Tm[c], samples=60, seed=seed_of(e.doc_id, e.rev_new),
                                      delta_lengths=delta) * max(1, e.n_regions)
                per.append(dict(stream=s, chunker=c, doc_id=e.doc_id, rev_new=e.rev_new, edit_class=e.edit_class,
                                predicted_raw=float(p), measured=float(meas.get((c, e.doc_id, e.rev_new), np.nan))))
    P = pd.DataFrame(per).dropna(subset=["measured"])
    P.to_parquet(os.path.join(RESULTS, f"e2_predict_edits{a.tag}.parquet"), index=False)
    fac = (P[P.stream == "wiki"].groupby("chunker").measured.mean() / P[P.stream == "wiki"].groupby("chunker").predicted_raw.mean()).to_dict()
    P["predicted"] = P.predicted_raw * P.chunker.map(fac)
    df = P.groupby(["stream", "chunker"]).agg(measured=("measured", "mean"), predicted_raw=("predicted_raw", "mean"),
                                               predicted=("predicted", "mean"), edits=("measured", "size")).reset_index()
    df["factor_from_wiki"] = df.chunker.map(fac)
    df["ape"] = (df.predicted - df.measured).abs() / df.measured.clip(lower=1e-9)
    # document-clustered bootstrap CIs of APE per chunker and of Kendall tau per stream
    rng = np.random.default_rng(0); ci = {}
    for s, g in P[P.stream != "wiki"].groupby("stream"):
        docs = g.doc_id.unique(); by = {d: h for d, h in g.groupby("doc_id")}
        apes, taus = [], []
        for _ in range(a.boot):
            bs = pd.concat([by[d] for d in rng.choice(docs, len(docs))])
            m = bs.groupby("chunker")[["measured", "predicted"]].mean()
            apes.append(((m.predicted - m.measured).abs() / m.measured.clip(lower=1e-9)).to_dict())
            taus.append(kendalltau(m.predicted, m.measured).statistic)
        A = pd.DataFrame(apes)
        ci[s] = dict(ape_ci95={c: [float(A[c].quantile(.025)), float(A[c].quantile(.975))] for c in A.columns},
                     kendall_tau_ci95=[float(np.nanquantile(taus, .025)), float(np.nanquantile(taus, .975))])
    cls = P[P.stream != "wiki"].groupby(["stream", "chunker", "edit_class"])[["measured", "predicted"]].mean()
    cls["ape"] = (cls.predicted - cls.measured).abs() / cls.measured.clip(lower=1e-9)
    out = dict(rows=df.to_dict("records"), per_class=cls.reset_index().to_dict("records"), config=cfg,
               note=("Per-chunker APE is the primary result; FIX/DOC/SENTWIN are closed forms (trivial), REC/STRUCT-M "
                     "(greedy coupling) and SentCDC (Monte Carlo) are the non-trivial predictors and are summarised separately."))
    for s in ("k8s", "cpy"):
        d = df[df.stream == s]
        if len(d):
            nt = d[d.chunker.isin(NONTRIVIAL)]
            out[s] = dict(MdAPE_all=float(d.ape.median()), MdAPE_nontrivial=float(nt.ape.median()),
                          ape={c: float(x) for c, x in zip(d.chunker, d.ape)},
                          kendall_tau=float(kendalltau(d.predicted, d.measured).statistic), items=len(d), **ci.get(s, {}))
    out["meta"] = run_meta(seeds="sha256(doc_id:rev_new)")
    json.dump(out, open(os.path.join(RESULTS, f"e2_predict{a.tag}.json"), "w"), indent=1, default=str)
    print(df.round(3).to_string()); print({k: v for k, v in out.items() if k in ("k8s", "cpy")})


if __name__ == "__main__":
    main()
