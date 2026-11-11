"""Paper tables from results/ (E-phase analysis; no new measurements).

T3 (`results/t3_answers.{csv,tex}`): RAG answer quality per stream x configuration (E9): n, contains with its
document-clustered 95% CI, token F1, EM, paired difference vs REC with CI, Holm-corrected McNemar p and the TOST
outcome (with n vs n_required, so an unattainable equivalence is visible), plus the freshness subset (REC-stale20).
T4 (`results/t4_findings.csv`): the decision-guide inputs. Each row is one measured statement with its number(s) and
the source file; the paper's T4 prose must cite only these rows. Rows are produced only when the source exists.
usage: python experiments/tables.py [--tag ""]
"""
import argparse, glob, json, os

import numpy as np
import pandas as pd
from common import RESULTS


def R(n):
    return os.path.join(RESULTS, n)


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def t3(tag):
    fa = R(f"e9_answers{tag}.csv")
    if not os.path.exists(fa):
        print("T3: e9 answers missing"); return
    A = pd.read_csv(fa)
    if os.path.exists(R(f"e9_answers{tag}_stale.csv")):
        stale = pd.read_csv(R(f"e9_answers{tag}_stale.csv"))
        A = pd.concat([A, stale[stale.config == "REC-stale20"]])
    G = A.groupby(["stream", "config"]).agg(n=("qid", "size"), contains=("contains", "mean"), f1=("f1", "mean"), em=("em", "mean")).reset_index()
    S = pd.read_csv(R(f"e9_stats{tag}.csv")) if os.path.exists(R(f"e9_stats{tag}.csv")) else pd.DataFrame(columns=["stream", "config"])
    keep = [c for c in ["stream", "config", "ci_lo", "ci_hi", "diff", "diff_ci_lo", "diff_ci_hi", "p_holm", "n_required", "tost_equivalent"] if c in S.columns]
    T = G.merge(S[keep], on=["stream", "config"], how="left")
    order = {c: i for i, c in enumerate(["closed-book", "REC", "STRUCT-M", "SentCDC", "KARA", "REC-stale20"])}
    T = T.assign(_k=T.config.map(lambda c: order.get(c, 50))).sort_values(["stream", "_k"]).drop(columns="_k")
    T.to_csv(R(f"t3_answers{tag}.csv"), index=False)
    lines = [r"\begin{tabular}{llrcccc}", r"\toprule",
             r"Stream & Config & $n$ & Contains [95\% CI] & F1 & $\Delta$ vs REC [95\% CI] & TOST ($n$/$n_{req}$) \\", r"\midrule"]
    for r in T.itertuples():
        ci = f"[{f3(r.ci_lo)}, {f3(r.ci_hi)}]" if hasattr(r, "ci_lo") and pd.notna(r.ci_lo) else ""
        dl = (f"{r.diff:+.3f} [{r.diff_ci_lo:+.3f}, {r.diff_ci_hi:+.3f}]" if hasattr(r, "diff") and pd.notna(r.diff) else "--")
        to = (("eq." if bool(r.tost_equivalent) else "not shown") + f" ({r.n}/{int(r.n_required)})"
              if hasattr(r, "n_required") and pd.notna(getattr(r, "n_required", np.nan)) else "--")
        lines.append(f"{r.stream.upper()} & {r.config} & {r.n} & {f3(r.contains)} {ci} & {f3(r.f1)} & {dl} & {to} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(R(f"t3_answers{tag}.tex"), "w").write("\n".join(lines) + "\n")
    print("T3:", len(T), "rows")


def t4(tag):
    rows = []

    def add(topic, finding, value, source):
        rows.append(dict(topic=topic, finding=finding, value=value, source=source))

    fe = R(f"e1_summary{tag}.csv")
    if os.path.exists(fe):
        E = pd.read_csv(fe)
        d = E[(E["T"] == 128) & (E.interval == "revision")]
        for s, g in d.groupby("stream"):
            g = g.sort_values("toks_frac_mean")
            add("cost@T128", f"{s}: chunkers ranked by mean re-embedded fraction of the new version (S1, per revision)",
                "; ".join(f"{r.chunker} {r.toks_frac_mean:.3f} (size {r.chunk_tok_mean:.0f})" for r in g.itertuples()), os.path.basename(fe))
            ctx_free = g[g.chunker.isin(["REC", "STRUCT-M", "SentCDC", "GCDC", "BCDC", "STL", "STL+snap", "FIX", "FIXO"])]
            if len(ctx_free):
                b = ctx_free.iloc[0]
                add("cost@T128", f"{s}: lowest history-free cost at nominal T=128", f"{b.chunker} {b.toks_frac_mean:.3f}", os.path.basename(fe))
            for ref in ("FIX", "DOC"):
                if (g.chunker == ref).any() and (g.chunker == "REC").any():
                    x = float(g[g.chunker == ref].toks_frac_mean.iloc[0]); y = float(g[g.chunker == "REC"].toks_frac_mean.iloc[0])
                    add("cost@T128", f"{s}: {ref} vs REC re-embedded fraction", f"{x:.3f} vs {y:.3f} (ratio {x / y:.1f}x)", os.path.basename(fe))
        d = E[(E["T"] == 200)]
        for (s, c), g in d.groupby(["stream", "chunker"]):
            if c not in ("REC", "SentCDC", "KARA", "FIX", "DOC"):
                continue
            base = g[g.interval == "revision"]
            if len(base):
                tot = {r.interval: r.toks_mean * r.edits for r in g.itertuples()}
                b = tot.get("revision")
                add("sync", f"{s} {c}: total re-embedded tokens vs per-revision sync (T=200)",
                    "; ".join(f"{('sync at release dates' if k == 'release' else k)} {v / b:.2f}" for k, v in tot.items() if k != "revision"),
                    os.path.basename(fe))
    for s in ("wiki", "k8s", "cpy"):
        for r3 in ("", "_r3"):
            f = R(f"e2_thm1_{s}{r3}{tag}.json")
            if os.path.exists(f):
                j = json.load(open(f))
                add("theory", f"{s}: Theorem 1 at r={j['r']} (violations INS/DEL over covered edits)",
                    f"{j['viol_ins']}/{j['viol_del']} over {j['covered_edits']} of {j['edits']} edits ({j['covered_share']:.1%} covered)", os.path.basename(f))
    f = R(f"e2_predict{tag}.json")
    if os.path.exists(f):
        j = json.load(open(f))
        for s in ("k8s", "cpy"):
            if s in j:
                add("predictor", f"WIKI-calibrated INS predictor on {s}: MdAPE non-trivial / Kendall tau",
                    f"{j[s]['MdAPE_nontrivial']:.3f} / {j[s]['kendall_tau']:.3f}", os.path.basename(f))
    for f in sorted(glob.glob(R(f"e3_drift_*{tag}.csv"))):
        if not tag and "_smoke" in f:
            continue
        X = pd.read_csv(f); X = X[X.cohort == "all"]
        for c, g in X.groupby("chunker"):
            fin = g[g.t == -1]   # t = -1: each document's final version unless that edit index is itself a checkpoint (e3.py)
            if len(fin):
                r = fin.iloc[0]
                add("drift", f"{r.stream} {c}: key divergence / KS vs fresh at each document's final version ({int(r.docs)} docs whose final edit index is not a checkpoint)",
                    f"{r.divergence:.3f} / {r.ks:.3f}", os.path.basename(f))
            g = g[(g.t >= 0) & (g.docs >= 20)]   # revision-index checkpoints with >= 20 documents still alive
            if len(g):
                last = g[g.t == g.t.max()].iloc[0]
                add("drift", f"{last.stream} {c}: key divergence / KS vs fresh at t={int(last.t)} ({int(last.docs)} docs; largest t with >= 20 docs)",
                    f"{last.divergence:.3f} / {last.ks:.3f}", os.path.basename(f))
    f = R(f"e4_stats{tag}.csv")
    if os.path.exists(f):
        S = pd.read_csv(f)
        for s, g in S.groupby("stream"):
            add("quality", f"{s}: evidence recall@1024 range over chunkers (T=128, native background)",
                f"{g['mean'].min():.3f} ({g.loc[g['mean'].idxmin(), 'config']}) - {g['mean'].max():.3f} ({g.loc[g['mean'].idxmax(), 'config']})", os.path.basename(f))
            sig = g[(g.p_holm < 0.05)] if "p_holm" in g else g.iloc[0:0]
            add("quality", f"{s}: chunkers differing from REC at Holm p<0.05", ", ".join(f"{r.config} {r.diff:+.3f}" for r in sig.itertuples()) or "none",
                os.path.basename(f))
            eq = g[g.tost_equivalent == True] if "tost_equivalent" in g else g.iloc[0:0]
            add("quality", f"{s}: equivalent to REC within 2 pp (TOST, n >= n_required)", ", ".join(eq.config) or "none", os.path.basename(f))
    for exp, metric in (("e4", "recall@1024"), ("e9", "answer contains")):
        for sub, lab in (("_fresh", "all questions"), ("_fresh_prox", "edit-proximal questions")):
            f = R(f"{exp}_stats{tag}{sub}.csv")
            if os.path.exists(f):
                S = pd.read_csv(f); r = S[(S.stream == "all") & (S.config == "REC-stale20")]
                if len(r):
                    r = r.iloc[0]
                    add("freshness", f"{exp.upper()} stale-20 vs maintained REC, {metric}, {lab} (pooled; paired, clustered 95% CI)",
                        f"{r['mean']:.3f} vs {r['mean'] - r['diff']:.3f}: diff {r['diff']:+.3f} [{r.diff_ci_lo:+.3f}, {r.diff_ci_hi:+.3f}], n={int(r.n)}",
                        os.path.basename(f))
    f = R(f"e5b_stats{tag}.csv")
    if os.path.exists(f):
        S = pd.read_csv(f).set_index("policy")
        add("late", "E5b policies vs EXACT: recall diff [clustered 95% CI] / forward tokens rel. to EXACT",
            "; ".join(f"{p} {S.loc[p, 'diff']:+.3f} [{S.loc[p, 'diff_ci_lo']:+.3f}, {S.loc[p, 'diff_ci_hi']:+.3f}] / {S.loc[p, 'forward_rel']:.2f}"
                      for p in S.index if p != "EXACT"), os.path.basename(f))
    f = R(f"e5a_late{tag}.json")
    if os.path.exists(f):
        j = json.load(open(f)); bd = pd.DataFrame(j["by_distance"])
        if len(bd):
            r1 = bd.iloc[0]
            add("late", "LATE: cos(old,new) of unchanged chunks adjacent to an edit (mean / p05) vs LATE-ctl min",
                f"{r1.cos_late_mean:.4f} / {r1.cos_late_p05:.4f} vs {r1.cos_ctl_min:.6f}", os.path.basename(f))
    f = R(f"e5b_policies{tag}.json")
    if os.path.exists(f):
        j = json.load(open(f))
        add("late", "edit-proximal recall EXACT / STALE / WINDOW-1 and forward tokens",
            " / ".join(f"{k} {j['recall'][k]:.3f} ({j['cost'][k]['forward_tokens']:,} tok)" for k in ("EXACT", "STALE", "WINDOW-1") if k in j["recall"]), os.path.basename(f))
    f = R(f"e5c_ctx_cost{tag}.json")
    if os.path.exists(f):
        j = json.load(open(f))
        if "prompt_cache" in j and "no_prompt_cache" in j:
            add("ctx", "CTX seconds per chunk call without / with prompt cache (measured under contention)",
                f"{j['no_prompt_cache']['sec_per_call']:.2f} / {j['prompt_cache']['sec_per_call']:.2f}", os.path.basename(f))
    f = R(f"e7_analysis{tag}.json")
    if os.path.exists(f):
        j = json.load(open(f))
        for s, v in j.get("variance_decomposition", {}).items():
            k = "A_scope@revision/without_llm"
            if k in v:
                add("regime", f"{s}: share of log-cost variance explained by chunker (scope x regime design, no LLM regime)",
                    f"{v[k]['chunker']:.3f}", os.path.basename(f))
        add("index", "judge-3 valid (rho(N) measured and c_q > 0)", str(j.get("judge3_valid")), os.path.basename(f))
    f = R(f"e7_judge3{tag}.csv")
    if os.path.exists(f):
        X = pd.read_csv(f); v = X.differs.dropna() if "differs" in X else pd.Series(dtype=float)
        add("index", "cells where the chunker-aware cost-optimal index policy differs from the index-only one", f"{int(v.sum())} of {len(v)} defined",
            os.path.basename(f))
    if rows:
        pd.DataFrame(rows).to_csv(R(f"t4_findings{tag}.csv"), index=False)
    print("T4:", len(rows), "rows")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); a = ap.parse_args()
    t3(a.tag); t4(a.tag)


if __name__ == "__main__":
    main()
