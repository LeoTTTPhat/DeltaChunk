"""results/SUMMARY.md generator: every number planned for the paper, with the exact configuration and the source file.

Each section reads only files in results/ (written by the E-phase drivers) and prints PENDING when an input is missing,
so the summary can be regenerated at any time. Nothing is estimated or extrapolated: every number is read or computed
(means, document-clustered bootstrap CIs) from the raw result files named in the section.
usage: python experiments/summary.py [--tag ""] [--out results/SUMMARY.md]
"""
import argparse, glob, json, os, subprocess, time, traceback

import numpy as np
import pandas as pd
from common import P3, RESULTS

ORDER = ["FIX", "FIXO", "REC", "STRUCT-M", "STRUCT-NM", "SENTWIN", "SentCDC", "GCDC", "BCDC", "STL", "STL+snap", "KARA",
         "DIFF[REC]", "DIFF[STRUCT-M]", "DOC"]
STREAMS = ["wiki", "k8s", "cpy"]


def R(name):
    return os.path.join(RESULTS, name)


def rel(path):
    return os.path.relpath(path, P3)


def jload(name):
    return json.load(open(R(name)))


def fmt(x, nd=3):
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(fmt(v, nd) for v in x) + "]"
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    ax = abs(float(x))
    if ax != 0 and (ax < 10 ** -nd or ax >= 1e6):
        return f"{float(x):.{nd}g}"
    return f"{float(x):.{nd}f}"


def table(df, nd=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(r[c], nd) if not isinstance(r[c], str) else r[c] for c in cols) + " |")
    return "\n".join(out)


def cluster_ci(df, col, B=1000, seed=0):
    g = df.groupby("doc_id")[col].agg(["sum", "size"])
    s, n = g["sum"].values.astype(float), g["size"].values
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(s), size=(B, len(s)))
    boots = s[idx].sum(1) / n[idx].sum(1)
    return s.sum() / n.sum(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def sort_chunkers(df, col="chunker"):
    key = {c: i for i, c in enumerate(ORDER)}
    return df.assign(_k=df[col].map(lambda c: key.get(str(c).split("-stale")[0], 99))).sort_values(
        [c for c in ["stream"] if c in df.columns] + ["_k", col]).drop(columns="_k")


def meta_line(m, keys=("time", "threads", "seconds")):
    if not isinstance(m, dict):
        return ""
    parts = []
    for k in keys:
        if k in m:
            v = m[k]
            parts.append(f"{k}={json.dumps(v) if isinstance(v, dict) else v}")
    return "; ".join(parts)


class Doc:
    def __init__(self):
        self.lines = []

    def h(self, t, lvl=2):
        self.lines += ["", "#" * lvl + " " + t, ""]

    def p(self, *t):
        self.lines += list(t)

    def files(self, *names):
        ex = [n for n in names if os.path.exists(R(n)) or glob.glob(R(n))]
        self.lines += [""]; self.lines.append("Files: " + (", ".join(f"`results/{n}`" for n in ex) if ex else "none yet"))

    def pending(self, what):
        self.lines.append(f"**PENDING**: {what} not in results/ yet.")


# ------------------------------------------------------------------------------------------------------------- sections
def s_notes(D, tag):
    D.h("Paper-facing notes (binding)")
    k8s = jload("e0_k8s_stats.json") if os.path.exists(R("e0_k8s_stats.json")) else {}
    cpy = jload("e0_cpy_stats.json") if os.path.exists(R("e0_cpy_stats.json")) else {}
    D.p(f"- **Streams.** K8S and CPY are main-branch (first-parent) histories: a merged PR is one edit. K8S has "
        f"**{k8s.get('docs', 'n/a')} files** with >= {k8s.get('min_commits', 20)} first-parent changes (not 300), "
        f"{fmt(k8s.get('versions'))} versions; CPY {cpy.get('docs', 'n/a')} files, {fmt(cpy.get('versions'))} versions "
        f"(results/e0_k8s_stats.json, results/e0_cpy_stats.json).",
        "- **Size calibration.** T=128 is a nominal target calibrated separately per stream. Achieved sizes vary by "
        "chunker and stream, including on the E4 final snapshots. At T=200 "
        "SentCDC (WIKI, K8S), STRUCT-M and GCDC (WIKI) and BCDC (K8S) cannot reach the target, so T=200 rows are not "
        "size-matched; achieved per-stream mean chunk sizes are printed next to every count. E4/E9 quality runs use T=128 "
        "(runner session 1 fix); the sync-interval, scope (S2/S3) and regime analyses exist at T=200 only (calibrated, "
        "achieved sizes stated). `results/e1_matched_size.csv` interpolates cost over the T sweep at common sizes.",
        "- **toks_frac** = re-embedded tokens / tokens of the new document version (Bian et al.'s workload fraction only "
        "in T-rep as `toks_frac_workload`).",
        "- **CPY release axis** is labelled **'sync at release dates'**: main's file state at the timestamps of `v3.*` tags "
        "(mostly maintenance-branch tags), not the release branches' contents.",
        "- **Theorem 1 denominators.** Guard-split edits are excluded from the bound check; every Theorem 1 statement "
        "must give `covered_edits` / `edits` per stream and r (table in E2 below).",
        "- **TOST rule.** Equivalence (2 pp margin) is declared only if the Newcombe paired 90% interval lies in "
        "[-0.02, 0.02], >= 1 discordant pair exists, and n >= n_required (power 0.8, discordance floored at 5%: n >= 773-897 "
        "pairs). With a few hundred questions per stratum, 'equivalent' is mostly unattainable: report CIs and say so.",
        "- **Steadlith zero-token words.** Words made only of characters the bge tokenizer drops (U+200E, control "
        "characters) have 0 tokens, which Steadlith rejects; such versions are re-chunked with max(1, ntok(word)) "
        "(experiments/common.py; first seen on 4 WIKI documents, 90 versions). State it with the STL baseline.",
        "- **E5c** runs one client thread with a 45-minute cap per invocation (the full 200 chunks x 3 reps would take ~4 h); "
        "report the completed calls/documents/repetitions printed in the E5 section.",
        "- Counts are primary (hardware independent); times are measured under contention on a shared M3 Ultra; derived "
        "costs are labelled 'derived'.")


def s_t1(D, tag):
    D.h("E0 / T1: EditStream-RAG statistics")
    f = f"t1_streams{tag}.csv"
    if not os.path.exists(R(f)):
        return D.pending("T1")
    T = pd.read_csv(R(f))
    cols = [c for c in ["stream", "docs", "versions", "edits", "share_noop", "share_ws", "share_structure", "share_single",
                        "share_multi", "l_old_p50", "l_new_p50", "l_new_p95", "doc_tok_mean", "edits_per_doc_day",
                        "time_span_days"] if c in T.columns]
    D.p(table(T[cols]), "", "Token deciles per document: " + "; ".join(f"{r.stream}: {r.doc_tok_deciles}" for r in T.itertuples()))
    D.files(f, f"t1_streams{tag}.meta.json", "e0_filter_stats.json", "e0_k8s_stats.json", "e0_cpy_stats.json", "data_versions.json")


def load_e1(tag):
    f = R(f"e1_edits{tag}.parquet")
    if not os.path.exists(f):
        return None
    cols = ["stream", "doc_id", "chunker", "T", "interval", "ins", "dele", "toks", "toks_frac", "chunk_tok_mean", "guard_ins",
            "S2_ins", "S3_ins", "S2_toks", "S3_toks", "tok_new", "stl_fallbacks_doc"]
    import pyarrow.parquet as pq
    have = set(pq.read_schema(f).names)
    return pd.read_parquet(f, columns=[c for c in cols if c in have])


def s_e1(D, tag, E):
    D.h("E1: re-embedding cost per edit (counts)")
    if E is None:
        return D.pending("E1 (e1_edits.parquet)")
    m = jload(f"e1_edits{tag}.meta.json") if os.path.exists(R(f"e1_edits{tag}.meta.json")) else {}
    D.p(f"Rows: {fmt(len(E))} (chunker x edit x T/interval). Run: {meta_line(m)}.",
        "Scope S1 (per-document multiset), per-revision sync. Mean per edit with document-clustered bootstrap 95% CI "
        "(1,000 resamples). `size` = achieved mean chunk tokens of the new versions.", "")
    for T in (128, 200):
        d = E[(E["T"] == T) & (E.interval == "revision")]
        if T == 128:   # size-independent chunkers only exist at T=200 (unmatched by design)
            d = pd.concat([d, E[(E["T"] == 200) & (E.interval == "revision") & E.chunker.isin(["SENTWIN", "STRUCT-NM"])]])
        rows = []
        for (s, c), g in d.groupby(["stream", "chunker"]):
            mf, lo, hi = cluster_ci(g, "toks_frac")
            rows.append(dict(stream=s, chunker=c + (" (T-indep.)" if c in ("SENTWIN", "STRUCT-NM") else ""),
                             edits=len(g), docs=g.doc_id.nunique(), toks_frac=mf, ci_lo=lo, ci_hi=hi,
                             INS=g.ins.mean(), INS_p95=g.ins.quantile(0.95), DEL=g.dele.mean(), size=g.chunk_tok_mean.mean()))
        if not rows:
            continue
        X = sort_chunkers(pd.DataFrame(rows).assign(chunker=lambda x: x.chunker))
        D.h(f"E1 at T={T}" + (" (size-matched; primary)" if T == 128 else " (calibrated, NOT size-matched for all chunkers)"), 3)
        D.p(table(X))
    # scopes at T=200 revision
    d = E[(E["T"] == 200) & (E.interval == "revision")]
    if "S2_ins" in d.columns and d.S2_ins.notna().any():
        g = d.groupby(["stream", "chunker"]).agg(S1_INS=("ins", "mean"), S2_INS=("S2_ins", "mean"), S3_INS=("S3_ins", "mean"),
                                                 S1_TOKS=("toks", "mean"), S2_TOKS=("S2_toks", "mean")).reset_index()
        D.h("E1 scopes S1/S2/S3 (T=200, per revision; S2 = global dedup per stream, S3 = cross-stream)", 3)
        D.p(table(sort_chunkers(g)))
    # sync intervals (F6)
    d = E[E["T"] == 200]
    ivs = [i for i in ["revision", "daily", "weekly", "monthly", "release"] if i in set(d.interval)]
    if len(ivs) > 1:
        tot = d.groupby(["stream", "chunker", "interval"]).toks.sum().unstack("interval")
        base = tot["revision"]
        X = (tot[ivs].div(base, axis=0)).reset_index()
        X.columns = [("sync at release dates" if c == "release" else c) for c in X.columns]
        D.h("E1 sync interval (F6): total re-embedded tokens relative to per-revision sync (T=200, S1)", 3)
        D.p("CPY 'release' = sync at release dates (main's file state at v3.* tag timestamps).", "", table(sort_chunkers(X)))
    if "stl_fallbacks_doc" in E.columns:
        fb = E[(E.chunker.str.startswith("STL"))].groupby(["stream", "chunker"]).stl_fallbacks_doc.max()
        D.p("", "Steadlith long-word split fallbacks (max per document): " + ", ".join(f"{k[0]}/{k[1]}={int(v)}" for k, v in fb.items()))
    D.files(f"e1_edits{tag}.parquet", f"e1_summary{tag}.csv", f"e1_matched_size{tag}.csv", f"e1_edits{tag}.meta.json", "calibration.json", "runbooks/")


def s_e1rep(D, tag):
    D.h("E1-rep: replication of Bian et al. (T-rep)")
    f = f"e1_rep{tag}.json"
    if not os.path.exists(R(f)):
        return D.pending("E1-rep")
    j = jload(f)
    X = pd.DataFrame([dict(chunker=k, recompute_pct=v["mean_recompute_pct"], median_len=v["median_len"],
                           size_matched_300=v["size_matched_300"]) for k, v in j["summary"].items()])
    D.p(f"Pairs: {j.get('pairs')} adjacent WIKI revision pairs; metric = % of new-version embedding-input tokens recomputed "
        "under byte-identical reuse (Bian's workload fraction).", "", table(X, 1), "",
        "Bian et al. English Table I (their data, for reference, not reproduced here): " + json.dumps(j.get("bian_english_table_I")))
    D.files(f, f"e1_rep{tag}.csv", "bian_params.json")


def s_e1fid(D, tag):
    D.h("E1-fid: LangChain index() fidelity")
    f = f"e1_fid{tag}.json"
    if not os.path.exists(R(f)):
        return D.pending("E1-fid")
    j = jload(f)
    rows = [dict(setting=k, edits=v["edits"], lc_added=v["lc_added_total"], S1_distinct=v["S1_distinct_total"],
                 match_S1_distinct=v["match_S1_distinct"], match_S2=v["match_S2"]) for k, v in j.items() if isinstance(v, dict) and "edits" in v]
    D.p(table(pd.DataFrame(rows)))
    D.files(f)


def s_e2(D, tag):
    D.h("E2: Theorem 1, Monte Carlo and out-of-domain predictor")
    rows = []
    for s in STREAMS:
        for r3 in (False, True):
            f = f"e2_thm1_{s}{'_r3' if r3 else ''}{tag}.json"
            if os.path.exists(R(f)):
                j = jload(f)
                rows.append(dict(stream=s, r=j.get("r"), lam=j.get("lam"), edits=j.get("edits"), covered_edits=j.get("covered_edits"),
                                 covered_share=j.get("covered_share"), guard_edits=j.get("guard_edits"), viol_ins=j.get("viol_ins"),
                                 viol_del=j.get("viol_del"), mean_ins=j.get("mean_ins"), mean_bound=j.get("mean_bound"),
                                 mc_edits=j.get("mc_edits"), mc_pred=j.get("mc_mean_pred"), mc_meas=j.get("mc_mean_meas")))
    if rows:
        D.p("Theorem 1 (corrected bound) per SentCDC edit; denominators = covered_edits (guard-split edits excluded).", "",
            table(pd.DataFrame(rows)))
    else:
        D.pending("E2 Theorem 1 JSONs")
    f = f"e2_predict{tag}.json"
    if os.path.exists(R(f)):
        j = jload(f)
        rows = []
        for s in ("k8s", "cpy"):
            if s in j:
                v = j[s]
                for c, ape in v.get("ape", {}).items():
                    ci = v.get("ape_ci95", {}).get(c, [np.nan, np.nan])
                    rows.append(dict(stream=s, chunker=c, APE=ape, ci_lo=ci[0], ci_hi=ci[1]))
                D.p(f"- {s}: MdAPE (all) {fmt(v.get('MdAPE_all'))}, MdAPE non-trivial (SentCDC, REC, STRUCT-M) "
                    f"{fmt(v.get('MdAPE_nontrivial'))}, Kendall tau {fmt(v.get('kendall_tau'))} "
                    f"(95% CI {', '.join(fmt(x) for x in v.get('kendall_tau_ci95', []))}), items {v.get('items')}")
        if rows:
            D.p("", "Out-of-domain INS predictor (calibrated on WIKI), per-chunker APE with document-clustered 95% CI:", "",
                table(sort_chunkers(pd.DataFrame(rows))))
        if "note" in j:
            D.p("", "Note: " + str(j["note"]))
    else:
        D.pending("e2_predict")
    D.files(*[f"e2_thm1_{s}{x}{tag}.json" for s in STREAMS for x in ("", "_r3")], f"e2_predict{tag}.json", f"e2_predict_edits{tag}.parquet")


def s_e3(D, tag):
    D.h("E3: drift of history-dependent chunkers")
    fs = [f for f in sorted(glob.glob(R(f"e3_drift_*{tag}.csv"))) if tag or "_smoke" not in f]
    if not fs:
        D.pending("E3 drift")
    else:
        X = pd.concat([pd.read_csv(f) for f in fs])
        X = X[X.t >= 0]
        out = []
        for (s, coh, c), g in X.groupby(["stream", "cohort", "chunker"]):
            last = g[g.t == g.t.max()].iloc[0]
            out.append(dict(stream=s, cohort=coh, chunker=c, t_max=last.t, docs=last.docs, ks=last.ks, divergence=last.divergence,
                            out_of_range_m=last.out_of_range_m, out_of_range_f=last.out_of_range_f, mean_m=last.mean_m, mean_f=last.mean_f,
                            renorm_toks_frac=last.get("renorm_toks_frac", np.nan)))
        D.p("Drift vs fresh chunking at the largest t of each cohort (m = maintained, f = fresh); full curves in the CSVs.", "",
            table(pd.DataFrame(out)))
        D.files(*[os.path.basename(f) for f in fs])
    f = f"e3_quality{tag}.json"
    if os.path.exists(R(f)):
        S = pd.DataFrame(jload(f)["summary"])
        D.p("", "Sentence-probe recall@1024, maintained vs fresh layout (WIKI-100, docs whose history reaches t):", "", table(S))
        D.files(f, f"e3_quality{tag}.csv")
        if not tag and os.path.exists(R("e3q_ci_mde.csv")):
            M = pd.read_csv(R("e3q_ci_mde.csv"))
            D.p("", "E3q gap (maintained - fresh) with document-clustered 95% CIs and MDE (alpha 0.05 two-sided, power 0.8; "
                "mde_boot = 2.80 x clustered-bootstrap SE; `docs` = documents holding the probes at t, so t=200 rests on 4 "
                "documents and its CI is not reliable):", "",
                table(M[["chunker", "t", "probes", "docs", "gap", "gap_ci_lo", "gap_ci_hi", "discordant", "mde_boot"]]))
            D.files("e3q_ci_mde.csv")
    else:
        D.pending("E3q")


def s_e4(D, tag):
    D.h("E4: retrieval quality at nominal T=128")
    for variant, f in (("native background (primary)", f"e4_quality{tag}.csv"), ("no background (control)", f"e4_quality_nobg{tag}.csv"),
                       ("stale-20 index (REC)", f"e4_quality{tag}_stale.csv"),
                       ("history-dependent layouts: DIFF[REC], DIFF[STRUCT-M] maintained over the full history; KARA-fresh = "
                        "KARA on the final version without history (optional review item 8; REC/STRUCT-M/KARA rows are the "
                        "primary run's)", f"e4_quality{tag}_diff.csv")):
        if not os.path.exists(R(f)):
            D.pending(f"E4 {variant}"); continue
        Q = pd.read_csv(R(f)); m = {}
        mf = R(f.replace(".csv", ".meta.json"))
        if os.path.exists(mf):
            m = json.load(open(mf))
        cols = [c for c in ["recall@1024", "recall@2048", "recall_lex@1024", "ndcg10", "used_tok@1024"] if c in Q.columns]
        G = Q.groupby(["stream", "chunker"])[cols].mean().reset_index()
        G.insert(2, "n", Q.groupby(["stream", "chunker"]).size().values)
        if m.get("achieved_tok"):
            G["size"] = [m["achieved_tok"].get(c.split("-stale")[0], {}).get(s, np.nan) for s, c in zip(G.stream, G.chunker)]
        D.h(f"E4 {variant}", 3)
        D.p(f"Calibration T={m.get('T', '200 (build default)')}; background chunks {m.get('background_chunks')} ({m.get('bg_mode')}); "
            f"questions {m.get('questions')}; model {m.get('model')} on {m.get('device')}.", "", table(sort_chunkers(G)))
    f = f"e4_stats{tag}.csv"
    if os.path.exists(R(f)):
        S = pd.read_csv(R(f))
        cols = ["stream", "config", "n", "mean", "ci_lo", "ci_hi", "diff", "diff_ci_lo", "diff_ci_hi", "p_holm", "discordant",
                "tost90_lo", "tost90_hi", "n_required", "tost_equivalent", "mde_boot", "mde_mcnemar"]
        D.p("", "Paired statistics vs REC (recall@1024; clustered bootstrap 95% CIs; McNemar Holm; Newcombe TOST 2 pp):", "",
            table(sort_chunkers(S[[c for c in cols if c in S.columns]], "config")))
    f = f"e4_stats{tag}_diff.csv"
    if os.path.exists(R(f)):
        S = pd.read_csv(R(f))
        cols = ["stream", "config", "n", "mean", "ci_lo", "ci_hi", "diff", "diff_ci_lo", "diff_ci_hi", "p_holm", "mde_boot", "mde_mcnemar"]
        D.p("", "History-dependent layouts, paired vs REC (recall@1024; same statistics):", "",
            table(sort_chunkers(S[[c for c in cols if c in S.columns]], "config")))
    fresh_block(D, "e4", tag, "recall@1024")
    D.files(f"e4_quality{tag}.csv", f"e4_quality{tag}.meta.json", f"e4_quality_nobg{tag}.csv", f"e4_quality{tag}_stale.csv", f"e4_stats{tag}.csv",
            f"e4_quality{tag}_diff.csv", f"e4_stats{tag}_diff.csv", "qa/")


def fresh_block(D, exp, tag, metric):
    """Freshness subset (experiments/fresh_stats.py): REC-stale20 vs maintained REC, paired on the same questions."""
    for sub, label in (("_fresh", "all questions"), ("_fresh_prox", "edit-proximal questions only (evidence sentence unit absent "
                                                                      "from version n-20, i.e. from the stale index's version, by construction)")):
        f = f"{exp}_stats{tag}{sub}.csv"
        if not os.path.exists(R(f)):
            continue
        S = pd.read_csv(R(f))
        cols = ["stream", "config", "n", "mean", "ci_lo", "ci_hi", "diff", "diff_ci_lo", "diff_ci_hi", "mcnemar_p", "discordant"]
        D.p("", f"Freshness, stale-20 vs maintained REC ({metric}; {label}; paired, clustered bootstrap 95% CI, McNemar):", "",
            table(S[[c for c in cols if c in S.columns]]))
        D.files(f, f.replace("_stats", "_quality" if exp == "e4" else "_answers"))


def s_e5(D, tag):
    D.h("E5: contextual embeddings (LATE, STALE/WINDOW-D, CTX)")
    f = f"e5a_late{tag}.json"
    if os.path.exists(R(f)):
        j = jload(f)
        D.p(f"E5a: {j.get('edits')} WIKI edits, {fmt(j.get('chunks'))} unchanged chunks; cosine(old, new) of unchanged chunks by distance "
            "to the nearest changed chunk (LATE vs LATE-ctl):", "", table(pd.DataFrame(j["by_distance"]), 4))
    else:
        D.pending("E5a")
    f = f"e5b_policies{tag}.json"
    if os.path.exists(R(f)):
        j = jload(f)
        X = pd.DataFrame([dict(policy=k, recall=v, forward_tokens=j["cost"].get(k, {}).get("forward_tokens"),
                               vectors=j["cost"].get(k, {}).get("vectors")) for k, v in j["recall"].items()])
        D.p("", f"E5b: {j.get('docs')} docs, last {j.get('last')} edits, {j.get('questions')} edit-proximal questions:", "", table(X))
        sf = f"e5b_stats{tag}.csv"
        if os.path.exists(R(sf)):
            S = pd.read_csv(R(sf))
            cols = ["policy", "n", "docs", "recall1024", "ci_lo", "ci_hi", "forward_rel", "vectors_rel", "diff", "diff_ci_lo",
                    "diff_ci_hi", "discordant", "p_holm", "tost90_lo", "tost90_hi", "n_required", "tost_equivalent", "mde_boot"]
            D.p("", "E5b paired vs EXACT (experiments/e5b_stats.py; document-clustered bootstrap 95% CI, McNemar Holm, Newcombe "
                "TOST 2 pp; forward_rel / vectors_rel = cost relative to EXACT):", "", table(S[[c for c in cols if c in S.columns]], 4))
    else:
        D.pending("E5b")
    f = f"e5c_ctx_cost{tag}.json"
    if os.path.exists(R(f)):
        j = jload(f)
        X = pd.DataFrame([dict(variant=k, **{kk: vv for kk, vv in v.items() if kk != "per_stratum"},
                               **{f"s/call {kk}": vv for kk, vv in v.get("per_stratum", {}).items()})
                          for k, v in j.items() if k in ("no_prompt_cache", "prompt_cache")])
        D.p("", f"E5c CTX per-call cost (one client thread, ABBA, measured under contention); completed: {json.dumps(j.get('completed'))}", "", table(X))
    else:
        D.pending("E5c")
    D.files(f"e5a_late{tag}.json", f"e5a_late{tag}.parquet", f"e5b_policies{tag}.json", f"e5b_policies{tag}.csv", f"e5b_stats{tag}.csv", f"e5c_ctx_cost{tag}.json", f"e5c_ctx_calls{tag}.csv")


def s_e6(D, tag):
    D.h("E6: HNSW under churn (reduced) and rebuild time rho(N)")
    f = f"e6_index{tag}.csv"
    if os.path.exists(R(f)):
        X = pd.read_csv(R(f))
        fin = X[X["final"] == True] if "final" in X.columns and (X["final"] == True).any() else X[X.edits == X.groupby(["chunker", "policy"]).edits.transform("max")]
        cols = [c for c in ["recall_exact", "recall_fresh", "recall_ef50", "recall_ef100", "recall_ef200", "tomb_share", "tomb_share_churned",
                            "lat_ms", "rebuilds", "rebuild_s", "edits", "live"] if c in fin.columns]
        keys = ["chunker", "policy"] + (["tau"] if "tau" in fin.columns else [])
        G = fin.groupby(keys)[cols].mean().reset_index()
        G.insert(len(keys), "seeds", fin.groupby(keys).seed.nunique().values if "seed" in fin.columns else 1)
        D.p("Final checkpoint, mean over HNSW seeds:", "", table(G))
        D.files(f, f"e6_index{tag}.meta.json")
    else:
        D.pending("E6")
    f = f"e7_rho{tag}.json"
    if os.path.exists(R(f)):
        j = jload(f)
        D.p("", "rho(N) (HNSW build seconds): " + ", ".join(f"{k}={fmt(v)}" for k, v in j.items() if k != "meta"),
            "(N above the real-vector count uses perturbed copies of real vectors; labelled.)")
        D.files(f)
    else:
        D.pending("E7 rho")


def s_e7(D, tag):
    D.h("E7: regimes, unit costs, timing, variance decomposition, judge-3, staleness")
    f = f"e7_unit_costs{tag}.json"
    if os.path.exists(R(f)):
        j = jload(f)
        D.p("Unit costs (Theil-Sen over timed calls; seconds ~ alpha0 + alpha * tokens): " + ", ".join(
            f"{k}={fmt(j.get(k), 4)}" for k in ["alpha0", "alpha", "alpha_ci95", "single_call_s", "beta", "gamma"] if k in j))
    else:
        D.pending("E7 micro (unit costs)")
    f = f"e7_timing{tag}.csv"
    if os.path.exists(R(f)):
        X = pd.read_csv(R(f))
        G = X.groupby(["chunker", "regime"]).agg(reps=("rep", "nunique"), edits=("edits", "mean"), per_1k_edits_s=("per_1k_edits_s", "mean"),
                                                 per_1k_sd=("per_1k_edits_s", "std"), t_chunk=("t_chunk", "mean"), t_embed=("t_embed", "mean"),
                                                 t_index=("t_index", "mean"), calls=("calls", "mean"), encoded=("encoded", "mean")).reset_index()
        D.p("", "Measured maintenance time (WIKI-20 slices, rotated order, cold cache; measured under contention):", "", table(sort_chunkers(G)))
    else:
        D.pending("E7 timing")
    f = f"e7c_llm{tag}.json"
    if os.path.exists(R(f)):
        j = jload(f)
        D.p("", "E7c LLM extraction: " + ", ".join(f"{k}={fmt(v)}" for k, v in j.items() if k != "meta"))
    else:
        D.pending("E7c")
    f = f"e7_analysis{tag}.json"
    if os.path.exists(R(f)):
        j = jload(f)
        D.p("", f"judge-3 valid: {j.get('judge3_valid')}; rho_N={fmt(j.get('rho_N'))}; c_q={fmt(j.get('c_q'), 4)} "
            f"(95% CI {j.get('c_q_ci95')}); LLM s/call={fmt(j.get('llm_sec_per_call'))}.")
        vd = j.get("variance_decomposition", {})
        rows = []
        for s, v in vd.items():
            for design, comp in v.items():
                if isinstance(comp, dict) and "chunker" in comp:
                    rows.append(dict(stream=s, design=design, **{k: comp.get(k) for k in comp if not isinstance(comp.get(k), dict)}))
        if rows:
            D.p("", "Variance decomposition of log derived cost (balanced designs; shares of SS):", "", table(pd.DataFrame(rows)))
        if "judge3" in j:
            D.p("", "judge-3 summary: " + json.dumps(j["judge3"])[:1500])
    else:
        D.pending("e7_analysis")
    f = f"e7_judge3{tag}.csv"
    if os.path.exists(R(f)):
        X = pd.read_csv(R(f))
        if "differs" in X.columns:
            v = X.differs.dropna()
            D.p("", f"judge-3: cost-optimal index policy differs (chunker-aware vs index-only) in {int(v.sum())} of {len(v)} defined cells "
                f"({len(X) - len(v)} undefined).")
    f = f"e7_staleness{tag}.csv"
    if os.path.exists(R(f)):
        X = pd.read_csv(R(f))
        D.p("", "Staleness simulation (derived, trace-driven; budget grid shared by all chunkers per stream):", "",
            table(sort_chunkers(X[X.chunker.isin(["FIX", "REC", "STRUCT-M", "SentCDC", "STL", "KARA", "DIFF[REC]", "DOC"])])))
    D.files(f"e7_unit_costs{tag}.json", f"e7_unit_calls{tag}.csv", f"e7_timing{tag}.csv", f"e7c_llm{tag}.json", f"e7_analysis{tag}.json",
            f"e7_derived_costs{tag}.csv", f"e7_judge3{tag}.csv", f"e7_staleness{tag}.csv")


def s_e8(D, tag):
    D.h("E8: sensitivity (counts)")
    fs = [f for f in sorted(glob.glob(R(f"e8_sens_*{tag}.csv"))) if tag or "_smoke" not in f]
    if not fs:
        return D.pending("E8")
    for f in fs:
        X = pd.read_csv(f)
        D.p("", f"`{os.path.basename(f)}`:", "", table(X))
    D.files(*[os.path.basename(f) for f in fs])


def s_e9(D, tag):
    D.h("E9 / T3: RAG answer quality")
    f = f"e9_answers{tag}.csv"
    if not os.path.exists(R(f)):
        return D.pending("E9 answers")
    A = pd.read_csv(R(f))
    if os.path.exists(R(f"e9_answers{tag}_stale.csv")):
        stale = pd.read_csv(R(f"e9_answers{tag}_stale.csv"))
        A = pd.concat([A, stale[stale.config == "REC-stale20"]])
    G = A.groupby(["stream", "config"]).agg(n=("qid", "size"), em=("em", "mean"), f1=("f1", "mean"), contains=("contains", "mean")).reset_index()
    D.p("Answers by qwen3:8b (temperature 0) on the E4 contexts (budget 1,024 tokens); closed-book = no context.", "", table(G))
    f = f"e9_stats{tag}.csv"
    if os.path.exists(R(f)):
        S = pd.read_csv(R(f))
        cols = ["stream", "config", "n", "mean", "ci_lo", "ci_hi", "diff", "diff_ci_lo", "diff_ci_hi", "p_holm", "tost90_lo", "tost90_hi",
                "n_required", "tost_equivalent", "mde_boot", "mde_mcnemar"]
        D.p("", "Paired statistics vs REC (contains):", "", table(S[[c for c in cols if c in S.columns]]))
    if os.path.exists(R("t3_answers.csv")):
        D.p("", "T3 table: `results/t3_answers.csv` / `results/t3_answers.tex` (experiments/tables.py).")
    fresh_block(D, "e9", tag, "contains")
    D.files(f, f"e9_answers{tag}_stale.csv", f"e9_stats{tag}.csv", "t3_answers.csv", "t3_answers.tex")


def s_tierB(D, tag):
    """Tier B (experiments/tierB.sh): REC-O counts, E8 quality points, E6 REBUILD tau 0.05/0.1 (seed 0)."""
    D.h("Tier B (run only after every Tier A result existed)")
    D.p("Driver: `experiments/tierB.sh` (hard stop for launches Nov 10 14:00). Items absent below were not run.")
    f = f"e1_edits{tag}_tierB.parquet"
    if os.path.exists(R(f)):
        B = pd.read_parquet(R(f), columns=["stream", "doc_id", "chunker", "T", "interval", "ins", "toks_frac", "chunk_tok_mean"])
        B = B[B.chunker == "REC-O"]
        E = load_e1(tag)
        if E is not None:
            B = pd.concat([B, E[E.chunker.isin(["REC", "FIXO"])][B.columns]])
        rows = []
        for (s, T, c), g in B[B.interval == "revision"].groupby(["stream", "T", "chunker"]):
            mf, lo, hi = cluster_ci(g, "toks_frac")
            rows.append(dict(stream=s, T=T, chunker=c, edits=len(g), docs=g.doc_id.nunique(), toks_frac=mf, ci_lo=lo, ci_hi=hi,
                             INS=g.ins.mean(), size=g.chunk_tok_mean.mean()))
        D.h("REC-O (REC + 40-token overlap) counts vs REC and FIXO (S1, per revision)", 3)
        D.p(table(pd.DataFrame(rows).sort_values(["stream", "T", "chunker"])))
        D.files(f, f"e1_summary{tag}_tierB.csv")
    else:
        D.p("", "REC-O counts: not run.")
    fs = sorted(glob.glob(R(f"e4_quality{tag}_e8q_*.csv")))
    if fs:
        D.h("E8 quality points (E4 protocol, native 30k background; recall@1024)", 3)
        for f in fs:
            t = os.path.basename(f)[len(f"e4_quality{tag}"):-4]
            Q = pd.read_csv(f); m = {}
            if os.path.exists(f[:-4] + ".meta.json"):
                m = json.load(open(f[:-4] + ".meta.json"))
            G = Q.groupby(["stream", "chunker"])[["recall@1024", "ndcg10"]].mean().reset_index()
            G.insert(2, "n", Q.groupby(["stream", "chunker"]).size().values)
            if m.get("achieved_tok"):
                G["size"] = [m["achieved_tok"].get(c, {}).get(s, np.nan) for s, c in zip(G.stream, G.chunker)]
            D.p("", f"`{t}` (calibration T={m.get('T')}; SentCDC@lam=1 = calibrated r with lambda=1):", "", table(G))
            sf = R(f"e4_stats{tag}{t}.csv")
            if os.path.exists(sf):
                S = pd.read_csv(sf)
                cols = ["stream", "config", "n", "mean", "diff", "diff_ci_lo", "diff_ci_hi", "p_holm", "mde_boot"]
                D.p("", "Paired vs REC (same run):", "", table(S[[c for c in cols if c in S.columns]]))
        D.files(*[os.path.basename(f) for f in fs], *[os.path.basename(f) for f in sorted(glob.glob(R(f"e4_stats{tag}_e8q_*.csv")))])
    else:
        D.p("", "E8 quality points: not run.")
    f = f"e6_index{tag}_tauB.csv"
    if os.path.exists(R(f)):
        X = pd.read_csv(R(f))
        fin = X[X["final"] == True] if "final" in X.columns and (X["final"] == True).any() else X[X.edits == X.groupby(["chunker", "policy"]).edits.transform("max")]
        cols = [c for c in ["recall_exact", "recall_fresh", "tomb_share_churned", "lat_ms", "rebuilds", "rebuild_s", "edits"] if c in fin.columns]
        keys = ["chunker", "policy"] + (["tau"] if "tau" in fin.columns else [])
        D.h("E6 REBUILD at tau 0.05 / 0.1 (seed 0)", 3)
        D.p(table(fin.groupby(keys)[cols].mean().reset_index()))
        D.files(f)
    else:
        D.p("", "E6 tau 0.05/0.1: not run.")

    variants = (
        ("_e4_reco", "REC with 40-token overlap"),
        ("_e4_sentwin", "Single-unit vectors with two neighboring units on each side"),
        ("_e4_e5", "E4 repeated with the E5 encoder"),
    )
    for suffix, title in variants:
        stats = f"e4_stats{tag}{suffix}.csv"
        meta = f"e4_quality{tag}{suffix}.meta.json"
        if not (os.path.exists(R(stats)) and os.path.exists(R(meta))):
            continue
        M = jload(meta)
        S = pd.read_csv(R(stats))
        cols = ["stream", "config", "n", "mean", "diff", "diff_ci_lo", "diff_ci_hi", "p_holm", "tost_equivalent"]
        D.h(title, 3)
        D.p(f"Encoder: `{M['model']}` on {M['device']}. Nominal T={M['T']}. Background chunks: {M['bg_target']}.",
            "", table(S[cols]))
        D.files(stats, meta, f"e4_quality{tag}{suffix}.csv")


def s_review(D, tag):
    """Analyses requested by the paper pre-draft review (experiments/review_stats.py, runbook_vectors.py)."""
    if tag:
        return
    D.h("Review analyses (pre-draft review requests)")
    if os.path.exists(R("review_versions.json")):
        V = jload("review_versions.json")
        rows = [dict(stream=k, raw_rows=v["raw_rows"], dropped_empty=v["dropped"].get("empty", 0),
                     dropped_identical=v["dropped"].get("identical_after_empty", 0), replay_versions=v["replay_versions"],
                     docs=v["replay_docs"], replay_edits=v["replay_edits"]) for k, v in V.items() if isinstance(v, dict) and "raw_rows" in v]
        D.p("**Version accounting.** The extracted stream file holds `raw_rows` versions; the replay (common.load_stream) drops "
            "whitespace-only versions (revision-deleted text or blanking) and then consecutive text-identical versions:", "",
            table(pd.DataFrame(rows)))
        D.files("review_versions.json")
    else:
        D.pending("version accounting")
    if os.path.exists(R("e1_ci_T128.csv")):
        C = pd.read_csv(R("e1_ci_T128.csv")); J = jload("e1_ci_T128.json")["summary"]
        D.p("", "**E1 at T=128 (per revision, S1) with document-clustered bootstrap 95% CIs** (2,000 resamples over documents); "
            "`dfrac_vs_REC` = paired difference in toks_frac vs REC on the same edits, clustered CI. Content-aligned = REC, "
            "STRUCT-M, SentCDC, GCDC, BCDC, KARA, DIFF[REC], DIFF[STRUCT-M]. Note the achieved sizes differ (`chunk_tok_mean`), "
            "so the raw T=128 differences mix placement and size (the interpolated comparison is in e1_matched_size.csv).", "",
            table(sort_chunkers(C[["stream", "chunker", "chunk_tok_mean", "toks_frac_mean", "toks_frac_ci_lo", "toks_frac_ci_hi",
                                   "dfrac_vs_REC", "dfrac_ci_lo", "dfrac_ci_hi"]]), 4), "",
            "Content-aligned cluster per stream: " + "; ".join(
                f"{s}: means {fmt(v['min_mean'], 3)}-{fmt(v['max_mean'], 3)}, all 95% CIs overlap = {v['all_cis_overlap']}, "
                f"max |paired diff vs REC| = {fmt(v['max_abs_paired_diff_vs_REC'], 3)}" for s, v in J.items()) + ". "
            "Reading: the per-chunker CIs do NOT all overlap and most paired differences vs REC exclude 0, so 'tie' must be "
            "stated as 'within a few points' (magnitudes above), not as statistical equivalence.")
        D.files("e1_ci_T128.csv", "e1_ci_T128.json")
    else:
        D.pending("E1 clustered CIs at T=128")
    if os.path.exists(R("jina_window.json")):
        J = jload("jina_window.json")
        rows = [dict(stream=k, **v) for k, v in J["streams"].items()]
        D.p("", f"**Late-chunking window.** Share of documents whose text fits in one {J['window']}-token jina-v2 window "
            f"({J['content_tokens']} content tokens; jina tokenizer, no special tokens):", "", table(pd.DataFrame(rows)))
        D.files("jina_window.json")
    else:
        D.pending("jina window share")
    if os.path.exists(R("e2_predict_rank.csv")):
        P = pd.read_csv(R("e2_predict_rank.csv")); J = jload("e2_predict_rank.json")["across_chunkers"]
        D.p("", "**Predictor ranking without the WIKI scale factor** (e2_predict_edits.parquet; WIKI is the calibration "
            "stream, so its scaled errors are in-domain). Per edit: Kendall tau-b of predicted_raw vs measured INS, unscaled "
            "APE of the mean and per-edit MdAPE (edits with measured > 0):", "",
            table(sort_chunkers(P[["stream", "chunker", "edits", "measured_mean", "predicted_raw_mean", "ape_mean_unscaled",
                                   "mdape_edit_unscaled", "tau_edit"]])), "",
            "Across the non-trivial predictors (SentCDC, REC, STRUCT-M, SENTWIN; 4 points per stream, so tau is coarse): " +
            "; ".join(f"{k}: tau raw {fmt(v['tau_raw'], 2)}, tau scaled {fmt(v['tau_scaled'], 2)}, MdAPE scaled "
                      f"{fmt(v['mdape_scaled'], 3)}, unscaled {fmt(v['mdape_unscaled'], 3)}" for k, v in J.items()) + ".")
        D.files("e2_predict_rank.csv", "e2_predict_rank.json")
    ks = [f for f in (f"kara_stability_{x}.json" for x in STREAMS) if os.path.exists(R(f))]
    if ks:
        rows = []
        for f in ks:
            j = jload(f)
            for r in j["summary"]:
                rows.append(dict(stream=f.split("_")[-1][:-5], docs=j["docs"], chunker=r["chunker"], measure=r["measure"],
                                 mean=r["mean"], median=r["median"], n=r["size"]))
        D.p("", "**History dependence (experiments/kara_stability.py, T=128).** Key divergence 1 - |K(A) & K(B)| / |K(B)| "
            "per document, mean/median over documents: final layout after the per-revision replay vs the monthly replay "
            "(same final text, different histories), each vs a fresh chunking of the final text, and the divergence after "
            "ONE update from a fresh layout of version i to version i+1 vs the fresh layout of i+1 (<= 10 evenly spaced "
            "edits per document):", "", table(pd.DataFrame(rows)))
        D.files(*ks)
    D.p("", "**Runbook vectors.** Not stored (disk); `experiments/runbook_vectors.py --stream S --chunker C` regenerates the "
        "bge-small vectors of a runbook deterministically from its point file (replays the chunker at the T=200 calibration, "
        "matches chunk keys, writes big-ann-benchmarks .fbin). Checked on 3 K8S documents for REC, KARA, DIFF[REC], SentCDC "
        "and 3 WIKI documents for KARA: 0 keys not reproduced (runner session 3).")


def s_figs(D, tag):
    D.h("Figures")
    figs = sorted(f for f in os.listdir(R("figures")) if f.endswith(".pdf") and ((tag and tag in f) or (not tag and "_smoke" not in f)))
    D.p("\n".join(f"- `results/figures/{f}`" for f in figs) if figs else "none yet")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or R(f"SUMMARY{a.tag}.md")
    D = Doc()
    try:
        commit = subprocess.run(["git", "-C", P3, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "?"
    D.p("# P3 EditStream-RAG: results summary", "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} (+07) by `experiments/summary.py --tag '{a.tag}'` at commit {commit}. "
        "Every number below is read or computed from the named file in results/; sections marked PENDING have no result yet. "
        "Hardware: Apple M3 Ultra (shared by three projects), <= 8 worker threads per job (multi-process drivers: 1 thread per "
        "worker), bge-small-en-v1.5 fp16 on MPS unless stated; versions in each file's `meta`.")
    E = None
    try:
        E = load_e1(a.tag)
    except Exception:
        D.p("E1 load error: " + traceback.format_exc(limit=1))
    for fn in (s_notes, s_t1, lambda D, t: s_e1(D, t, E), s_e1rep, s_e1fid, s_e2, s_e3, s_e4, s_e5, s_e6, s_e7, s_e8, s_e9, s_tierB, s_review, s_figs):
        try:
            fn(D, a.tag)
        except Exception as ex:
            D.p("", f"**ERROR** in this section: {type(ex).__name__}: {ex}")
    open(out, "w").write("\n".join(D.lines) + "\n")
    print("wrote", out, len(D.lines), "lines")


if __name__ == "__main__":
    main()
