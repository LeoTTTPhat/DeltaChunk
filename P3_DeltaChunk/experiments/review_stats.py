"""Analyses requested by the paper pre-draft review (PLAN.md "Orchestrator notes", Nov 9 22:05). Read-only over existing
inputs; each subcommand writes one results file.

  versions  WIKI/K8S/CPY version accounting: rows in the extracted stream file vs versions the replay uses
            (load_stream drops whitespace-only versions, then text-identical consecutive versions)
            -> results/review_versions.json
  e1ci      document-clustered bootstrap 95% CIs (2,000 resamples over documents) of the per-edit re-embedded share
            toks_frac and of INS at T=128, per-revision sync, scope S1, per stream x chunker; paired difference vs REC
            on the same edits (clustered CI) -> results/e1_ci_T128.csv
  jina      share of documents whose text fits in one 8,192-token jina-v2 window (8,190 content tokens + CLS/SEP, as in
            deltachunk.late), final versions per stream and all WIKI versions -> results/jina_window.json
  e3qmde    E3q (sentence probes): clustered CI of maintained - fresh and the minimum detectable effect (MDE, two-sided
            alpha 0.05, power 0.8: MDE = 2.80 x clustered-bootstrap SE of the paired difference) per chunker and t
            -> results/e3q_ci_mde.csv
  predict   predictor ranking without the WIKI scale: Kendall tau across non-trivial predictors and per edit, unscaled
            APE/MdAPE -> results/e2_predict_rank.{csv,json}
usage: PYTHONPATH=src python experiments/review_stats.py {versions|e1ci|jina|e3qmde|predict}
"""
import json, os, sys

import numpy as np
import pandas as pd
from common import P3, RESULTS, STREAMS, load_stream, run_meta

CONTENT_ALIGNED = ["REC", "STRUCT-M", "SentCDC", "GCDC", "BCDC", "KARA", "DIFF[REC]", "DIFF[STRUCT-M]"]
Z = 1.959964 + 0.841621   # two-sided alpha 0.05 + power 0.8


def boot_ratio(v, c, B=2000, seed=0):
    """Document-clustered bootstrap of a mean over edits: resample documents, mean = sum / count."""
    g = pd.DataFrame({"v": np.asarray(v, float), "c": np.asarray(c)}).groupby("c").v.agg(["sum", "size"])
    s, n = g["sum"].values, g["size"].values
    idx = np.random.default_rng(seed).integers(0, len(s), size=(B, len(s)))
    boots = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(boots.std(ddof=1)), len(s)


def versions():
    import pyarrow.parquet as pq
    out = {}
    for s in ["wiki", "k8s", "cpy"]:
        raw = pq.read_table(os.path.join(P3, "data", "streams", STREAMS[s]), columns=["doc_id"]).to_pandas()
        df = load_stream(s, columns=["doc_id", "rev_idx", "text"])
        out[s] = dict(file=STREAMS[s], raw_rows=len(raw), raw_docs=int(raw.doc_id.nunique()),
                      dropped=df.attrs["dropped"], replay_versions=len(df), replay_docs=int(df.doc_id.nunique()),
                      replay_edits=len(df) - int(df.doc_id.nunique()))
        print(s, out[s], flush=True)
        del df
    out["note"] = ("raw_rows = versions in the extracted stream file; load_stream drops whitespace-only versions (revision-deleted "
                   "text or blanking) and then consecutive text-identical versions; replay_edits = replay_versions - docs.")
    out["meta"] = run_meta()
    json.dump(out, open(os.path.join(RESULTS, "review_versions.json"), "w"), indent=1)


def e1ci():
    cols = ["stream", "doc_id", "chunker", "rev_old", "rev_new", "toks_frac", "ins", "chunk_tok_mean"]
    D = pd.read_parquet(os.path.join(RESULTS, "e1_edits.parquet"), columns=cols,
                        filters=[("T", "==", 128), ("interval", "==", "revision")])
    rows = []
    for (s, ch), g in D.groupby(["stream", "chunker"]):
        r = dict(stream=s, chunker=ch, content_aligned=ch in CONTENT_ALIGNED, edits=len(g),
                 chunk_tok_mean=float(g.chunk_tok_mean.mean()))
        for m in ["toks_frac", "ins"]:
            mu, lo, hi, se, nd = boot_ratio(g[m], g.doc_id)
            r.update({f"{m}_mean": mu, f"{m}_ci_lo": lo, f"{m}_ci_hi": hi}); r["docs"] = nd
        rows.append(r)
    R = pd.DataFrame(rows)
    # paired difference vs REC on the same edits (same doc, rev_old, rev_new)
    key = ["stream", "doc_id", "rev_old", "rev_new"]
    ref = D[D.chunker == "REC"].set_index(key)[["toks_frac", "ins"]]
    for i, r in R.iterrows():
        if r.chunker == "REC":
            continue
        g = D[(D.stream == r.stream) & (D.chunker == r.chunker)].set_index(key)[["toks_frac", "ins"]]
        j = g.join(ref.add_suffix("_ref"), how="inner")
        if not len(j):
            continue
        d = j.toks_frac - j.toks_frac_ref
        mu, lo, hi, se, _ = boot_ratio(d, j.index.get_level_values("doc_id"))
        R.loc[i, ["dfrac_vs_REC", "dfrac_ci_lo", "dfrac_ci_hi", "paired_edits"]] = [mu, lo, hi, len(j)]
    R.to_csv(os.path.join(RESULTS, "e1_ci_T128.csv"), index=False)
    summ = {}
    for s, g in R[R.content_aligned].groupby("stream"):
        summ[s] = dict(min_mean=float(g.toks_frac_mean.min()), max_mean=float(g.toks_frac_mean.max()),
                       lowest_ci_hi=float(g.toks_frac_ci_hi.min()), highest_ci_lo=float(g.toks_frac_ci_lo.max()),
                       all_cis_overlap=bool(g.toks_frac_ci_lo.max() <= g.toks_frac_ci_hi.min()),
                       n_configs=len(g), max_abs_paired_diff_vs_REC=float(g.dfrac_vs_REC.abs().max()))
    json.dump(dict(summary=summ, B=2000, cluster="document", T=128, interval="revision", scope="S1",
                   meta=run_meta()), open(os.path.join(RESULTS, "e1_ci_T128.json"), "w"), indent=1)
    print(R.round(4).to_string()); print(json.dumps(summ, indent=1))


def jina():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-small-en")
    W = 8192 - 2

    def lens(texts, bs=1000):
        out = []
        for i in range(0, len(texts), bs):
            out += [len(x) for x in tok(texts[i:i + bs], add_special_tokens=False)["input_ids"]]
        return np.array(out)
    out = dict(window=8192, content_tokens=W, tokenizer="jinaai/jina-embeddings-v2-small-en", streams={})
    for s in ["wiki", "k8s", "cpy"]:
        df = load_stream(s, columns=["doc_id", "rev_idx", "text"])
        fin = df.groupby("doc_id", sort=False).text.last()
        n = lens(list(fin.values))
        r = dict(docs=len(n), final_fit_share=float((n <= W).mean()), final_fit_docs=int((n <= W).sum()),
                 final_tok_p50=float(np.median(n)), final_tok_p90=float(np.percentile(n, 90)), final_tok_max=int(n.max()))
        if s == "wiki":
            m = lens(list(df.text.values)); fits = pd.Series(m <= W, index=df.doc_id.values)
            r.update(version_fit_share=float(fits.mean()), versions=len(m),
                     docs_all_versions_fit=int(fits.groupby(level=0).all().sum()))
        out["streams"][s] = r; print(s, r, flush=True)
        del df
    out["meta"] = run_meta()
    json.dump(out, open(os.path.join(RESULTS, "jina_window.json"), "w"), indent=1)


def e3qmde():
    D = pd.read_csv(os.path.join(RESULTS, "e3_quality.csv"))
    rows = []
    for (ch, t), g in D.groupby(["chunker", "t"]):
        d = g.maintained - g.fresh
        mu, lo, hi, se, nd = boot_ratio(d, g.doc_id)
        disc = int((d != 0).sum())
        rows.append(dict(chunker=ch, t=t, probes=len(g), docs=nd, maintained=float(g.maintained.mean()),
                         fresh=float(g.fresh.mean()), gap=mu, gap_ci_lo=lo, gap_ci_hi=hi, discordant=disc,
                         mde_boot=Z * se, mde_mcnemar_unclustered=float(Z * np.sqrt(max(disc, 1) / len(g) / len(g)))))
    R = pd.DataFrame(rows)
    # pooled over t > 0 per chunker (probes at different t are different checkpoints of the same documents: cluster = doc)
    for ch, g in D[D.t > 0].groupby("chunker"):
        d = g.maintained - g.fresh; mu, lo, hi, se, nd = boot_ratio(d, g.doc_id)
        R = pd.concat([R, pd.DataFrame([dict(chunker=ch, t="pooled_t>0", probes=len(g), docs=nd, maintained=float(g.maintained.mean()),
                                              fresh=float(g.fresh.mean()), gap=mu, gap_ci_lo=lo, gap_ci_hi=hi,
                                              discordant=int((d != 0).sum()), mde_boot=Z * se)])])
    R.to_csv(os.path.join(RESULTS, "e3q_ci_mde.csv"), index=False)
    print(R.round(4).to_string())


def predict():
    """Predictor ranking quality without the WIKI scale factor: per stream, Kendall tau-b between mean predicted (raw and
    WIKI-scaled) and measured INS across the non-trivial predictors (SentCDC, REC, STRUCT-M, SENTWIN; FIX and DOC are
    near-trivial); per stream x chunker, per-edit Kendall tau-b (predicted_raw vs measured), unscaled APE of the mean and
    per-edit MdAPE over edits with measured > 0."""
    from scipy.stats import kendalltau
    E = pd.read_parquet(os.path.join(RESULTS, "e2_predict_edits.parquet"))
    J = pd.DataFrame(json.load(open(os.path.join(RESULTS, "e2_predict.json")))["rows"])
    NT = ["SentCDC", "REC", "STRUCT-M", "SENTWIN"]
    rows = []
    for (s, ch), g in E.groupby(["stream", "chunker"]):
        tau, p = kendalltau(g.predicted_raw, g.measured)
        pos = g[g.measured > 0]
        rows.append(dict(stream=s, chunker=ch, edits=len(g), measured_mean=float(g.measured.mean()),
                         predicted_raw_mean=float(g.predicted_raw.mean()),
                         ape_mean_unscaled=float(abs(g.predicted_raw.mean() - g.measured.mean()) / g.measured.mean()),
                         mdape_edit_unscaled=float(np.median(np.abs(pos.predicted_raw - pos.measured) / pos.measured)),
                         tau_edit=float(tau), tau_edit_p=float(p)))
    R = pd.DataFrame(rows)
    across = {}
    for s, g in J[J.chunker.isin(NT)].groupby("stream"):
        t_raw = kendalltau(g.predicted_raw, g.measured)[0]; t_sc = kendalltau(g.predicted, g.measured)[0]
        across[s] = dict(chunkers=list(g.chunker), tau_raw=float(t_raw), tau_scaled=float(t_sc),
                         mdape_scaled=float(g.ape.median()),
                         mdape_unscaled=float((abs(g.predicted_raw - g.measured) / g.measured).median()))
    R.to_csv(os.path.join(RESULTS, "e2_predict_rank.csv"), index=False)
    json.dump(dict(across_chunkers=across, nontrivial=NT, meta=run_meta()), open(os.path.join(RESULTS, "e2_predict_rank.json"), "w"), indent=1)
    print(R.round(3).to_string()); print(json.dumps(across, indent=1))


if __name__ == "__main__":
    {"versions": versions, "e1ci": e1ci, "jina": jina, "e3qmde": e3qmde, "predict": predict}[sys.argv[1]]()
