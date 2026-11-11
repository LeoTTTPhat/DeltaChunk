"""Round 2 (C6): boundary-independent retrieval criteria recomputed from the saved packed contexts (B = 1,024 tokens).

For every saved context file results/qa/contexts<tag>.jsonl (one record per question and arm: the packed chunk texts in
rank order) each chunk text is located in the evidence document's indexed canonical text (all occurrences), and three
hit criteria are evaluated:
  span      the primary E4 criterion (evidence-document chunks cover >= 50% of the evidence sentence and the whole
            answer span); recomputed here only to validate the location step against e4_quality<tag>.csv,
  answer    evidence-document chunks cover the whole answer span (no sentence-coverage requirement),
  anywhere  the normalized answer string occurs inside one packed chunk of any document (no attribution).
Paired differences against the within-run REC arm use a document-clustered bootstrap (2,000 resamples) and exact McNemar
tests with Holm correction per run and criterion.
usage: PYTHONPATH=src .venv/bin/python experiments/r2_span.py [--tags "",_nobg,_sharedbg,_sharedbg_matched,...]
"""
import argparse, json, os, re, time

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from common import RESULTS, doc_groups, load_stream, run_meta

from deltachunk.chunkers import Version

TAGS = ["", "_nobg", "_sharedbg", "_sharedbg_matched", "_diff", "_e4_reco", "_e4_sentwin", "_e8q_T200", "_e8q_T320",
        "_e8q_lam1", "_r2matched"]
_WS = re.compile(r"\s+")


def norm(s):
    return _WS.sub(" ", s.lower()).strip()


def covered(spans, lo, hi):
    pts = sorted((max(lo, a), min(hi, b)) for a, b in spans if b > lo and a < hi)
    tot, cur = 0, lo
    for a, b in pts:
        a = max(a, cur)
        if b > a:
            tot += b - a; cur = b
    return tot


def occurrences(text, piece, limit=50, nmap=None):
    """Spans of `piece` in `text`; if there is no exact occurrence, search with whitespace runs collapsed (Steadlith's
    chunk text can differ from the canonical text in whitespace) and map the match back to `text`."""
    out, i = [], text.find(piece)
    while i >= 0 and len(out) < limit:
        out.append((i, i + len(piece))); i = text.find(piece, i + 1)
    if out or nmap is None:
        return out
    ntext, idx = nmap
    npiece = _WS.sub(" ", piece).strip()
    if not npiece:
        return out
    i = ntext.find(npiece)
    while i >= 0 and len(out) < limit:
        out.append((idx[i], idx[i + len(npiece) - 1] + 1)); i = ntext.find(npiece, i + 1)
    return out


def norm_map(text):
    """Whitespace-collapsed text and, per collapsed character, its index in `text`."""
    chars, idx, prev_ws = [], [], False
    for k, ch in enumerate(text):
        if ch.isspace():
            if not prev_ws:
                chars.append(" "); idx.append(k)
            prev_ws = True
        else:
            chars.append(ch); idx.append(k); prev_ws = False
    return "".join(chars), idx


def evaluate(q, doc_text, chunks, nmap=None):
    e = doc_text.find(q["evidence"]) if doc_text else -1
    ans_n = norm(q["answer"])
    anywhere = int(any(ans_n in norm(c) for c in chunks))
    if e < 0:
        return dict(span=0, answer=0, anywhere=anywhere, located=0, evidence_present=False)
    a = q["evidence"].lower().find(q["answer"].lower())
    a0, a1 = (e + a, e + a + len(q["answer"])) if a >= 0 else (e, e + len(q["evidence"]))
    e0, e1 = e, e + len(q["evidence"])
    spans, located = [], 0
    for c in chunks:
        occ = occurrences(doc_text, c, nmap=nmap)
        if occ:
            located += 1; spans += occ
    span = int(covered(spans, e0, e1) >= 0.5 * (e1 - e0) and covered(spans, a0, a1) == a1 - a0)
    answer = int(covered(spans, a0, a1) == a1 - a0)
    return dict(span=span, answer=answer, anywhere=anywhere, located=located, evidence_present=True)


def clustered_diff(d, cl, rng, B=2000):
    g = pd.DataFrame(dict(d=d, cl=cl)).groupby("cl")["d"].agg(["sum", "size"])
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    idx = rng.integers(0, len(s), size=(B, len(s)))
    boots = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", default=",".join(TAGS))
    ap.add_argument("--suffix", default="", help="output file suffix (e.g. _smoke)"); a = ap.parse_args(); t0 = time.time()
    qs = {}
    for s in ("wiki", "k8s", "cpy"):
        for l in open(os.path.join(RESULTS, "qa", f"questions_{s}.jsonl")):
            r = json.loads(l); qs[r["qid"]] = r
    need = {(r["stream"], str(r["doc_id"])) for r in qs.values()}
    texts, nmaps = {}, {}
    for s in ("wiki", "k8s", "cpy"):
        for d, vers in doc_groups(load_stream(s)):
            if (s, str(d)) in need:
                texts[(s, str(d))] = Version(vers[-1]["text"]).text
                nmaps[(s, str(d))] = norm_map(texts[(s, str(d))])
    rows = []
    for tag in a.tags.split(","):
        f = os.path.join(RESULTS, "qa", f"contexts{tag}.jsonl")
        if not os.path.exists(f):
            print("missing", f); continue
        for l in open(f):
            r = json.loads(l)
            q = qs.get(r["qid"])
            if q is None:
                continue
            k = (q["stream"], str(q["doc_id"]))
            ev = evaluate(q, texts.get(k), r["context"], nmaps.get(k))
            rows.append(dict(tag=tag or "_native", qid=r["qid"], chunker=r["chunker"], stream=q["stream"],
                             doc=f'{q["stream"]}:{q["doc_id"]}', n_chunks=len(r["context"]), **ev))
        print(tag or "_native", len(rows), round(time.time() - t0), "s", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(RESULTS, "round2"), exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "round2", f"span_metrics{a.suffix}.csv"), index=False)
    # validation of the location step against the saved primary criterion
    val = []
    for tag in df.tag.unique():
        f = os.path.join(RESULTS, f"e4_quality{'' if tag == '_native' else tag}.csv")
        if not os.path.exists(f):
            continue
        e4 = pd.read_csv(f, usecols=["qid", "chunker", "recall@1024"])
        m = df[df.tag.eq(tag)].merge(e4, on=["qid", "chunker"], how="inner")
        for ch, g in m.groupby("chunker"):
            val.append(dict(tag=tag, chunker=ch, n=len(g), agree=float((g.span == g["recall@1024"]).mean()),
                            span_recomputed=float(g.span.mean()), span_saved=float(g["recall@1024"].mean()),
                            located_share=float((g.located / g.n_chunks.clip(lower=1)).mean())))
    pd.DataFrame(val).to_csv(os.path.join(RESULTS, "round2", f"span_validation{a.suffix}.csv"), index=False)
    # summaries and paired differences vs REC within each run
    rng = np.random.default_rng(20261110)
    summ, pairs = [], []
    for tag, g in df.groupby("tag"):
        ref = [c for c in g.chunker.unique() if c == "REC"]
        for ch, h in g.groupby("chunker"):
            for crit in ("span", "answer", "anywhere"):
                summ.append(dict(tag=tag, chunker=ch, criterion=crit, n=len(h), pooled=float(h[crit].mean()),
                                 **{s: float(h[h.stream.eq(s)][crit].mean()) for s in ("wiki", "k8s", "cpy") if (h.stream == s).any()}))
        if not ref:
            continue
        R = g[g.chunker.eq("REC")].set_index("qid")
        for ch in [c for c in g.chunker.unique() if c != "REC"]:
            X = g[g.chunker.eq(ch)].set_index("qid")
            common_all = X.index.intersection(R.index)
            for scope in ("pooled", "wiki", "k8s", "cpy"):
                common = common_all if scope == "pooled" else common_all[R.loc[common_all, "stream"].eq(scope).to_numpy()]
                if len(common) == 0:
                    continue
                for crit in ("span", "answer", "anywhere"):
                    x, y = X.loc[common, crit].to_numpy(), R.loc[common, crit].to_numpy()
                    diff = x - y
                    m, ci = clustered_diff(diff, R.loc[common, "doc"].to_numpy(), rng)
                    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
                    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
                    pairs.append(dict(tag=tag, scope=scope, chunker=ch, criterion=crit, n=len(common), diff=m, ci_lo=ci[0],
                                      ci_hi=ci[1], mcnemar_p=p, discordant=b + c))
    P = pd.DataFrame(pairs)
    P["p_holm"] = np.nan
    for (tag, scope, crit), g in P.groupby(["tag", "scope", "criterion"]):
        order = g.sort_values("mcnemar_p").index; k = len(order); prev = 0.0
        for i, j in enumerate(order):
            prev = max(prev, min(1.0, (k - i) * P.loc[j, "mcnemar_p"])); P.loc[j, "p_holm"] = prev
    pd.DataFrame(summ).to_csv(os.path.join(RESULTS, "round2", f"span_summary{a.suffix}.csv"), index=False)
    P.to_csv(os.path.join(RESULTS, "round2", f"span_pairs{a.suffix}.csv"), index=False)
    json.dump(run_meta(seconds=round(time.time() - t0, 1), tags=a.tags.split(","), rows=len(df)),
              open(os.path.join(RESULTS, "round2", f"span_metrics{a.suffix}.meta.json"), "w"), indent=1)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(val).to_string())
    print(P.round(4).to_string())


if __name__ == "__main__":
    main()
