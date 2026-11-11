"""E4: retrieval quality at matched size (bge-small, exact inner-product search).

Corpus per chunker = chunks of the final snapshots of every document of the QA streams (history-dependent chunkers:
the layout reached by replaying the full history) + background distractor chunks of non-sampled simplewiki articles.
B5 fixes (review finding B4-E4): (1) `--bg-mode native` (default) chunks the background with the chunker under test
(`rec` = the build's REC-chunked background for every chunker; `none` = no-background control); (2) packing fills the
budget B by skipping chunks that do not fit (rank order kept) and logs the tokens used; (3) a hit is SPAN coverage:
the packed chunks of the evidence document cover >= 50% of the evidence sentence's characters and all of the answer
span inside it (`recall@B`); the build's lexical rule (>= 50% of evidence words anywhere in the context + answer
string) is kept as `recall_lex@B` for comparison. nDCG@10: relevant = chunk of the evidence document covering >= 50% of
the evidence span. The packed context goes to E9.
E-phase fix (Nov 9, runner session 1): the calibration point is `--T` (default 128, env E4_T). The build used T=200,
where SentCDC (WIKI 129, K8S 176), STRUCT-M/GCDC (WIKI 157/158) cannot reach the target, so "quality at matched size"
was not size-matched; T=128 is the per-stream size-matched point (every chunker within 7%, results/calibration.json
notes). The achieved mean chunk tokens per stream and chunker are written to the meta JSON (`achieved_tok`).
Runner session 2 (Nov 9 20:00): memwatch stopped E4 three times in 15 min (machine at 470-485 GB) and every attempt
restarted from the first chunker. Each chunker's rows, contexts and achieved sizes are now checkpointed in
results/raw/e4/<config hash>/<chunker>.pkl (hash over src/, the arguments and the question ids) and skipped on resume.
usage: python experiments/e4.py [--T 128] [--streams wiki,k8s,cpy] [--chunkers ...] [--bg 30000] [--bg-mode native|rec|none] [--tag _smoke] [--qtag _smoke]
"""
import argparse, hashlib, json, os, pickle, re, time

import numpy as np
import pandas as pd
from common import P3, RAW, RESULTS, calibration, code_hash, doc_groups, load_stream, run_meta

from deltachunk.chunkers import Version, make
from deltachunk.embed import Embedder
from deltachunk.index import exact_topk
from deltachunk.text import ntok_many

QUALITY_CHUNKERS = ["FIX", "REC", "STRUCT-M", "STRUCT-NM", "SentCDC", "STL", "KARA"]
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_W = re.compile(r"\w+")


def final_chunks(name, params, vers):
    """(chunks, canonical text of the indexed version). '<X>-fresh' = the history-dependent chunker X applied to the
    indexed version alone (no history: KARA's own fresh layout, DIFF[b]'s base layout)."""
    fresh = name.endswith("-fresh")
    ch = make(name[:-len("-fresh")] if fresh else name, params)
    if ch.stateful and not fresh:
        st = None
        for v in vers:
            V = Version(v["text"]); cs, st = ch.update(V, st)
        return cs, V.text
    V = Version(vers[-1]["text"])
    return (ch.update(V, None)[0] if ch.stateful else ch.chunk(V)), V.text


def ev_span(doc_text: str, q: dict):
    """(evidence span, answer span) in the indexed version's canonical text, or None if the evidence is absent."""
    e = doc_text.find(q["evidence"])
    if e < 0:
        return None
    a = q["evidence"].lower().find(q["answer"].lower())
    a = (e + a, e + a + len(q["answer"])) if a >= 0 else (e, e + len(q["evidence"]))
    return (e, e + len(q["evidence"])), a


def covered(spans, lo, hi) -> int:
    """Characters of [lo, hi) covered by the union of spans."""
    pts = sorted((max(lo, a), min(hi, b)) for a, b in spans if b > lo and a < hi)
    tot, cur = 0, lo
    for a, b in pts:
        a = max(a, cur)
        if b > a:
            tot += b - a; cur = b
    return tot


def span_hit(spans, sp) -> bool:
    if sp is None:
        return False
    (e0, e1), (a0, a1) = sp
    return covered(spans, e0, e1) >= 0.5 * (e1 - e0) and covered(spans, a0, a1) == a1 - a0


def evidence_hit(text_lc: str, ev_words: list[str], answer_lc: str) -> bool:
    if answer_lc not in text_lc:
        return False
    words = set(_W.findall(text_lc))
    return sum(w in words for w in ev_words) >= 0.5 * max(1, len(ev_words))


def sentence_windows(chunks, text):
    """Map each single-unit vector to its context, with two neighbors on each side."""
    spans = [(chunks[max(0, i - 2)].start, chunks[min(len(chunks) - 1, i + 2)].end) for i in range(len(chunks))]
    return [text[start:end] for start, end in spans], spans


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--streams", default="wiki,k8s,cpy")
    ap.add_argument("--chunkers", default=",".join(QUALITY_CHUNKERS)); ap.add_argument("--bg", type=int, default=30000)
    ap.add_argument("--bg-mode", default="native", choices=["native", "rec", "none"])
    ap.add_argument("--budgets", default="1024,2048"); ap.add_argument("--tag", default=""); ap.add_argument("--qtag", default="")
    ap.add_argument("--topk", type=int, default=50); ap.add_argument("--docs", type=int)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=[DEFAULT_MODEL, "intfloat/e5-small-v2"])
    ap.add_argument("--device", choices=["cpu", "mps"], help="pin the encoder device and isolate scoring checkpoints")
    ap.add_argument("--stale", type=int, default=0, help="index the version this many retained revisions before the final one "
                    "(E9 freshness subset: stale vs maintained index); chunker names get a '-stale<k>' suffix")
    ap.add_argument("--T", default=os.environ.get("E4_T", "128"), help="calibration point (128 = size-matched)")
    a = ap.parse_args(); t0 = time.time()
    cals = {s: calibration(s).get(str(a.T), {}) for s in a.streams.split(",")}
    assert all(cals.values()), f"no calibration for T={a.T}"
    achieved = {}
    qs = []
    for s in a.streams.split(","):
        f = os.path.join(RESULTS, "qa", f"questions_{s}{a.qtag}.jsonl")
        if os.path.exists(f):
            qs += [json.loads(l) for l in open(f)]
    if not qs:
        raise SystemExit("no questions; run qa_gen.py first")
    qh = hashlib.sha256("\n".join(f"{q['qid']}|{q['question']}|{q['evidence']}" for q in qs).encode()).hexdigest()[:12]
    model_key = () if a.model == DEFAULT_MODEL else (a.model,)
    if a.device is not None:
        model_key += (f"device:{a.device}",)
    ck = os.path.join(RAW, "e4", code_hash("e4", a.streams, a.bg, a.bg_mode, a.budgets, a.topk, a.docs, a.stale, a.T, qh,
                                           sorted(cals.items()), *model_key))
    os.makedirs(ck, exist_ok=True); resumed = []
    emb = Embedder(a.model, device=a.device)
    passage_prefix = "passage: " if a.model == "intfloat/e5-small-v2" else ""
    Q = emb.encode_queries([q["question"] for q in qs])
    bgf = os.path.join(P3, "data", "streams", "wiki_bg.parquet")
    bg_raw = list(pd.read_parquet(bgf).text) if (a.bg and a.bg_mode != "none" and os.path.exists(bgf)) else []

    def background(name):
        """Background distractor chunks (wiki calibration), chunked by `name` (native) or REC (rec mode)."""
        if not bg_raw:
            return [], np.zeros((0, 384), np.float32), []
        nm = name.removesuffix("-fresh") if a.bg_mode == "native" else "REC"
        ch = make(nm, cals.get("wiki", {}).get(nm)); out, contexts = [], []
        for t in bg_raw:
            V = Version(t)
            chunks = ch.update(V, None)[0] if ch.stateful else ch.chunk(V)
            anchors = [c.text for c in chunks]
            out += anchors
            contexts += sentence_windows(chunks, V.text)[0] if nm == "SENTWIN" else anchors
            if len(out) >= a.bg:
                break
        contexts = contexts[:a.bg]
        return contexts, emb.encode(out[:a.bg], prefix=passage_prefix), ntok_many(contexts)
    bg_cache = {}
    docs = {s: doc_groups(load_stream(s, a.docs)) for s in a.streams.split(",")}
    budgets = [int(b) for b in a.budgets.split(",")]
    rows, ctx, bg_n = [], {}, 0
    for name in a.chunkers.split(","):
        is_sentwin = name == "SENTWIN"
        ckf = os.path.join(ck, f"{name}{'-w2' if is_sentwin else ''}.pkl")
        if os.path.exists(ckf):
            c = pickle.load(open(ckf, "rb"))
            rows += c["rows"]; ctx.update(c["ctx"]); achieved[name] = c["achieved"]; bg_n = c["bg_n"]; resumed.append(name)
            print(name, "resumed from", os.path.relpath(ckf, P3), flush=True)
            continue
        n_rows0, ctx0 = len(rows), set(ctx)
        texts, contexts, src, spans, doc_text = [], [], [], [], {}
        # Partial checkpoint of the final-snapshot pass (per document, saved every ~2 min): the stateful KARA replay over
        # the full histories takes > 20 min and was stopped by memwatch repeatedly; resume skips finished documents.
        # keyed by what determines the final layouts only (not background/questions), so the no-background control and
        # later E4 variants reuse the replay; kept after the run (tens of MB)
        finf = os.path.join(RAW, "e4", "final_" + code_hash("e4final", name, a.streams, a.docs, a.stale, a.T,
                                                            sorted(cals.items())) + ".pkl")
        fin = pickle.load(open(finf, "rb")) if os.path.exists(finf) else {}
        if fin:
            print(name, "final-snapshot pass resumed:", len(fin), "documents", flush=True)
        t_save = time.time()
        for s, groups in docs.items():
            for d, vers in groups:
                kd = (s, str(d))
                if kd in fin:
                    cs, dt = fin[kd]
                else:
                    cs, dt = final_chunks(name, cals[s].get(name.removesuffix("-fresh")), vers[:max(1, len(vers) - a.stale)])
                    fin[kd] = (cs, dt)
                    if time.time() - t_save > float(os.environ.get("E4_CKPT_S", "120")):
                        pickle.dump(fin, open(finf + ".tmp", "wb")); os.replace(finf + ".tmp", finf); t_save = time.time()
                        print(name, "final-snapshot checkpoint:", len(fin), "documents", flush=True)
                doc_text[(s, str(d))] = dt
                anchors = [c.text for c in cs]
                if is_sentwin:
                    windows, window_spans = sentence_windows(cs, dt)
                    contexts.extend(windows)
                    spans.extend(window_spans)
                else:
                    for c in cs:
                        sp = (c.start, c.end)
                        if c.start < 0:   # adapter without offsets: locate the text
                            i = dt.find(c.text); sp = (i, i + len(c.text)) if i >= 0 else (-1, -1)
                        spans.append(sp)
                texts += anchors; src += [(s, str(d))] * len(cs)
        if True:   # keep the finished pass if encoding/scoring is stopped
            pickle.dump(fin, open(finf + ".tmp", "wb")); os.replace(finf + ".tmp", finf)
        X = emb.encode(texts, prefix=passage_prefix)
        toks = ntok_many(texts)
        _acc = {}
        for (s_, _), tk in zip(src, toks):
            _acc.setdefault(s_, []).append(tk)
        achieved[name] = {s_: round(float(np.mean(v)), 1) for s_, v in _acc.items()}
        if a.bg_mode == "native" or "bg" not in bg_cache:
            bg_cache["bg"] = background(name)
        bg_texts, BG, bg_tok = bg_cache["bg"]
        if is_sentwin:
            all_texts = contexts + bg_texts
            all_tok = ntok_many(contexts) + bg_tok
        else:
            all_texts = texts + bg_texts
            all_tok = toks + bg_tok
        src_all = src + [("bg", None)] * len(bg_texts); spans_all = spans + [(-1, -1)] * len(bg_texts)
        I = exact_topk(np.vstack([X, BG]), Q, a.topk)
        lc = [t.lower() for t in all_texts]
        cname = name + (f"-stale{a.stale}" if a.stale else "")
        for qi, q in enumerate(qs):
            ev = _W.findall(q["evidence"].lower()); ans = q["answer"].lower()
            key_d = (q["stream"], str(q["doc_id"]))
            sp = ev_span(doc_text.get(key_d, ""), q)
            def own(j):
                return src_all[j] == key_d
            rel = [own(j) and sp is not None and covered([spans_all[j]], *sp[0]) >= 0.5 * (sp[0][1] - sp[0][0]) for j in I[qi]]
            row = dict(qid=q["qid"], stream=q["stream"], chunker=cname, proximal=q.get("proximal", False),
                       evidence_present=sp is not None, rank_first=next((r for r, x in enumerate(rel) if x), -1),
                       bg_mode=a.bg_mode, bg_chunks=len(bg_texts))
            row["ndcg10"] = 1 / np.log2(row["rank_first"] + 2) if 0 <= row["rank_first"] < 10 else 0.0  # one relevant item (first hit)
            for B in budgets:
                used, packed, seen_windows = 0, [], set()
                for j in I[qi]:
                    if used + all_tok[j] > B:
                        continue          # skip what does not fit, keep filling in rank order
                    if is_sentwin:
                        key = (src_all[j], spans_all[j], all_texts[j])
                        if key in seen_windows:
                            continue
                        seen_windows.add(key)
                    used += all_tok[j]; packed.append(j)
                row[f"recall@{B}"] = int(span_hit([spans_all[j] for j in packed if own(j)], sp))
                row[f"recall_lex@{B}"] = int(evidence_hit(" ".join(lc[j] for j in packed), ev, ans))
                row[f"used_tok@{B}"] = used
                if B == budgets[0]:
                    ctx[(q["qid"], cname)] = [all_texts[j] for j in packed]
            rows.append(row)
        bg_n = len(bg_texts)
        tmp = ckf + ".tmp"
        pickle.dump(dict(rows=rows[n_rows0:], ctx={k: v for k, v in ctx.items() if k not in ctx0}, achieved=achieved[name],
                         bg_n=bg_n), open(tmp, "wb"))
        os.replace(tmp, ckf)
        print(name, len(texts), "chunks", {f"recall@{B}": round(float(np.mean([r[f'recall@{B}'] for r in rows if r['chunker'] == name + (f"-stale{a.stale}" if a.stale else "")])), 3) for B in budgets}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, f"e4_quality{a.tag}.csv"), index=False)
    with open(os.path.join(RESULTS, "qa", f"contexts{a.tag}.jsonl"), "w") as f:
        for (qid, name), c in ctx.items():
            f.write(json.dumps(dict(qid=qid, chunker=name, context=c)) + "\n")
    meta = run_meta(seconds=round(time.time() - t0, 1), questions=len(qs), background_chunks=bg_n if a.chunkers else 0,
                    resumed_chunkers=resumed, checkpoint=os.path.relpath(ck, P3),
                    bg_mode=a.bg_mode, bg_target=a.bg, embed=emb.stats, T=str(a.T), achieved_tok=achieved,
                    device=emb.device, model=emb.model_name, passage_prefix=passage_prefix,
                    sentence_window_radius=2 if "SENTWIN" in a.chunkers.split(",") else None)
    json.dump(meta, open(os.path.join(RESULTS, f"e4_quality{a.tag}.meta.json"), "w"), indent=1)
    print(df.groupby(["stream", "chunker"])[[f"recall@{B}" for B in budgets] + [f"recall_lex@{budgets[0]}", f"used_tok@{budgets[0]}", "ndcg10"]].mean().round(3))


if __name__ == "__main__":
    main()
