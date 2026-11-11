"""E5b: maintenance policies for late-chunking embeddings over the last `last` edits of WIKI-100 documents.

Chunks: STRUCT-M (calibrated). Encoder: jina-v2-small late chunking (LateEncoder), STRUCT-M boundaries.
Policies replayed over versions n-last .. n of every document:
  EXACT    : every chunk re-embedded with a whole-document forward pass after each edit;
  STALE    : only new chunks get vectors (from the whole-document pass); unchanged chunks keep their old vectors;
  WINDOW-D : new chunks + unchanged chunks within D chunks of a new chunk are refreshed by a forward pass over the
             region spanning those chunks (+-W tokens of context), D in {0,1,2,4}.
Costs: forward tokens and vectors written (index ops) per policy. Quality: evidence recall@1024 of edit-proximal
questions (proximal=True) against the final-version index of each policy; the whole index (every replayed document,
including the other documents' chunks that act as distractors) is maintained under policy p (the realistic design;
the build's docstring wrongly said the distractors had EXACT vectors). Packing skips chunks that do not fit.
B5 (review finding B4-E5b): `--docs-from-q` replays exactly the documents of a dedicated edit-proximal question set
(qa_gen.py --proximal-only, e.g. --qtag _prox) so the question count is ~300, not tens.
usage: python experiments/e5b.py [--docs 100] [--last 20] [--tag _smoke] [--qtag ""] [--docs-from-q]
"""
import argparse, json, os, re, time
from collections import Counter

import numpy as np
import pandas as pd
from common import RESULTS, calibration, doc_groups, load_stream, run_meta

from deltachunk.chunkers import Version, make
from deltachunk.late import LateEncoder
from deltachunk.text import ntok_many

_W = re.compile(r"\w+")


def hit(text_lc, ev, ans):
    if ans not in text_lc:
        return False
    w = set(_W.findall(text_lc)); return sum(x in w for x in ev) >= 0.5 * max(1, len(ev))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--docs", type=int, default=100); ap.add_argument("--last", type=int, default=20)
    ap.add_argument("--tag", default=""); ap.add_argument("--qtag", default=""); ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--docs-from-q", dest="docs_from_q", action="store_true")
    a = ap.parse_args(); t0 = time.time()
    ch = make("STRUCT-M", calibration("wiki").get("200", {}).get("STRUCT-M"))
    enc = LateEncoder()
    if a.docs_from_q:
        qd = {str(json.loads(l)["doc_id"]) for l in open(os.path.join(RESULTS, "qa", f"questions_wiki{a.qtag}.jsonl"))
              if json.loads(l).get("proximal")}
        groups = [(d, v) for d, v in doc_groups(load_stream("wiki")) if str(d) in qd]
    else:
        groups = doc_groups(load_stream("wiki", a.docs))
    policies = ["EXACT", "STALE"] + [f"WINDOW-{D}" for D in (0, 1, 2, 4)]
    final = {p: {} for p in policies}; cost = {p: Counter() for p in policies}; texts_final = {}
    for d, vers in groups:
        vs = [Version(v["text"]) for v in vers[-(a.last + 1):]]
        state = {p: None for p in policies}
        for k, v in enumerate(vs):
            cs = ch.chunk(v); spans = [(c.start, c.end) for c in cs]
            f0 = enc.forward_tokens; full = enc.embed_spans(v.text, spans); ftoks = enc.forward_tokens - f0
            for p in policies:
                if state[p] is None or p == "EXACT":
                    state[p] = {c.text: full[j] for j, c in enumerate(cs)}
                    cost[p]["forward_tokens"] += ftoks; cost[p]["vectors"] += len(cs); continue
                old = state[p]; new_idx = [j for j, c in enumerate(cs) if c.text not in old]
                vec = {}
                if p == "STALE":
                    for j, c in enumerate(cs):
                        vec[c.text] = full[j] if j in new_idx else old[c.text]
                    cost[p]["forward_tokens"] += ftoks if new_idx else 0; cost[p]["vectors"] += len(new_idx)
                else:
                    D = int(p.split("-")[1]); refresh = sorted({x for j in new_idx for x in range(max(0, j - D), min(len(cs), j + D + 1))})
                    if refresh:
                        a0 = cs[refresh[0]].start; b0 = cs[refresh[-1]].end
                        # +-W tokens of context around the refreshed span (approx. 4 chars per token)
                        ra, rb = max(0, a0 - 4 * a.W), min(len(v.text), b0 + 4 * a.W)
                        f0 = enc.forward_tokens; sub = enc.embed_spans(v.text, [spans[j] for j in refresh], region=(ra, rb))
                        cost[p]["forward_tokens"] += enc.forward_tokens - f0
                        rv = {cs[j].text: sub[i] for i, j in enumerate(refresh)}
                    else:
                        rv = {}
                    for j, c in enumerate(cs):
                        vec[c.text] = rv.get(c.text, old.get(c.text, full[j]))
                    cost[p]["vectors"] += len(refresh)
                state[p] = vec
        texts_final[str(d)] = [c.text for c in ch.chunk(vs[-1])]
        for p in policies:
            final[p][str(d)] = state[p]
    qs = []
    f = os.path.join(RESULTS, "qa", f"questions_wiki{a.qtag}.jsonl")
    if os.path.exists(f):
        qs = [json.loads(l) for l in open(f)]
        qs = [q for q in qs if q.get("proximal") and q["doc_id"] in texts_final]
    rows = []
    if qs:
        Q = enc.st.encode([q["question"] for q in qs], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        for p in policies:
            T, X = [], []
            for d, tx in texts_final.items():
                for t in tx:
                    T.append(t); X.append(final[p][d][t])
            X = np.stack(X); toks = ntok_many(T); order = np.argsort(-(Q @ X.T), axis=1)[:, :50]
            for qi, q in enumerate(qs):
                used, packed = 0, []
                for j in order[qi]:
                    if used + toks[j] > 1024:
                        continue
                    used += toks[j]; packed.append(T[j].lower())
                rows.append(dict(policy=p, qid=q["qid"], recall1024=int(hit(" ".join(packed), _W.findall(q["evidence"].lower()), q["answer"].lower()))))
    df = pd.DataFrame(rows)
    out = dict(docs=len(groups), last=a.last, questions=len(qs),
               recall=(df.groupby("policy").recall1024.mean().to_dict() if len(df) else {}),
               cost={p: dict(c) for p, c in cost.items()}, meta=run_meta(seconds=round(time.time() - t0, 1), device=enc.device))
    if len(df):
        df.to_csv(os.path.join(RESULTS, f"e5b_policies{a.tag}.csv"), index=False)
    json.dump(out, open(os.path.join(RESULTS, f"e5b_policies{a.tag}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "meta"}, indent=1))


if __name__ == "__main__":
    main()
