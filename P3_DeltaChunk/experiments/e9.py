"""E9 (answers): RAG answers with qwen3:8b for the E4 contexts (budget 1,024 tokens) of each configuration, plus a
closed-book baseline. Scores: exact match, token F1, contains (normalized). Answer set = first N questions in qid
hash order (stratified by stream as generated).
usage: python experiments/e9.py [--n 600] [--configs REC,STRUCT-M,SentCDC,KARA] [--tag _smoke] [--ctag _smoke] [--qtag _smoke]
"""
import argparse, collections, hashlib, json, os, re, string, time

import pandas as pd
from common import RESULTS, run_meta

from deltachunk.qa import LLM

RAG = """Answer the question using the context. Reply with the short answer only (a few words), no explanation.
Context:
{ctx}

Question: {q}
Answer:"""
CLOSED = """Answer the question. Reply with the short answer only (a few words), no explanation.
Question: {q}
Answer:"""


def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(p, g):
    p, g = norm(p).split(), norm(g).split()
    common = collections.Counter(p) & collections.Counter(g); ns = sum(common.values())
    if not p or not g or ns == 0:
        return 0.0
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--configs", default="REC,STRUCT-M,SentCDC,KARA"); ap.add_argument("--tag", default="")
    ap.add_argument("--ctag", default=""); ap.add_argument("--qtag", default=""); ap.add_argument("--streams", default="wiki,k8s,cpy")
    a = ap.parse_args(); t0 = time.time()
    qs = []
    for s in a.streams.split(","):
        f = os.path.join(RESULTS, "qa", f"questions_{s}{a.qtag}.jsonl")
        if os.path.exists(f):
            qs += [json.loads(l) for l in open(f)]
    qs.sort(key=lambda q: hashlib.sha256(f"e9:{q['qid']}".encode()).hexdigest()); qs = qs[:a.n]
    ctx = {}
    for l in open(os.path.join(RESULTS, "qa", f"contexts{a.ctag}.jsonl")):
        d = json.loads(l); ctx[(d["qid"], d["chunker"])] = d["context"]
    llm = LLM(); rows = []
    for cfg in a.configs.split(",") + ["closed-book"]:
        prompts = [CLOSED.format(q=q["question"]) if cfg == "closed-book" else
                   RAG.format(ctx="\n\n".join(ctx.get((q["qid"], cfg), [])), q=q["question"]) for q in qs]
        outs = llm.chat_many(prompts, max_tokens=32)
        for q, o in zip(qs, outs):
            o = o.strip().split("\n")[0]
            rows.append(dict(qid=q["qid"], stream=q["stream"], config=cfg, proximal=q.get("proximal", False), answer=o,
                             gold=q["answer"], em=int(norm(o) == norm(q["answer"])), f1=f1(o, q["answer"]),
                             contains=int(norm(q["answer"]) in norm(o))))
        print(cfg, pd.DataFrame([r for r in rows if r["config"] == cfg])[["em", "f1", "contains"]].mean().round(3).to_dict(), flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, f"e9_answers{a.tag}.csv"), index=False)
    json.dump(run_meta(seconds=round(time.time() - t0, 1), n=len(qs), llm=llm.stats, llm_model=llm.model),
              open(os.path.join(RESULTS, f"e9_answers{a.tag}.meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
