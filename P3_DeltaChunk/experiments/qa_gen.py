"""E9 (generation): chunker-independent QA from 3-sentence windows of final snapshots.

A window is three consecutive prose units of one paragraph (evidence = the middle unit, >= 10 tokens). The LLM writes a
self-contained question whose short answer is copied from the middle sentence. Filters: answer occurs in the evidence
sentence, answer not in the question, answer <= 8 words, question ends with '?'. Edit-proximal flag: the evidence unit
is absent from the version 20 retained revisions earlier.
`--proximal-only` (B5, review finding B4-E5b): candidates are drawn ONLY from evidence units that changed in the last 20
retained edits (absent from version n-20), across all documents of the stream, for E5b's dedicated edit-proximal set
(e.g. `--stream wiki --n 300 --proximal-only --tag _prox`).
usage: python experiments/qa_gen.py --stream wiki --n 1000 [--per_doc 2] [--tag _smoke] [--proximal-only]
"""
import argparse, hashlib, json, os, re, time

from common import RESULTS, doc_groups, load_stream, run_meta

from deltachunk.chunkers import Version
from deltachunk.qa import LLM

PROMPT = """You write test questions for a search engine.
Document title: {title}
Passage:
[BEFORE] {before}
[MIDDLE] {middle}
[AFTER] {after}

Write ONE factual question that is answered by a short span (1 to 6 words) copied exactly from the [MIDDLE] sentence.
The question must make sense on its own without the passage: name the specific subject instead of saying "it" or "the passage".
Answer with JSON only: {{"question": "...", "answer": "..."}}"""


def windows(v: Version):
    U = v.units
    for i in range(1, len(U) - 1):
        if U[i].kind == "prose" and U[i - 1].kind == "prose" and U[i + 1].kind == "prose" and \
                U[i - 1].block == U[i].block == U[i + 1].block and U[i].tok >= 10:
            yield i


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stream", required=True); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--per_doc", type=int, default=2); ap.add_argument("--tag", default=""); ap.add_argument("--docs", type=int)
    ap.add_argument("--oversample", type=float, default=1.6)
    ap.add_argument("--proximal-only", dest="proximal_only", action="store_true")
    a = ap.parse_args(); t0 = time.time()
    df = load_stream(a.stream, a.docs)
    titles = df.groupby("doc_id").title.first().to_dict() if "title" in df.columns else {}
    cands = []
    groups = doc_groups(df)
    want = int(a.n * a.oversample)
    per_doc = a.per_doc if a.proximal_only else max(a.per_doc, -(-want // max(1, len(groups))))
    for doc, vers in groups:
        v = Version(vers[-1]["text"])
        old = Version(vers[max(0, len(vers) - 21)]["text"]) if len(vers) > 1 else v
        old_units = {u.text for u in old.units}
        wl = [i for i in windows(v) if (not a.proximal_only) or (len(vers) > 1 and v.units[i].text not in old_units)]
        ws = sorted(wl, key=lambda i: hashlib.sha256(f"qa:{doc}:{i}".encode()).hexdigest())[:per_doc]
        for i in ws:
            cands.append(dict(stream=a.stream, doc_id=str(doc), unit=i, before=v.units[i - 1].text, evidence=v.units[i].text,
                              after=v.units[i + 1].text, title=titles.get(doc, str(doc)), proximal=v.units[i].text not in old_units))
    cands.sort(key=lambda c: hashlib.sha256(f"qa-order:{c['doc_id']}:{c['unit']}".encode()).hexdigest())
    cands = cands[:want]
    llm = LLM()
    prompts = [PROMPT.format(title=c["title"], before=c["before"], middle=c["evidence"], after=c["after"]) for c in cands]
    outs = llm.chat_many(prompts, max_tokens=120)
    kept, rej = [], dict(parse=0, not_in_evidence=0, in_question=0, too_long=0, not_question=0)
    for c, o in zip(cands, outs):
        m = re.search(r"\{.*\}", o, re.S)
        try:
            js = json.loads(m.group(0)); q, ans = js["question"].strip(), str(js["answer"]).strip().strip(".")
        except Exception:
            rej["parse"] += 1; continue
        if ans.lower() not in c["evidence"].lower(): rej["not_in_evidence"] += 1; continue
        if ans.lower() in q.lower(): rej["in_question"] += 1; continue
        if len(ans.split()) > 8 or not ans: rej["too_long"] += 1; continue
        if not q.endswith("?"): rej["not_question"] += 1; continue
        kept.append(dict(c, question=q, answer=ans, qid=hashlib.sha1(f"{c['doc_id']}:{c['unit']}".encode()).hexdigest()[:12]))
        if len(kept) >= a.n:
            break
    os.makedirs(os.path.join(RESULTS, "qa"), exist_ok=True)
    with open(os.path.join(RESULTS, "qa", f"questions_{a.stream}{a.tag}.jsonl"), "w") as f:
        for k in kept:
            f.write(json.dumps(k) + "\n")
    meta = dict(stream=a.stream, candidates=len(cands), kept=len(kept), rejected=rej, proximal=sum(k["proximal"] for k in kept),
                llm=llm.stats, meta=run_meta(seconds=round(time.time() - t0, 1), llm_model=llm.model))
    json.dump(meta, open(os.path.join(RESULTS, "qa", f"questions_{a.stream}{a.tag}.meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "meta"}))


if __name__ == "__main__":
    main()
