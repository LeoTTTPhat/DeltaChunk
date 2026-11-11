"""Round 2 (C6): second-model check of the generated retrieval questions, plus a blind author audit sheet.

Sample: a stratified, salted-hash sample of the retrieval questions (WIKI: 50 edit-proximal + 50 other; K8S: 100;
CPY: 100). A second local model that did not write the questions (gemma3:27b, Q4_K_M GGUF from the Ollama store,
served by our own llama-server) gets two calls per question, temperature 0:
  (a) blind: the three-unit passage the question was generated from and the question; it answers in at most eight
      words or says UNANSWERABLE. Agreement with the reference answer = normalized containment either way or token
      F1 >= 0.5.
  (b) judged: passage, question and reference answer; JSON verdicts on answerability, answer correctness, whether the
      question is understandable without the passage (self-contained), and whether it has one clear answer.
This is an LLM check, not human validation. Every response is cached on disk (keyed by model + prompt).
The author audit sheet (results/round2/audit_sheet.csv) holds a stratified 100-question subset with the passage and
empty columns for human judgments; it deliberately omits the LLM verdicts (they are in qcheck.csv, joined by qid).
usage: PYTHONPATH=src .venv/bin/python experiments/r2_qcheck.py [--url http://127.0.0.1:8302] [--n-per-stream 100]
       [--limit N]   (--sheet-only writes the audit sheet without any model call)
"""
import argparse, hashlib, json, os, re, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from common import P3, RESULTS, run_meta

MODEL = "gemma3:27b"
CACHE = os.path.join(P3, "data", "llm_cache_r2_qcheck.jsonl")
OUT = os.path.join(RESULTS, "round2")
_W = re.compile(r"\w+")

BLIND = ("Read the passage and answer the question using only the passage. Reply with the answer only, in at most "
         "eight words, copied from the passage when possible. If the passage does not answer the question, reply "
         "UNANSWERABLE.\n\nPassage:\n{passage}\n\nQuestion: {question}\nAnswer:")
JUDGE = ("You check questions for a retrieval benchmark. Each question was written from the passage below, and the "
         "reference answer should be copied from the passage.\n\nPassage:\n{passage}\n\nQuestion: {question}\n"
         "Reference answer: {answer}\n\nAnswer each item with yes or no.\n"
         "answerable: can the question be answered from the passage?\n"
         "answer_correct: is the reference answer a correct answer to the question according to the passage?\n"
         "self_contained: would a reader understand what is asked without seeing the passage (no unresolved words "
         "such as 'this', 'these' or 'the above' that need the passage)?\n"
         "unambiguous: does the question have one clear answer rather than several plausible ones?\n"
         'Reply with JSON only: {{"answerable": "yes|no", "answer_correct": "yes|no", "self_contained": "yes|no", '
         '"unambiguous": "yes|no"}}')


def norm(s):
    return " ".join(_W.findall(s.lower()))


def f1(a, b):
    x, y = norm(a).split(), norm(b).split()
    if not x or not y:
        return 0.0
    common = sum(min(x.count(w), y.count(w)) for w in set(x))
    if common == 0:
        return 0.0
    p, r = common / len(x), common / len(y)
    return 2 * p * r / (p + r)


def load_cache():
    c = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            try:
                r = json.loads(l); c[r["key"]] = r["response"]
            except Exception:
                pass
    return c


def call(url, prompt, json_mode, cache, max_tokens):
    key = hashlib.sha256(f"{MODEL}\n{json_mode}\n{prompt}".encode()).hexdigest()
    if key in cache:
        return cache[key]
    body = dict(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=max_tokens, seed=0)
    # JSON is requested in the prompt only: this llama-server build fails to initialize the JSON grammar sampler
    for attempt in range(4):
        try:
            r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=600)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            cache[key] = text
            with open(CACHE, "a") as f:
                f.write(json.dumps(dict(key=key, model=MODEL, json_mode=json_mode, response=text)) + "\n")
            return text
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))


def sample(n_per_stream):
    rows = []
    for s in ("wiki", "k8s", "cpy"):
        qs = [json.loads(l) for l in open(os.path.join(RESULTS, "qa", f"questions_{s}.jsonl"))]
        for q in qs:
            q["h"] = hashlib.sha256(f"r2-qcheck:{q['qid']}".encode()).hexdigest()
        if s == "wiki":
            prox = sorted([q for q in qs if q.get("proximal")], key=lambda q: q["h"])[:n_per_stream // 2]
            other = sorted([q for q in qs if not q.get("proximal")], key=lambda q: q["h"])[:n_per_stream - len(prox)]
            pick = prox + other
        else:
            pick = sorted(qs, key=lambda q: q["h"])[:n_per_stream]
        for q in pick:
            q["stratum"] = f"{s}-{'proximal' if q.get('proximal') else 'other'}" if s == "wiki" else s
        rows += pick
    return rows


def passage(q):
    return "\n".join(x for x in (q.get("before", ""), q["evidence"], q.get("after", "")) if x)


def audit_sheet(rows):
    # stratified 100-question subset: first 17 per WIKI stratum and 33 per git stream in the salted-hash order
    quota = {"wiki-proximal": 17, "wiki-other": 17, "k8s": 33, "cpy": 33}
    out = []
    for st, k in quota.items():
        sel = sorted([q for q in rows if q["stratum"] == st], key=lambda q: q["h"])[:k]
        out += sel
    sheet = pd.DataFrame([dict(audit_id=i + 1, qid=q["qid"], stream=q["stream"], stratum=q["stratum"], doc_id=q["doc_id"],
                               passage_before=q.get("before", ""), evidence_sentence=q["evidence"],
                               passage_after=q.get("after", ""), question=q["question"], reference_answer=q["answer"],
                               human_answerable="", human_answer_correct="", human_self_contained="",
                               human_unambiguous="", human_notes="") for i, q in enumerate(out)])
    os.makedirs(OUT, exist_ok=True)
    sheet.to_csv(os.path.join(OUT, "audit_sheet.csv"), index=False)
    with open(os.path.join(OUT, "audit_sheet_README.md"), "w") as f:
        f.write("# Author audit sheet for the generated questions\n\n"
                "`audit_sheet.csv` lists 100 retrieval questions in a stratified, salted-hash sample (17 edit-proximal and "
                "17 other WIKI questions, 33 K8S and 33 CPY questions). Each row gives the three-unit passage shown to the "
                "question generator, the question and its reference answer.\n\n"
                "Fill each `human_*` column with `yes` or `no`, judging from the passage only:\n\n"
                "- `human_answerable`: the passage answers the question.\n"
                "- `human_answer_correct`: the reference answer is a correct answer according to the passage.\n"
                "- `human_self_contained`: the question can be understood without the passage.\n"
                "- `human_unambiguous`: the question has one clear answer.\n\n"
                "The sheet omits the second-model verdicts on purpose so that the audit stays blind. They are in "
                "`qcheck.csv` (same `qid`). No human audit has been performed at the time of writing.\n\n"
                "Source text: Simple English Wikipedia (CC BY-SA 4.0), Kubernetes documentation (CC BY 4.0) and CPython "
                "documentation (PSF License); see `DATA_STATEMENT.md` in the release.\n")
    return sheet


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8302")
    ap.add_argument("--n-per-stream", type=int, default=100); ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--sheet-only", action="store_true")
    a = ap.parse_args(); t0 = time.time()
    rows = sample(a.n_per_stream)
    sheet = audit_sheet(rows)
    print("audit sheet", len(sheet), "questions", flush=True)
    if a.sheet_only:
        return
    if a.limit:
        rows = rows[:a.limit]
    cache = load_cache()

    def work(q):
        p = passage(q)
        blind = call(a.url, BLIND.format(passage=p, question=q["question"]), False, cache, 32).strip()
        judged = call(a.url, JUDGE.format(passage=p, question=q["question"], answer=q["answer"]), True, cache, 96)
        try:
            j = json.loads(re.search(r"\{.*\}", judged, re.S).group(0))
        except Exception:
            j = {}
        yes = lambda k: (str(j.get(k, "")).strip().lower().startswith("y")) if k in j else None
        unans = "unanswerable" in blind.lower()
        agree = (not unans) and (norm(q["answer"]) in norm(blind) or norm(blind) in norm(q["answer"]) and norm(blind) != ""
                                 or f1(blind, q["answer"]) >= 0.5)
        return dict(qid=q["qid"], stream=q["stream"], stratum=q["stratum"], doc_id=q["doc_id"], question=q["question"],
                    answer=q["answer"], blind_answer=blind, blind_unanswerable=unans, blind_agree=bool(agree),
                    blind_f1=f1(blind, q["answer"]), answerable=yes("answerable"), answer_correct=yes("answer_correct"),
                    self_contained=yes("self_contained"), unambiguous=yes("unambiguous"), judge_raw=judged.strip()[:300])
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(work, rows))
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(OUT, "qcheck.csv"), index=False)
    rng = np.random.default_rng(20261110)
    summ = []
    for scope, g in [("all", df)] + list(df.groupby("stream")) + list(df.groupby("stratum")):
        row = dict(scope=scope, n=len(g))
        for col in ("blind_agree", "blind_unanswerable", "answerable", "answer_correct", "self_contained", "unambiguous"):
            v = g[col].dropna().astype(float)
            dd = pd.DataFrame(dict(v=v, d=g.loc[v.index, "doc_id"].astype(str))).groupby("d")["v"].agg(["sum", "size"])
            idx = rng.integers(0, len(dd), size=(2000, len(dd)))
            boots = dd["sum"].to_numpy()[idx].sum(1) / dd["size"].to_numpy()[idx].sum(1)
            row[col] = float(v.mean()) if len(v) else None
            row[col + "_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if len(v) else None
            row[col + "_n"] = int(len(v))
        allok = g[["answerable", "answer_correct", "unambiguous"]].fillna(False).all(axis=1)
        row["all_three_yes"] = float(allok.mean())
        summ.append(row)
    meta = run_meta(seconds=round(time.time() - t0, 1), model=MODEL, url=a.url, questions=len(df), calls=2 * len(df),
                    sampling="salted sha256('r2-qcheck:<qid>') order within strata", prompts=dict(blind=BLIND, judge=JUDGE))
    json.dump(dict(summary=summ, meta=meta), open(os.path.join(OUT, "qcheck_summary.json"), "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summ)[["scope", "n", "blind_agree", "blind_unanswerable", "answerable", "answer_correct",
                              "self_contained", "unambiguous", "all_three_yes"]].round(3).to_string())


if __name__ == "__main__":
    main()
