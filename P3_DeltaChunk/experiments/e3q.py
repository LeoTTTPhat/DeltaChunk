"""E3 (quality part): sentence-probe retrieval gap of history-dependent chunkers vs fresh chunking on WIKI-100.

At checkpoints t in {0, 50, 100, 200, 300} (edit index; documents shorter than t use their last version), for each
pair (maintained = KARA / DIFF[REC] / DIFF[STRUCT-M] replayed from version 0, fresh = same base chunker on the same
version) we build an index over all WIKI-100 documents at that checkpoint (bge-small, exact search) and query with
LLM paraphrases of sentences present in that version (probes; same probes for both arms). Hit = the packed chunks
within 1,024 tokens contain the original sentence. Reported: recall per arm and the paired gap.
usage: python experiments/e3q.py [--docs 100] [--probes 300] [--tag _smoke]
"""
import argparse, hashlib, json, os, re, time

import numpy as np
import pandas as pd
from common import RESULTS, calibration, doc_groups, load_stream, run_meta

from deltachunk.chunkers import Version, make
from deltachunk.embed import Embedder
from deltachunk.index import exact_topk
from deltachunk.qa import LLM
from deltachunk.text import ntok_many

CHECK = [0, 50, 100, 200, 300]
PAIRS = {"KARA": "KARA", "DIFF[REC]": "REC", "DIFF[STRUCT-M]": "STRUCT-M"}
PARA = "Paraphrase the following sentence as a search query (one sentence, different wording, same meaning). Reply with the paraphrase only.\nSentence: {s}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--docs", type=int, default=100); ap.add_argument("--probes", type=int, default=300)
    ap.add_argument("--tag", default=""); ap.add_argument("--B", type=int, default=1024); a = ap.parse_args(); t0 = time.time()
    cal = calibration("wiki").get("200", {})
    groups = doc_groups(load_stream("wiki", a.docs))
    emb = Embedder(); llm = LLM(); rows = []
    allgroups = groups
    for t in CHECK:
        # B5 fix (review finding B4-E3): only documents whose history reaches edit t take part at checkpoint t, so the
        # t axis counts edits actually applied (the build used each document's last version when t exceeded it)
        groups = [(d, vers) for d, vers in allgroups if len(vers) - 1 >= t]
        if not groups:
            continue
        ver = {str(d): vers[t]["text"] for d, vers in groups}
        # probes: prose sentences of the checkpoint versions (salted-hash sample)
        cands = []
        for d, txt in ver.items():
            for i, u in enumerate(Version(txt).units):
                if u.kind == "prose" and u.tok >= 12:
                    cands.append((hashlib.sha256(f"e3q:{t}:{d}:{i}".encode()).hexdigest(), d, u.text))
        cands.sort(); cands = cands[:a.probes]
        paras = llm.chat_many([PARA.format(s=c[2]) for c in cands], max_tokens=80)
        Q = emb.encode_queries(paras)
        for hd, base in PAIRS.items():
            arms = {}
            for arm in ("maintained", "fresh"):
                texts = []
                for d, vers in groups:
                    k = min(t, len(vers) - 1)
                    if arm == "maintained":
                        ch = make(hd, cal.get(hd)); st = None
                        for v in vers[:k + 1]:
                            cs, st = ch.update(Version(v["text"]), st)
                    else:
                        ch = make(base, cal.get(base))
                        cs = ch.update(Version(vers[k]["text"]), None)[0] if ch.stateful else ch.chunk(Version(vers[k]["text"]))
                    texts += [c.text for c in cs]
                X = emb.encode(texts); toks = ntok_many(texts); I = exact_topk(X, Q, 50)
                hits = []
                for qi, c in enumerate(cands):
                    used, packed = 0, []
                    for j in I[qi]:
                        if used + toks[j] > a.B:
                            continue      # skip what does not fit (same packing rule as E4)
                        used += toks[j]; packed.append(texts[j])
                    s = re.sub(r"\s+", " ", c[2]); hits.append(int(any(s in re.sub(r"\s+", " ", p) for p in packed)))
                arms[arm] = hits
            for qi, c in enumerate(cands):
                rows.append(dict(t=t, chunker=hd, probe=qi, doc_id=c[1], docs_at_t=len(groups),
                                 maintained=arms["maintained"][qi], fresh=arms["fresh"][qi]))
            print(t, hd, np.mean(arms["maintained"]), np.mean(arms["fresh"]), flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, f"e3_quality{a.tag}.csv"), index=False)
    s = df.groupby(["chunker", "t"])[["maintained", "fresh", "docs_at_t"]].mean().reset_index()
    s["gap"] = s.maintained - s.fresh
    json.dump(dict(summary=s.to_dict("records"), meta=run_meta(seconds=round(time.time() - t0, 1), llm=llm.stats, embed=emb.stats)),
              open(os.path.join(RESULTS, f"e3_quality{a.tag}.json"), "w"), indent=1)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
