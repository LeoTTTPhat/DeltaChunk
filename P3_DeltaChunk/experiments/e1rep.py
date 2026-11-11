"""E1-rep: replication of Bian et al. (arXiv 2610.31019) on 120 adjacent simplewiki revision pairs.
Their frozen parameters (results/bian_params.json): fixed 304 (non-overlapping), recursive target 400, Steadlith CDC
min/max 176/352, max 512 tokens. Metric: fraction of the new version's embedding-input tokens recomputed under
byte-identical reuse. Both versions must have 1,024-20,000 tokens. One pair per article (salted-hash choice).
Our calibrated REC and SentCDC are reported next to them.
B5 (review finding B4-REP): their recursive rule's separators are implemented as documented (blank lines, then line
breaks, then sentence terminators + whitespace). Their paper does not say whether pieces are merged back up to the
target; since they calibrate every method to a median within 15% (~300 tokens), which a no-merge split cannot reach
(paragraph-sized pieces, median ~50 tokens), `Bian-Recursive400` merges adjacent pieces greedily up to 400 tokens
(LangChain's merge with Bian's separators); the literal no-merge reading is reported as `Bian-Recursive400-nomerge`.
The build's row (LangChain's default separators "\n\n", "\n", " ", "") is kept as `LangChain-REC400`. Size-matched rows (median within 15% of ~300 tokens, as Bian calibrate):
REC/STRUCT-M/SentCDC at the T=320 calibration, plus Bian's chunkers at our T=200 calibration; every row carries
`size_matched_300` (median chunk length within 15% of the Bian-Fixed304 median).
usage: python experiments/e1rep.py [--pairs 120] [--tag _smoke]
"""
import argparse, hashlib, json, os, time
from collections import Counter

import numpy as np
import pandas as pd
from common import RESULTS, calibration, doc_groups, load_stream, run_meta

from deltachunk.chunkers import Version, make
import re

from deltachunk.chunkers.base import Chunk, Chunker, Fixed, Recursive, span_chunk, window_split
from deltachunk.chunkers.stateful import SteadlithAdapter
from deltachunk.text import ntok, ntok_many

_SEPS = [re.compile(r"\n[ \t]*\n\s*"), re.compile(r"\n"), re.compile(r"(?<=[.!?])\s+")]


class BianRecursive(Chunker):
    """Bian et al.'s recursive rule: keep a span within the target, otherwise split it at blank lines, then line
    breaks, then sentence terminators followed by whitespace (recursively, no merging); token windows as last resort."""

    def __init__(self, T: int = 400, name: str = "Bian-Recursive400"):
        self.T, self.name = T, name

    def _rec(self, text, a, b, lvl):
        c = span_chunk(text, a, b)
        if c is None:
            return []
        if ntok(c.text) <= self.T:
            return [c]
        for k in range(lvl, len(_SEPS)):
            cuts = [m.end() for m in _SEPS[k].finditer(text, a, b) if a < m.start() and m.end() < b]
            if cuts:
                out, s0 = [], a
                for e in cuts + [b]:
                    out += self._rec(text, s0, e, k + 1 if k + 1 < len(_SEPS) else k); s0 = e
                return out
        return window_split(text, a, b, self.T)

    def chunk(self, v):
        return self._rec(v.text, 0, len(v.text), 0)


class BianRecursiveMerge(Chunker):
    """Bian's separators with LangChain's greedy merge up to T tokens (see module docstring)."""

    def __init__(self, T: int = 400, name: str = "Bian-Recursive400"):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.T, self.name = T, name
        self._sp = RecursiveCharacterTextSplitter(separators=[r"\n[ \t]*\n\s*", r"\n", r"(?<=[.!?])\s+"], is_separator_regex=True,
                                                  keep_separator="end", chunk_size=T, chunk_overlap=0, length_function=ntok)

    def chunk(self, v):
        from deltachunk.chunkers.base import enforce_max, locate
        return enforce_max(v.text, locate(v.text, self._sp.split_text(v.text)))


def frac(ch, vo, vn):
    co = Counter(c.key for c in ch.chunk(vo)); cn = ch.chunk(vn)
    toks = ntok_many([c.text for c in cn])
    left = Counter(co); rec = 0
    for c, t in zip(cn, toks):
        if left[c.key] > 0:
            left[c.key] -= 1
        else:
            rec += t
    return rec / max(1, sum(toks)), float(np.median(toks)) if toks else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", type=int, default=120); ap.add_argument("--tag", default="")
    ap.add_argument("--stream", default="wiki"); a = ap.parse_args(); t0 = time.time()
    calall = calibration("wiki"); cal = calall.get("200", {}); c320 = calall.get("320", {})
    chunkers = {"Bian-Fixed304": Fixed(304, 0, name="Bian-Fixed304"), "Bian-Recursive400": BianRecursiveMerge(400),
                "Bian-Recursive400-nomerge": BianRecursive(400, name="Bian-Recursive400-nomerge"),
                "LangChain-REC400": Recursive(400),
                "Bian-STL176/352": SteadlithAdapter(176, 352, False, name="Bian-STL"),
                "REC": make("REC", cal.get("REC")), "SentCDC": make("SentCDC", cal.get("SentCDC")),
                "STRUCT-M": make("STRUCT-M", cal.get("STRUCT-M")),
                # size-matched to Bian's ~300-token calibration
                "REC@cal320": make("REC", c320.get("REC")), "STRUCT-M@cal320": make("STRUCT-M", c320.get("STRUCT-M")),
                "SentCDC@cal320": make("SentCDC", c320.get("SentCDC")),
                # Bian's chunkers at our T=200 calibration
                "Bian-Fixed@cal200": make("FIX", cal.get("FIX")), "Bian-STL@cal200": make("STL", cal.get("STL"))}
    rows = []
    for doc, vers in doc_groups(load_stream(a.stream)):
        if len(rows) >= a.pairs:
            break
        V = None
        cand = list(range(1, len(vers)))
        cand.sort(key=lambda i: hashlib.sha256(f"bian-rep:{doc}:{i}".encode()).hexdigest())
        for i in cand[:20]:
            vo, vn = Version(vers[i - 1]["text"]), Version(vers[i]["text"])
            to, tn = sum(u.tok for u in vo.units), sum(u.tok for u in vn.units)
            if 1024 <= to <= 20000 and 1024 <= tn <= 20000 and vo.text != vn.text:
                V = (i, vo, vn, tn); break
        if V is None:
            continue
        i, vo, vn, tn = V
        row = dict(doc_id=str(doc), rev_new=int(vers[i]["rev_idx"]), tok_new=tn)
        for name, ch in chunkers.items():
            f, med = frac(ch, vo, vn); row[name] = f; row[name + "_medlen"] = med
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, f"e1_rep{a.tag}.csv"), index=False)
    ref = float(df["Bian-Fixed304_medlen"].median())
    summ = {name: dict(mean_recompute_pct=round(100 * df[name].mean(), 1), median_len=float(df[name + "_medlen"].median()),
                       size_matched_300=bool(abs(float(df[name + "_medlen"].median()) - ref) <= 0.15 * ref))
            for name in chunkers}
    out = dict(pairs=len(df), summary=summ, bian_english_table_I=json.load(open(os.path.join(RESULTS, "bian_params.json")))["their_english_results_table_I"],
               meta=run_meta(seconds=round(time.time() - t0, 1)))
    json.dump(out, open(os.path.join(RESULTS, f"e1_rep{a.tag}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "meta"}, indent=1))


if __name__ == "__main__":
    main()
