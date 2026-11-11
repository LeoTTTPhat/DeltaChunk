"""Invariants of text.py, diff.py, the chunkers and delta.py (design.md §2, §6, §7)."""
import os, random, re
from collections import Counter

import pytest

from deltachunk.chunkers import TIER_A, Version, make
from deltachunk.chunkers.base import Recursive
from deltachunk.delta import Runbook, ScopeState, apply_edit, coalesce, multiset_diff
from deltachunk.diff import edit_class, regions
from deltachunk.text import canon, normalize, ntok, segment, split_sentences

P3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS = lambda s: re.sub(r"\s", "", s)

MD = """---
title: Pods
weight: 10
---

# Pods

Pods are the smallest deployable units of computing that you can create and manage in Kubernetes.
A Pod is a group of one or more containers, with shared storage and network resources, and a
specification for how to run the containers.

## Using Pods

- item one is here
- item two continues
  on the next line

```yaml
apiVersion: v1
kind: Pod
```

| a | b |
|---|---|
| 1 | 2 |

Short. This sentence is long enough to stand alone as a unit of text, e.g. with an abbreviation.
"""

RST = """:mod:`pathlib` --- Paths
========================

.. module:: pathlib
   :synopsis: Object-oriented filesystem paths

This module offers classes representing filesystem paths with semantics appropriate
for different operating systems.  Path classes are divided between pure paths and
concrete paths.

Basic use
---------

Importing the main class::

   >>> from pathlib import Path
   >>> p = Path('.')

.. function:: foo(x)

   Return the thing.  It is computed lazily.
"""


def _real_docs(stream, k=3):
    f = os.path.join(P3, "data", "streams", f"{stream}.parquet")
    if not os.path.exists(f):
        return []
    import pandas as pd
    df = pd.read_parquet(f, columns=["doc_id", "text"])
    ids = sorted(df.doc_id.unique())[:k]
    return [df[df.doc_id == i].text.iloc[-1] for i in ids]


DOCS = [MD, RST] + _real_docs("k8s") + _real_docs("cpy") + _real_docs("wiki")


# ----------------------------------------------------------------------------- text
@pytest.mark.parametrize("doc", DOCS)
def test_units_cover_text_and_bounded(doc):
    u = segment(doc)
    assert NS(canon(u)) == NS(normalize(doc))
    assert all(x.tok <= 480 for x in u)
    assert [x.identity for x in segment(doc)] == [x.identity for x in u]   # deterministic
    assert u[-1].sep == "" and all(x.sep for x in u[:-1])


def test_whitespace_and_reflow_invariance():
    a = "First sentence is right here. Second sentence follows it closely.\n\nNext paragraph text goes here."
    b = "First  sentence is right\nhere.   Second sentence follows\tit closely.  \n\nNext paragraph text goes here.   "
    assert [x.identity for x in segment(a)] == [x.identity for x in segment(b)]
    assert edit_class(a, b, segment(a), segment(b)) == "ws"


def test_sentence_split_and_glue():
    s = split_sentences("Dr. Smith went to Washington. He arrived at 5 p.m. on Monday. It rained, e.g. a lot.")
    assert s == ["Dr. Smith went to Washington.", "He arrived at 5 p.m. on Monday.", "It rained, e.g. a lot."]
    u = segment("Short. This sentence is long enough to stand alone as a unit of text.")
    assert len(u) == 1   # "Short." (< 8 tokens) glued to the next sentence


def test_block_kinds():
    k = Counter(x.kind for x in segment(MD))
    assert k["yaml"] == 4 and k["heading"] == 2 and k["list"] == 2 and k["code"] == 4 and k["table"] == 3
    r = Counter(x.kind for x in segment(RST))
    assert r["heading"] == 2 and r["code"] >= 2 and r["directive"] >= 2


# ----------------------------------------------------------------------------- diff
def test_diff_classes():
    a = "One sentence is here in the text. Another sentence is here in the text.\n\nThird paragraph is here now."
    split = "One sentence is here in the text.\n\nAnother sentence is here in the text.\n\nThird paragraph is here now."
    ua, us = segment(a), segment(split)
    assert regions(ua, us) and edit_class(a, split, ua, us) == "structure"
    h1, h2 = "# Title\n\n" + a, "## Title\n\n" + a
    assert regions(segment(h1), segment(h2))
    assert edit_class(a, a, ua, ua) == "noop"
    b = a.replace("Third", "Fourth")
    assert edit_class(a, b, ua, segment(b)) == "single"
    c = b.replace("One", "Uno")
    assert edit_class(a, c, ua, segment(c)) == "multi"


# ----------------------------------------------------------------------------- chunkers
def _covers(v, cs, name):
    if name in ("STL", "STL+snap"):   # Steadlith chunk text is its own normalized rendering
        return NS("".join(c.text for c in cs)) == NS(v.text)
    if name == "FIXO":
        return all(v.text[c.start:c.end] == c.text for c in cs)
    pos = 0
    for c in cs:
        if c.start < pos or v.text[c.start:c.end] != c.text or v.text[pos:c.start].strip():
            return False
        pos = c.end
    return not v.text[pos:].strip()


@pytest.mark.parametrize("name", TIER_A)
def test_chunker_coverage_bounds_determinism(name):
    for doc in DOCS:
        v = Version(doc)
        ch = make(name)
        cs = ch.update(v, None)[0] if ch.stateful else ch.chunk(v)
        assert cs and _covers(v, cs, name), name
        assert max(ntok(c.text) for c in cs) <= 512
        ch2 = make(name)
        cs2 = ch2.update(Version(doc), None)[0] if ch2.stateful else ch2.chunk(Version(doc))
        assert [c.key for c in cs] == [c.key for c in cs2]


def test_rec_equals_langchain_hf_splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    lc = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tok, chunk_size=200, chunk_overlap=0)
    ours = Recursive(200)
    for doc in DOCS:
        v = Version(doc)
        assert ours.split_text(v.text) == lc.split_text(v.text)


@pytest.mark.parametrize("name", ["DIFF[REC]", "DIFF[STRUCT-M]", "KARA"])
def test_stateful_unchanged_and_local(name):
    doc = DOCS[-1]
    ch = make(name)
    c0, st = ch.update(Version(doc), None)
    c1, st = ch.update(Version(doc), st)
    assert [c.key for c in c0] == [c.key for c in c1]
    pars = Version(doc).text.split("\n\n")
    edited = "\n\n".join(pars[:-1] + [pars[-1] + " An appended sentence that is new."])
    c2, _ = ch.update(Version(edited), st)
    ins = sum((Counter(c.key for c in c2) - Counter(c.key for c in c1)).values())
    assert 1 <= ins <= 3


def test_doc_reembeds_everything():
    a = DOCS[0]; b = a.replace("smallest", "tiniest")
    ch = make("DOC")
    ka, kb = [c.key for c in ch.chunk(Version(a))], [c.key for c in ch.chunk(Version(b))]
    assert not set(ka) & set(kb)


# ----------------------------------------------------------------------------- delta
def test_multiset_and_scopes_hand_computed():
    tok = {k: 10 for k in "abcdefgxyz"}
    tok["a"] = 3
    st = ScopeState()
    r = apply_edit(st, "d1", ["a", "b", "b", "c"], tok)        # first version: all new
    assert (r["S1_ins"], r["S1_del"], r["S1_toks"]) == (4, 0, 33)
    assert (r["S2_ins"], r["S3_ins"]) == (3, 3)              # distinct keys
    r = apply_edit(st, "d2", ["c", "x"], tok)                # c already live in d1
    assert (r["S1_ins"], r["S2_ins"], r["S3_ins"], r["S2_del"]) == (2, 1, 1, 0)
    r = apply_edit(st, "d1", ["a", "b", "y"], tok)           # drop one b and c; add y
    assert (r["S1_ins"], r["S1_del"]) == (1, 2)
    assert (r["S2_ins"], r["S2_del"]) == (1, 0)              # c still live in d2; b still live once
    r = apply_edit(st, "d2", ["x"], tok)                     # c removed everywhere
    assert (r["S1_del"], r["S2_del"]) == (1, 1)
    r = apply_edit(st, "d2", ["x", "c"], tok)                # c comes back: S2 embeds, S3 cache hit
    assert (r["S2_ins"], r["S3_ins"], r["S3_toks"]) == (1, 0, 0)
    ins, dele = multiset_diff(["a", "a", "b"], ["a", "b", "b"])
    assert ins == Counter({"a": 1}) and dele == Counter({"b": 1})


def test_coalesce():
    day = 86400
    vs = [dict(ts=t) for t in [0, 100, day + 5, day + 6, 40 * day]]
    assert coalesce(vs, "revision") == vs
    # every plan starts at the first version and ends at the final one (B5: same history for every plan)
    assert [v["ts"] for v in coalesce(vs, "daily")] == [0, 100, day + 6, 40 * day]
    assert [v["ts"] for v in coalesce(vs, "monthly")] == [0, day + 6, 40 * day]
    assert [v["ts"] for v in coalesce(vs, "release", [50, 60, day + 5, 50 * day])] == [0, day + 5, 40 * day]
    assert [v["ts"] for v in coalesce(vs, "release", [day + 5])] == [0, day + 5, 40 * day]
    assert [v["ts"] for v in coalesce(vs, "release", [])] == [0, 40 * day]
    import pytest
    with pytest.raises(ValueError):
        coalesce([dict(ts=5), dict(ts=3)], "daily")


def test_runbook():
    rb = Runbook("wiki-REC")
    rb.edit(["a", "b", "c"], []); rb.edit(["d"], ["a", "b"]); rb.edit([], ["d"])
    assert rb.steps == [("insert", 0, 3), ("delete", 0, 2), ("insert", 3, 4), ("delete", 3, 4)]
    y = rb.to_yaml()
    assert "max_pts: 3" in y and 'operation: "delete"' in y
