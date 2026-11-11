"""Regression tests for the B5 fixes of the independent build review (code_review.md)."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

P3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(P3, "experiments"))

from deltachunk.chunkers import Version, make  # noqa: E402
from deltachunk.chunkers.base import check_layout, locate  # noqa: E402
from deltachunk.delta import Runbook  # noqa: E402


# --------------------------------------------------------------------------- B4-LOC: locate() spans
def test_locate_searches_from_previous_end():
    text = "x +--+ y\n+--+"
    cs = locate(text, ["x +--+ y", "+--+"])
    assert [(c.start, c.end) for c in cs] == [(0, 8), (9, 13)]
    check_layout(text, cs)


def test_locate_raises_when_not_in_order():
    with pytest.raises(ValueError):
        locate("abc def", ["def", "abc"])


def _streams_or_skip(name, docs):
    from common import doc_groups, load_stream
    try:
        return doc_groups(load_stream(name, docs))
    except Exception as e:  # streams are git-ignored data
        pytest.skip(f"stream {name} unavailable: {e}")


@pytest.mark.parametrize("stream", ["k8s", "cpy"])
def test_rec_and_diff_rec_layouts_ordered_non_overlapping(stream):
    """REC spans and DIFF[REC] layouts stay ordered, non-overlapping and text-exact over real histories
    (the build had overlapping DIFF[REC] chunks in 123/855 CPY versions)."""
    from common import calibration
    cal = calibration(stream).get("200", {})
    groups = _streams_or_skip(stream, 6)
    for _, vers in groups:
        rec, diff = make("REC", cal.get("REC")), make("DIFF[REC]", cal.get("DIFF[REC]")); st = None
        for v in vers[:25]:
            V = Version(v["text"])
            check_layout(V.text, rec.chunk(V), "REC")
            cs, st = diff.update(V, st)          # update() itself asserts the invariant
            check_layout(V.text, cs, "DIFF[REC]")


# --------------------------------------------------------------------------- B4-T1: monotone ts, replay order
@pytest.mark.parametrize("stream", ["wiki", "k8s", "cpy"])
def test_load_stream_monotone_ts_and_no_empty(stream):
    from common import load_stream
    try:
        df = load_stream(stream)
    except Exception as e:
        pytest.skip(str(e))
    assert (df.groupby("doc_id").ts.diff().dropna() >= 0).all()
    assert not df.text.str.strip().eq("").any()
    assert (df.groupby("doc_id").rev_idx.diff().dropna() > 0).all()


def test_global_scopes_asserts_order_and_live_counts():
    from e1_merge import global_scopes
    ok = pd.DataFrame([dict(doc_id="a", rev_idx=0, ins_keys=[1, 2], ins_toks=[3, 4], del_keys=[]),
                       dict(doc_id="a", rev_idx=1, ins_keys=[5], ins_toks=[1], del_keys=[1])])
    sc, _ = global_scopes(ok)
    assert list(sc.S2_ins) == [2, 1] and list(sc.S2_del) == [0, 1]
    bad_order = pd.DataFrame([dict(doc_id="a", rev_idx=1, ins_keys=[1], ins_toks=[1], del_keys=[]),
                              dict(doc_id="a", rev_idx=0, ins_keys=[2], ins_toks=[1], del_keys=[])])
    with pytest.raises(AssertionError):
        global_scopes(bad_order)
    bad_del = pd.DataFrame([dict(doc_id="a", rev_idx=0, ins_keys=[1], ins_toks=[1], del_keys=[9])])
    with pytest.raises(AssertionError):
        global_scopes(bad_del)


# --------------------------------------------------------------------------- B4-G1: git streams on main's first-parent chain
@pytest.mark.parametrize("stream,repo", [("k8s", "data/raw/k8s"), ("cpy", "data/raw/cpython")])
def test_git_stream_pairs_are_ancestors(stream, repo):
    path = os.path.join(P3, "data", "streams", f"{stream}.parquet")
    if not os.path.exists(path) or not os.path.isdir(os.path.join(P3, repo)):
        pytest.skip("git stream data unavailable")
    df = pd.read_parquet(path, columns=["doc_id", "rev_idx", "rev_id"]).sort_values(["doc_id", "rev_idx"])
    prev = df.groupby("doc_id").rev_id.shift()
    pairs = pd.DataFrame({"a": prev, "b": df.rev_id}).dropna().sample(30, random_state=1)
    for a, b in zip(pairs.a, pairs.b):
        rc = subprocess.run(["git", "-C", os.path.join(P3, repo), "merge-base", "--is-ancestor", a, b],
                            capture_output=True, env=dict(os.environ, GIT_NO_LAZY_FETCH="1")).returncode
        assert rc == 0, (a, b)


# --------------------------------------------------------------------------- runbook cadence + points
def test_runbook_search_every_edits_and_points():
    rb = Runbook("x", search_every=2)
    rb.edit(["a"], [], origin=("d", 0)); rb.edit([], []); rb.edit(["b", "c"], ["a"], origin=("d", 1)); rb.edit([], [])
    assert sum(op == "search" for op, _, _ in rb.steps) == 2
    assert rb.points == [(0, "a", "d", 0), (1, "b", "d", 1), (2, "c", "d", 1)]


# --------------------------------------------------------------------------- KARA hashes the counted key
def test_kara_serializes_stripped_units():
    k = make("KARA", {"chunk_size": 100})
    assert k._ch.serialize_units(["Alpha. ", "Beta.\n\n"]) == b"Alpha. Beta."


# --------------------------------------------------------------------------- E7 replay: regimes agree on index state
class _FakeEmb:
    device = "cpu"

    def _encode(self, texts):
        rng = np.random.default_rng(len(texts))
        X = rng.normal(size=(len(texts), 384)).astype(np.float32)
        return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_e7_replay_batched_equals_online_index_state():
    from e7 import replay
    base = "First sentence here. Second one follows.\n\nA new paragraph starts. It ends now."
    texts = [base, base + "\n\nAdded para one.", base.replace("Second", "Another") + "\n\nAdded para one.",
             base, base + "\n\nAdded para two. More."]
    events = [(i, "d", i, t) for i, t in enumerate(texts)] + [(i, "e", i, t[::-1]) for i, t in enumerate(texts[:3])]
    events.sort(key=lambda x: x[:3])
    on = replay("REC", {"REC": {"T": 20}}, events, _FakeEmb(), "online")
    ba = replay("REC", {"REC": {"T": 20}}, events, _FakeEmb(), "batched", batch=64)
    assert on["live_vectors"] == ba["live_vectors"]
    assert on["dele"] == ba["dele"] + ba["cancelled_pending"]
    assert on["ins"] == ba["ins"]


# --------------------------------------------------------------------------- TOST / stats
def test_tost_never_equivalent_without_discordance():
    from stats_quality import n_required, newcombe_paired
    lo, hi = newcombe_paired(3, 0, 0, 2)
    assert lo < -0.02 and hi > 0.02            # n = 5, no discordant pairs: interval is wide, not [0, 0]
    lo, hi = newcombe_paired(500, 30, 28, 442)
    assert -0.02 < lo < 0 < hi < 0.02
    assert n_required(0.0) > 100


def test_balanced_ss_drops_unbalanced_cells():
    from e7_analysis import balanced_ss
    rows = []
    for c, base in (("A", 1.0), ("B", 2.0), ("C", 4.0)):
        for s in ("S1", "S2"):
            for r, m in (("online", 1.0), ("llm", 50.0)):
                if c == "C" and s == "S2":
                    continue                     # C misses a cell -> must be dropped
                rows.append(dict(chunker=c, scope=s, regime=r, cost_per_1k_edits_s=base * m * (1.5 if s == "S2" else 1)))
    out = balanced_ss(pd.DataFrame(rows), ["chunker", "scope", "regime"])
    assert out["chunkers"] == 2 and out["cells"] == 8
    assert abs(sum(out[f] for f in ("chunker", "scope", "regime", "interactions")) - 1) < 1e-9


# --------------------------------------------------------------------------- E4 span hit / E1-rep Bian rule
def test_span_hit_and_coverage():
    from e4 import covered, span_hit
    assert covered([(0, 5), (3, 10)], 2, 8) == 6
    q = dict(evidence="The cat sat on the mat.", answer="the mat")
    doc = "Intro. The cat sat on the mat. End."
    from e4 import ev_span
    sp = ev_span(doc, q)
    assert sp is not None and span_hit([(7, 30)], sp) and not span_hit([(7, 12)], sp)
    assert ev_span("other text", q) is None


def test_bian_recursive_rule_covers_and_bounds():
    from e1rep import BianRecursive
    from deltachunk.text import ntok
    text = "\n\n".join(["Sentence number %d is here. " % i * 30 for i in range(6)])
    V = Version(text); cs = BianRecursive(400).chunk(V)
    check_layout(V.text, cs)
    assert all(ntok(c.text) <= 400 for c in cs)
    assert "".join(V.text.split()) == "".join("".join(c.text for c in cs).split())
