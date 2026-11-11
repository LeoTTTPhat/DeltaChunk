"""Chunk-key deltas under three reuse scopes, sync-interval coalescing, big-ann-benchmarks runbook export.

S1 (per document): INS = |K(d') - K(d)| and DEL = |K(d) - K(d')| as multisets (LlamaIndex / KARA semantics).
S2 (global live set, LangChain index() semantics): a key is embedded iff it is in no live document; index inserts =
   keys whose global refcount goes 0 -> >0, deletes = keys whose refcount goes >0 -> 0.  Embeds counted per distinct key.
S3 (global history cache): a key is embedded iff it was never embedded before; index ops as in S2.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter
from dataclasses import dataclass, field


def multiset_diff(new_keys, old_keys) -> tuple[Counter, Counter]:
    n, o = Counter(new_keys), Counter(old_keys)
    return n - o, o - n


@dataclass
class ScopeState:
    live: Counter = field(default_factory=Counter)      # S2/S3 refcounts over all documents' current keys
    seen: set = field(default_factory=set)              # S3 history cache
    docs: dict = field(default_factory=dict)            # doc -> Counter of its current keys


def apply_edit(st: ScopeState, doc: str, new_keys: list[str], tok_of: dict[str, int]) -> dict:
    """Apply one version of `doc` and return the counts for S1, S2, S3."""
    old = st.docs.get(doc, Counter())
    new = Counter(new_keys)
    ins1, del1 = new - old, old - new
    res = {"S1_ins": sum(ins1.values()), "S1_del": sum(del1.values()),
           "S1_toks": sum(tok_of[k] * c for k, c in ins1.items())}
    # S2/S3: global refcounts
    added, removed = [], []
    for k, c in ins1.items():
        if st.live[k] == 0:
            added.append(k)
        st.live[k] += c
    for k, c in del1.items():
        st.live[k] -= c
        if st.live[k] <= 0:
            del st.live[k]; removed.append(k)
    # S2 embeds: keys of the new version that were not live anywhere before this edit (= `added`, since a key already in
    # this document's old version or in another document stays live)
    res["S2_ins"] = len(added); res["S2_toks"] = sum(tok_of[k] for k in added)
    res["S2_del"] = len(removed)
    emb3 = [k for k in added if k not in st.seen]
    res["S3_ins"] = len(emb3); res["S3_toks"] = sum(tok_of[k] for k in emb3)
    res["S3_del"] = len(removed)
    st.seen.update(added)
    st.docs[doc] = new
    res["_added"], res["_removed"] = added, removed
    return res


# ----------------------------------------------------------------------------- sync intervals
def bucket(ts: int, interval: str) -> object:
    if interval == "revision":
        return None
    d = dt.datetime.fromtimestamp(int(ts), dt.timezone.utc)
    if interval == "daily":
        return d.date()
    if interval == "weekly":
        y, w, _ = d.isocalendar(); return (y, w)
    if interval == "monthly":
        return (d.year, d.month)
    raise ValueError(interval)


def coalesce(versions: list[dict], interval: str, release_ts: list[int] | None = None) -> list[dict]:
    """Sync plan for one document. Every plan starts at the first version (initial indexing) and ends at the final
    version, so all plans cover the same history (B5 fix of review finding B4-F6: coarser plans used to start at the
    last version of the first bucket, absorbing that bucket's edits into initial indexing). In between it keeps the last
    version of each interval bucket; 'release': the last version at or before each release timestamp. `ts` must be
    monotone in revision order (common.load_stream makes it so); this is asserted."""
    if any(b["ts"] < a["ts"] for a, b in zip(versions, versions[1:])):
        raise ValueError("coalesce: timestamps must be monotone within a document (use ts cummax)")
    if interval == "revision" or len(versions) <= 1:
        return list(versions)
    if interval == "release":
        mid, j = [], 0
        for t in sorted(release_ts or []):
            while j < len(versions) and versions[j]["ts"] <= t:
                j += 1
            if j and (not mid or mid[-1] is not versions[j - 1]):
                mid.append(versions[j - 1])
    else:
        mid = [v for i, v in enumerate(versions)
               if i + 1 == len(versions) or bucket(versions[i + 1]["ts"], interval) != bucket(v["ts"], interval)]
    out = [versions[0]]
    for v in mid + [versions[-1]]:
        if v is not out[-1]:
            out.append(v)
    return out


# ----------------------------------------------------------------------------- runbook export
class Runbook:
    """big-ann-benchmarks streaming runbook: insert ranges are contiguous new point ids; deletes are emitted as
    contiguous id runs; a search step after every `search_every` EDITS (not steps). `points` maps every point id to
    its chunk key and the (doc, rev) that inserted it, so the dataset can be materialized (points export)."""

    def __init__(self, name: str, search_every: int = 0):
        self.name, self.search_every = name, search_every
        self.steps, self.next_id, self.ids, self.max_live = [], 0, {}, 0
        self.points, self.n_edits = [], 0

    def edit(self, added: list[str], removed: list[str], origin: tuple | None = None):
        if removed:
            ids = sorted(self.ids.pop(k) for k in removed if k in self.ids)
            run = None
            for i in ids:
                if run and i == run[1]:
                    run[1] += 1
                else:
                    if run: self.steps.append(("delete", run[0], run[1]))
                    run = [i, i + 1]
            if run: self.steps.append(("delete", run[0], run[1]))
        if added:
            s = self.next_id
            for k in added:
                self.points.append((self.next_id, k) + tuple(origin or ()))
                self.ids[k] = self.next_id; self.next_id += 1
            self.steps.append(("insert", s, self.next_id))
        self.max_live = max(self.max_live, len(self.ids))
        self.n_edits += 1
        if self.search_every and self.n_edits % self.search_every == 0:
            self.steps.append(("search", 0, 0))

    def to_yaml(self) -> str:
        lines = [f"{self.name}:", f"  max_pts: {max(1, self.max_live)}"]
        for i, (op, s, e) in enumerate(self.steps, 1):
            lines.append(f"  {i}:"); lines.append(f"    operation: \"{op}\"")
            if op != "search":
                lines.append(f"    start: {s}"); lines.append(f"    end: {e}")
        return "\n".join(lines) + "\n"
