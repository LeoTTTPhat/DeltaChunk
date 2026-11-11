"""Chunker registry (experiments.md configuration IDs)."""
from deltachunk.chunkers.base import (ByteCDC, Chunk, Chunker, DocLevel, Fixed, GuardedCDC, Recursive, SentCDC, SentWin,
                                      StructMerge, StructNoMerge, Version, maxp_cuts, spans_from_cuts)
from deltachunk.chunkers.stateful import DiffAnchored, KaraAdapter, SteadlithAdapter

TIER_A = ["FIX", "FIXO", "SENTWIN", "REC", "STRUCT-M", "STRUCT-NM", "SentCDC", "GCDC", "BCDC", "STL", "STL+snap",
          "KARA", "DIFF[REC]", "DIFF[STRUCT-M]", "DOC"]


def make(name: str, cal: dict | None = None):
    """Build a chunker by configuration ID; `cal` overrides size parameters (results/calibration.json)."""
    c = dict(cal or {})
    if name == "FIX": return Fixed(c.get("T", 200), 0)
    if name == "FIXO": return Fixed(c.get("T", 200), c.get("overlap", 40))
    if name == "SENTWIN": return SentWin()
    if name == "REC": return Recursive(c.get("T", 200), 0)
    if name == "REC-O": return Recursive(c.get("T", 200), c.get("overlap", 40))
    if name == "STRUCT-M": return StructMerge(c.get("T", 200))
    if name == "STRUCT-NM": return StructNoMerge()
    if name == "SentCDC": return SentCDC(c.get("r", 3), c.get("lam", 0.5))
    if name == "GCDC": return GuardedCDC(c.get("T_min", 100), c.get("p", 0.3))
    if name == "BCDC": return ByteCDC(c.get("avg", 900))
    if name == "STL": return SteadlithAdapter(c.get("min_tokens", 120), c.get("max_tokens", 320), False)
    if name == "STL+snap": return SteadlithAdapter(c.get("min_tokens", 120), c.get("max_tokens", 320), True)
    if name == "KARA": return KaraAdapter(c.get("chunk_size", 260))
    if name == "DIFF[REC]": return DiffAnchored(Recursive(c.get("T", 200), 0))
    if name == "DIFF[STRUCT-M]": return DiffAnchored(StructMerge(c.get("T", 200)))
    if name == "DOC": return DocLevel(Recursive(c.get("T", 200), 0))
    raise KeyError(name)
