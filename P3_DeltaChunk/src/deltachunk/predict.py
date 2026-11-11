"""Non-circular INS predictors (design.md §3.3) and the derived cost calculator.

* SentCDC: exact Monte Carlo of E[INS] under the random-oracle model, for a given edit shape
  (n units, region [a, a+l_old) replaced by l_new units, window r, paragraph bias λ with share p_par of
  paragraph-final units). Only a window of W = 4r+4 units on each side of the edit is simulated (cuts further away
  cannot change the chunks overlapping Z+ except through an improbable run without local minima).
* Greedy packing (REC, STRUCT-M): coupling simulation of two greedy packings of synthetic documents whose paragraph
  lengths are drawn from an empirical distribution; the edit changes, inserts or deletes paragraphs.
* Cost calculator: counts x measured unit costs per regime (labeled derived).
"""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np

from deltachunk.chunkers.base import maxp_cuts, spans_from_cuts


def _chunks(ids, h, r):
    return Counter(tuple(ids[s:e]) for s, e in spans_from_cuts(len(ids), maxp_cuts([h[i] for i in ids], r)))


def sentcdc_ins_exact(n: int, a: int, l_old: int, l_new: int, r: int) -> float:
    """Exact E[INS] by enumerating all hash orderings of the n + l_new distinct units (small n only)."""
    tot = n + l_new; old = list(range(n)); new = old[:a] + list(range(n, tot)) + old[a + l_old:]
    s = c = 0
    for perm in itertools.permutations(range(tot)):
        h = [p / tot for p in perm]
        s += sum((_chunks(new, h, r) - _chunks(old, h, r)).values()); c += 1
    return s / c


def sentcdc_ins_mc(n: int, a: int, l_old: int, l_new: int, r: int, lam: float = 1.0, p_par: float = 0.0,
                   samples: int = 2000, seed: int = 0, window: int | None = None) -> tuple[float, float]:
    """Monte Carlo E[INS] (mean, standard error)."""
    rng = np.random.default_rng(seed)
    W = 4 * r + 4 if window is None else window
    lo = max(0, a - W); hi = min(n, a + l_old + W)
    n_loc = hi - lo; a_loc = a - lo
    tot = n_loc + l_new
    vals = np.empty(samples)
    old = list(range(n_loc)); new = old[:a_loc] + list(range(n_loc, tot)) + old[a_loc + l_old:]
    for t in range(samples):
        h = rng.random(tot)
        if p_par > 0 and lam != 1.0:
            h = np.where(rng.random(tot) < p_par, h * lam, h)
        hl = h.tolist()
        vals[t] = sum((_chunks(new, hl, r) - _chunks(old, hl, r)).values())
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(samples)) if samples > 1 else 0.0


# ----------------------------------------------------------------------------- greedy packing coupling
def greedy_pack(lengths: list[int], T: int) -> list[tuple[int, int]]:
    """Pack paragraphs greedily; a chunk is flushed before a paragraph that would overflow T (long paragraphs alone)."""
    out, s, tok = [], 0, 0
    for i, L in enumerate(lengths):
        if i > s and tok + L > T:
            out.append((s, i)); s, tok = i, 0
        tok += L
    if lengths:
        out.append((s, len(lengths)))
    return out


def greedy_ins(old_ids: list, new_ids: list, length: dict, T: int) -> int:
    co = Counter(tuple(old_ids[s:e]) for s, e in greedy_pack([length[i] for i in old_ids], T))
    cn = Counter(tuple(new_ids[s:e]) for s, e in greedy_pack([length[i] for i in new_ids], T))
    return sum((cn - co).values())


def greedy_ins_mc(par_lengths: np.ndarray, n_par: int, pos: float, kind: str, T: int, samples: int = 500, seed: int = 0,
                  delta_lengths: np.ndarray | None = None) -> float:
    """E[INS] for a greedy packer: synthetic documents of n_par paragraphs with lengths drawn from `par_lengths`;
    edit at relative position `pos`: kind = modify (paragraph gets a new length from delta_lengths/par_lengths),
    insert or delete one paragraph."""
    rng = np.random.default_rng(seed); vals = []
    for _ in range(samples):
        L = list(rng.choice(par_lengths, size=max(1, n_par)))
        length = {i: int(x) for i, x in enumerate(L)}
        old = list(range(len(L))); k = min(len(L) - 1, int(pos * len(L)))
        nid = len(L)
        if kind == "insert":
            length[nid] = int(rng.choice(par_lengths)); new = old[:k] + [nid] + old[k:]
        elif kind == "delete" and len(L) > 1:
            new = old[:k] + old[k + 1:]
        else:
            src = delta_lengths if delta_lengths is not None and len(delta_lengths) else par_lengths
            length[nid] = max(1, int(length[k] + rng.choice(src))) if delta_lengths is not None else int(rng.choice(src))
            new = old[:k] + [nid] + old[k + 1:]
        vals.append(greedy_ins(old, new, length, T))
    return float(np.mean(vals))


# ----------------------------------------------------------------------------- derived cost calculator
def derived_cost(ins: float, dele: float, toks: float, calls: float, unit: dict, chunk_ms: float = 0.0) -> float:
    """Seconds = calls*alpha0 + toks*alpha + ins*beta + dele*gamma + chunk time (all unit costs measured; DERIVED)."""
    return (calls * unit.get("alpha0", 0) + toks * unit.get("alpha", 0) + ins * unit.get("beta", 0)
            + dele * unit.get("gamma", 0) + chunk_ms / 1000.0)
