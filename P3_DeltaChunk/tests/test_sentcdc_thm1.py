"""Theorem 1 (corrected): INS <= floor((l'+2r-1)/(r+1)) + 2 + A, DEL <= floor((l+2r-1)/(r+1)) + 2 + A_Z.

Pure-hash model: units are ids with i.i.d. hashes; a chunk's key is the tuple of its unit ids; cuts by maxp_cuts.
"""
import itertools, random
from collections import Counter

import pytest

from deltachunk.chunkers import SentCDC, Version, maxp_cuts, spans_from_cuts
from deltachunk.chunkers.stateful import DiffAnchored
from deltachunk.chunkers.base import StructMerge
from deltachunk.diff import regions
from deltachunk.text import segment


def chunks(ids, h, r):
    return [tuple(ids[s:e]) for s, e in spans_from_cuts(len(ids), maxp_cuts([h[i] for i in ids], r))]


def damage(n, a, b, lnew, hashes, r):
    """old units 0..n-1, replace [a, b) by lnew fresh units n..n+lnew-1; hashes indexed by unit id."""
    old = list(range(n)); new = old[:a] + list(range(n, n + lnew)) + old[b:]
    co, cn = Counter(chunks(old, hashes, r)), Counter(chunks(new, hashes, r))
    return sum((cn - co).values()), sum((co - cn).values())


def bound(l, r, A=0):
    return (l + 2 * r - 1) // (r + 1) + 2 + A


def test_old_bound_counterexample():
    h = [0.49, 0.18, 0.32, 0.89, 0.92, 0.93]   # unit 5 replaces unit 1
    old = [0, 1, 2, 3, 4]; new = [0, 5, 2, 3, 4]
    assert chunks(old, h, 1) == [(0, 1), (2, 3, 4)]
    assert chunks(new, h, 1) == [(0,), (5, 2), (3, 4)]
    ins = sum((Counter(chunks(new, h, 1)) - Counter(chunks(old, h, 1))).values())
    assert ins == 3 and ins <= bound(1, 1)


def test_min_gap():
    rng = random.Random(1)
    for r in range(1, 7):
        for _ in range(300):
            h = [rng.random() for _ in range(rng.randint(2, 80))]
            c = maxp_cuts(h, r)
            assert all(y - x >= r + 1 for x, y in zip(c, c[1:]))


def _exhaustive(N, rs):
    viol = checked = 0
    for r in rs:
        for n in range(0, N + 1):
            for lnew in range(0, N - n + 1):
                tot = n + lnew
                if tot == 0:
                    continue
                for a in range(0, n + 1):
                    for b in range(a, n + 1):
                        if b == a and lnew == 0:
                            continue
                        for perm in itertools.permutations(range(tot)):
                            ins, dele = damage(n, a, b, lnew, [p / tot for p in perm], r)
                            checked += 1
                            if ins > bound(lnew, r) or dele > bound(b - a, r):
                                viol += 1
    return viol, checked


def test_exhaustive_small():
    viol, checked = _exhaustive(6, (1, 2, 3))
    assert checked > 10000 and viol == 0


@pytest.mark.slow
def test_exhaustive_n8():
    viol, checked = _exhaustive(8, (1, 2, 3))
    print("exhaustive n<=8 checked", checked)
    assert viol == 0


def _random_edits(count, seed=0):
    rng = random.Random(seed); viol = 0; worst = 0
    for t in range(count):
        r = rng.randint(1, 6); n = rng.randint(1, 120)
        a = rng.randint(0, n); b = rng.randint(a, min(n, a + rng.choice([0, 1, 1, 2, 3, 5, 10])))
        lnew = rng.choice([0, 0, 1, 1, 1, 2, 3, 5, 10])
        if a == b and lnew == 0:
            lnew = 1
        lam = rng.choice([1.0, 0.5])
        h = [rng.random() * (lam if rng.random() < 0.3 else 1.0) for _ in range(n + lnew)]
        ins, dele = damage(n, a, b, lnew, h, r)
        worst = max(worst, ins - bound(lnew, r))
        viol += ins > bound(lnew, r) or dele > bound(b - a, r)
    return viol, worst


def test_random_edits_quick():
    viol, worst = _random_edits(20000)
    assert viol == 0 and worst <= 0


@pytest.mark.slow
def test_random_edits_1m():
    viol, worst = _random_edits(1_000_000, seed=7)
    assert viol == 0


def _two_regions(rng, r):
    n = rng.randint(10, 120)
    a1 = rng.randint(0, n - 2); b1 = rng.randint(a1, min(n - 1, a1 + 3)); l1 = rng.randint(0, 3)
    a2 = rng.randint(b1 + 1, n); b2 = rng.randint(a2, min(n, a2 + 3)); l2 = rng.randint(0, 3)
    if (a1 == b1 and l1 == 0) or (a2 == b2 and l2 == 0):
        return None
    h = [rng.random() for _ in range(n + l1 + l2)]
    old = list(range(n))
    new = old[:a1] + list(range(n, n + l1)) + old[b1:a2] + list(range(n + l1, n + l1 + l2)) + old[b2:]
    ins = sum((Counter(chunks(new, h, r)) - Counter(chunks(old, h, r))).values())
    # new coordinates of the regions and their zones Z+ = [a-r, b'+r+1] (b' = last new unit index)
    s1, e1 = a1, a1 + l1 - 1
    s2 = a2 - (b1 - a1) + l1; e2 = s2 + l2 - 1
    z1 = (s1 - r, e1 + r + 1); z2 = (s2 - r, e2 + r + 1)
    if z1[1] >= z2[0]:  # overlapping zones: merge into one region spanning both
        b_ = bound(e2 - s1 + 1, r)
    else:
        b_ = bound(l1, r) + bound(l2, r)
    return ins, b_


def test_multi_region_composed_bound():
    rng = random.Random(3); viol = n = 0
    for _ in range(20000):
        r = rng.randint(1, 6); res = _two_regions(rng, r)
        if res is None:
            continue
        n += 1; viol += res[0] > res[1]
    assert n > 10000 and viol == 0


# ----------------------------------------------------------------------------- full pipeline on synthetic text
WORDS = "alpha beta gamma delta river mountain city table window market garden summer winter science music".split()


def _sent(rng):
    w = [rng.choice(WORDS) for _ in range(rng.randint(8, 18))]
    return w[0].capitalize() + " " + " ".join(w[1:]) + "."


def _doc(rng, pars):
    return "\n\n".join(" ".join(_sent(rng) for _ in range(rng.randint(1, 5))) for _ in range(pars))


def _pipeline_check(old_txt, new_txt, r):
    ch = SentCDC(r=r, lam=0.5, T_max=10 ** 9)
    vo, vn = Version(old_txt), Version(new_txt)
    co, cn = Counter(c.key for c in ch.chunk(vo)), Counter(c.key for c in ch.chunk(vn))
    ins = sum((cn - co).values())
    regs = regions(vo.units, vn.units)
    if not regs:
        return ins, 0
    # merge regions whose zones overlap, then sum bounds (anchors: none in this generator)
    zones = []
    for g in regs:
        z = [g.a2 - r, g.b2 - 1 + r + 1, g.a2, g.b2]
        if zones and zones[-1][1] >= z[0]:
            zones[-1][1] = z[1]; zones[-1][3] = z[3]
        else:
            zones.append(z)
    return ins, sum(bound(z[3] - z[2], r) for z in zones)


def test_pipeline_paragraph_split_merge_ws():
    rng = random.Random(11); viol = 0; cases = 0
    for _ in range(300):
        r = rng.randint(1, 4)
        old = _doc(rng, rng.randint(2, 8))
        pars = old.split("\n\n")
        k = rng.randrange(len(pars))
        kind = rng.choice(["split", "merge", "ws", "edit"])
        if kind == "split":
            s = pars[k].split(". ")
            if len(s) < 2:
                continue
            m = rng.randint(1, len(s) - 1)
            pars[k] = ". ".join(s[:m]) + ".\n\n" + ". ".join(s[m:])
        elif kind == "merge" and k + 1 < len(pars):
            pars[k] = pars[k] + " " + pars.pop(k + 1)
        elif kind == "ws":
            pars[k] = pars[k].replace(" ", "  ", 2).replace(". ", ".\n", 1)
        else:
            pars[k] = pars[k] + " " + _sent(rng)
        new = "\n\n".join(pars)
        ins, b = _pipeline_check(old, new, r)
        cases += 1
        if kind == "ws":
            assert ins == 0
        viol += ins > b
    assert cases > 200 and viol == 0


def test_structure_changes_are_nonempty_diffs():
    rng = random.Random(5)
    old = _doc(rng, 3)
    pars = old.split("\n\n")
    s = pars[0].split(". ")
    if len(s) >= 2:
        split = ". ".join(s[:1]) + ".\n\n" + ". ".join(s[1:])
        assert regions(segment(old), segment("\n\n".join([split] + pars[1:])))
    assert regions(segment("# A\n\n" + old), segment("## A\n\n" + old))


def test_diff_history_dependence_counterexample():
    """DIFF is not a function of content: insert-then-delete a paragraph yields a layout different from no edit."""
    found = False
    for seed in range(200):
        rng = random.Random(seed)
        pars = [" ".join(_sent(rng) for _ in range(rng.randint(1, 3))) for _ in range(6)]
        ins = " ".join(_sent(rng) for _ in range(rng.randint(1, 4)))
        k = rng.randint(1, 5)
        t0 = "\n\n".join(pars); t1 = "\n\n".join(pars[:k] + [ins] + pars[k:])
        d = DiffAnchored(StructMerge(T=60), T_small=0)
        _, st = d.update(Version(t0), None); _, st = d.update(Version(t1), st); c2, _ = d.update(Version(t0), st)
        if [c.key for c in c2] != [c.key for c in StructMerge(T=60).chunk(Version(t0))]:
            found = True; break
    assert found
