import numpy as np
from deltachunk.predict import greedy_ins, greedy_pack, sentcdc_ins_exact, sentcdc_ins_mc


def test_mc_matches_bruteforce_small():
    for (n, a, lo, ln, r) in [(5, 2, 1, 1, 1), (6, 0, 1, 2, 1), (6, 3, 2, 0, 2), (5, 5, 0, 2, 1), (4, 1, 1, 1, 3)]:
        ex = sentcdc_ins_exact(n, a, lo, ln, r)
        m, se = sentcdc_ins_mc(n, a, lo, ln, r, samples=6000, seed=1, window=100)
        assert abs(m - ex) < 4 * se + 1e-9, (n, a, lo, ln, r, ex, m, se)


def test_mc_window_truncation_negligible():
    full, se1 = sentcdc_ins_mc(200, 100, 1, 1, 3, samples=3000, seed=2, window=200)
    loc, se2 = sentcdc_ins_mc(200, 100, 1, 1, 3, samples=3000, seed=3)
    assert abs(full - loc) < 4 * np.hypot(se1, se2)


def test_greedy_pack_worst_case_prop2():
    # units of exactly T/2: inserting one unit at the front flips every later pairing (Prop. 2)
    L = {i: 50 for i in range(41)}
    old = list(range(40)); new = [40] + old
    assert greedy_ins(old, new, L, 100) == 21
    assert greedy_pack([50, 50, 120, 30], 100) == [(0, 2), (2, 3), (3, 4)]
