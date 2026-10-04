"""Independent reference of the paper's OA and IGC (Ho, Shu, Chen 2004, TEVC 8(6)): a test oracle, not part of pyiea.

Written from the paper ONLY (Sections II-III, printed pp. 523-527), before reading pyiea.
All objectives are MINIMIZED (as in pyiea).

Interpretation choices (paper is silent or ambiguous):
 R1  Generate_OA (p.523) is transcribed literally, with 1-based i, j.  `bitwise_AND(i-1, mask)`
     is a bit test ("returns the m-th least significant bit of alpha, where mask = 2^(m-1)").
     `mask` starts at n/2 and is halved; because k = j <= N <= n-1 has at most log2(n) bits,
     mask never drops below 1 while k > 0, so integer arithmetic is exact.
 R2  n = 2^ceil(log2(N+1)), also for N = 1 (n = 2).  The paper requires N > 1 for IGC (p.525,
     eq. (3)); the OA itself is well defined for N = 1, so the reference generates it.
 R3  Level 1 = parent 1 segment, level 2 = parent 2 segment (Step 4, p.526).  Positions where the
     parents agree are not factors (Step 1); they keep the common value in every combination.
 R4  Main effect S_jk = sum_t y_t [level of factor j in row t == k] (eq. (2), p.524).  For
     minimization level 1 is better iff S_j1 < S_j2 (p.524).  TIE S_j1 == S_j2: the paper says only
     "levels 1 and 2 have the same contribution"; the reference picks LEVEL 1 (parent 1).
 R5  MED_j = |S_j1 - S_j2|.  C2 = C1 with the factor of smallest MED switched to "the other level"
     (Step 9).  TIE among smallest MEDs: the reference picks the LOWEST factor index.
 R6  C1 and C2 are evaluated (they are "children"; Step 10 needs their fitness).  The reference
     evaluates exactly the n OA rows then C1 then C2, i.e. n + 2 objective calls per IGC with no
     cache.  The paper's "at most 2N fitness evaluations" (abstract, p.525) holds for the n <= 2N
     OA rows only; with the two children the raw count is n + 2.  With a genome cache, row 1 == P1
     (p.527 "P1 is the combination 1 in Step 5") and children that coincide with an OA row would
     not need a new call; `distinct_calls` reports that count too.
 R7  Step 10 (p.526-527): "Select the best two individuals from the n generated combinations,
     C1, C2, and P2 as the final children.  Note that P1 is the combination 1 in Step 5."  So the
     candidate set is the n OA rows (row 1 == P1) + C1 + C2 + P2.  P2's fitness is assumed known
     (it is a population member) - no extra call.  Duplicated genomes (e.g. C1 equal to an OA row)
     are collapsed so the two final children are distinct genomes when >= 2 distinct candidates
     exist; ranking is by objective value, ties broken by candidate order
     rows 1..n, then C1, C2, P2 (stable).  Both choices (dedup, tie order) are NOT in the paper.
 R8  The IGC does not check feasibility; the division phase must make all combinations feasible
     (p.526 "guarantees that all possible combinations ... always remain feasible").
"""

from __future__ import annotations

import math

import numpy as np


def ref_n(N: int) -> int:
    return 2 ** math.ceil(math.log2(N + 1))


def ref_generate_oa(N: int) -> np.ndarray:
    """Algorithm Generate_OA(OA, N), p.523, literal.  Returns levels in {1, 2}, shape (n, N)."""
    n = ref_n(N)
    oa = np.zeros((n, N), dtype=np.int64)
    for i in range(1, n + 1):
        for j in range(1, N + 1):
            level = 0
            k = j
            mask = n // 2
            while k > 0:
                if (k % 2) and ((i - 1) & mask) != 0:
                    level = (level + 1) % 2
                k = k // 2
                mask = mask // 2
            oa[i - 1, j - 1] = level + 1
    return oa


def ref_decode(p1, p2, segments, levels) -> np.ndarray:
    """Combination: start from P1, copy P2's alleles on factors at level 2."""
    g = np.array(p1, copy=True)
    for seg, lv in zip(segments, levels):
        if lv == 2:
            g[list(seg)] = np.asarray(p2)[list(seg)]
    return g


def ref_igc(p1, p2, segments, f, f_p2=None):
    """IGC Steps 3-10 for a FIXED list of segments (each a list of positions).

    Returns a dict with rows, genomes, responses, S (N x 2), better levels, MED, C1/C2 genomes and
    objectives, the Step-10 pair and the call counts.
    """
    N = len(segments)
    oa = ref_generate_oa(N)
    n = oa.shape[0]
    genomes = [ref_decode(p1, p2, segments, oa[t]) for t in range(n)]
    y = np.array([f(g) for g in genomes], dtype=float)
    S = np.zeros((N, 2))
    for j in range(N):
        for k in (1, 2):
            S[j, k - 1] = sum(y[t] for t in range(n) if oa[t, j] == k)
    better = np.where(S[:, 1] < S[:, 0], 2, 1)  # R4: tie -> level 1
    med = np.abs(S[:, 0] - S[:, 1])
    jmin = int(np.argmin(med))  # R5: first minimum
    c1_lv = better.copy()
    c2_lv = better.copy()
    c2_lv[jmin] = 3 - c2_lv[jmin]
    c1 = ref_decode(p1, p2, segments, c1_lv)
    c2 = ref_decode(p1, p2, segments, c2_lv)
    y1, y2 = float(f(c1)), float(f(c2))
    # Step 10 (R7)
    yp2 = float(f(p2)) if f_p2 is None else f_p2
    cands = [(genomes[t], y[t]) for t in range(n)] + [(c1, y1), (c2, y2), (np.asarray(p2), yp2)]
    seen, uniq = set(), []
    for g, v in cands:
        key = np.asarray(g, dtype=np.uint8).tobytes()
        if key not in seen:
            seen.add(key)
            uniq.append((g, v))
    order = sorted(range(len(uniq)), key=lambda i: (uniq[i][1], i))
    elite = [uniq[i] for i in order[:2]]
    distinct = {np.asarray(g, dtype=np.uint8).tobytes() for g in [*genomes, c1, c2]}
    return {
        "oa": oa,
        "genomes": genomes,
        "y": y,
        "S": S,
        "better": better,
        "med": med,
        "jmin": jmin,
        "c1": c1,
        "c2": c2,
        "y1": y1,
        "y2": y2,
        "c1_levels": c1_lv,
        "c2_levels": c2_lv,
        "elite": elite,
        "calls": n + 2,
        "distinct_calls": len(distinct),
    }


def _selfcheck():
    # Table II of the paper (p.524): maximize y = 100 x1 - 10 x2 - x3; we minimize -y.
    oa = ref_generate_oa(3)
    assert oa.tolist() == [[1, 1, 1], [1, 2, 2], [2, 1, 2], [2, 2, 1]], oa
    seg = [[0], [1], [2]]
    p1 = np.array([1, 3, 5])
    p2 = np.array([2, 4, 6])

    def f(g):
        return -(100 * g[0] - 10 * g[1] - g[2])

    r = ref_igc(p1, p2, seg, f)
    assert (-r["y"]).tolist() == [65, 54, 164, 155]
    assert (-r["S"]).tolist() == [[119, 319], [229, 209], [220, 218]]
    assert r["better"].tolist() == [2, 1, 1] and r["med"].tolist() == [200, 20, 2]
    assert r["c1"].tolist() == [2, 3, 5] and -r["y1"] == 165
    assert r["c2"].tolist() == [2, 3, 6] and -r["y2"] == 164
    for N in range(1, 70):
        o = ref_generate_oa(N)
        assert (o[0] == 1).all()
    print("ref selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
