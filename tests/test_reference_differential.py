"""Differential tests against an independent reference written from the paper only (tests/_reference_iea.py).

The reference was written before pyiea's source was read (independent review, 2026-10-05). The deterministic
part of the IGC (rows, responses, main effects, the smallest-difference factor, C1, C2 and the optional Step 10
with P2) must agree with it exactly, for every kind of objective.
"""

from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pytest
from _reference_iea import ref_generate_oa, ref_igc

import pyiea
from pyiea.evaluator import Evaluator
from pyiea.igc import decode, igc, main_effects
from pyiea.oa import generate_oa
from pyiea.problem import freeze


@pytest.mark.parametrize("n", [*range(1, 65), 100, 127, 128, 255, 256])
def test_oa_equals_the_reference_and_is_balanced_and_orthogonal(n):
    a, b = generate_oa(n).astype(np.int64), ref_generate_oa(n)
    assert np.array_equal(a, b)
    x = 2 * a - 3
    assert (x.sum(0) == 0).all() and np.array_equal(x.T @ x, len(a) * np.eye(n, dtype=np.int64))


class Additive:
    def __init__(self, rng, size, integer):
        self.w = rng.integers(-5, 6, size).astype(float) if integer else rng.normal(size=size)

    def __call__(self, g):
        return float(self.w @ g)


class Epistatic(Additive):
    def __init__(self, rng, size, integer):
        super().__init__(rng, size, integer)
        k = max(1, size // 2)
        self.pairs = rng.integers(0, size, (k, 2))
        self.v = rng.integers(-4, 5, k).astype(float) if integer else rng.normal(size=k) * 2

    def __call__(self, g):
        return super().__call__(g) + float((self.v * g[self.pairs[:, 0]] * g[self.pairs[:, 1]]).sum())


class Trap:
    def __init__(self, rng, size):
        self.perm = rng.permutation(size)

    def __call__(self, g):
        x = g[self.perm]
        return sum(
            0.0 if int(x[s : s + 4].sum()) == 4 else float(int(x[s : s + 4].sum()) + 1)
            for s in range(0, len(x) - len(x) % 4, 4)
        )


class Plateau:
    def __init__(self, rng, size):
        self.w, self.step = rng.integers(0, 3, size), int(rng.integers(2, 6))

    def __call__(self, g):
        return float(int(self.w @ g) // self.step)  # many ties


class HashObjective:
    def __init__(self, rng, size):
        self.salt = int(rng.integers(0, 2**31)).to_bytes(8, "little")

    def __call__(self, g):
        return (
            int.from_bytes(hashlib.sha256(self.salt + np.asarray(g, np.uint8).tobytes()).digest()[:6], "little") / 2**48
        )


def objective(rng, size):
    return [
        lambda: Additive(rng, size, True),
        lambda: Additive(rng, size, False),
        lambda: Epistatic(rng, size, True),
        lambda: Epistatic(rng, size, False),
        lambda: Trap(rng, size),
        lambda: Plateau(rng, size),
        lambda: HashObjective(rng, size),
        lambda: lambda g: 1.0,
    ][int(rng.integers(0, 8))]()


class StubProblem:
    """The Problem protocol with a fixed division."""

    version = "stub"

    def __init__(self, n_bits, segments):
        self.n_bits, self.segments = n_bits, segments

    def random_genome(self, rng):
        raise NotImplementedError

    def is_valid(self, g):
        return True

    def mutate(self, g, pm, rng):
        return g

    def divide(self, p1, p2, diff, max_segments, rng):
        return self.segments


def random_case(rng):
    size = int(rng.integers(3, 61))
    while True:
        if rng.random() < 0.33:
            p = pyiea.FixedCardinalityProblem(size, int(rng.integers(1, size)))
            p1, p2 = p.random_genome(rng), p.random_genome(rng)
        else:
            p1, p2 = freeze(rng.integers(0, 2, size)), freeze(rng.integers(0, 2, size))
        diff = np.flatnonzero(p1 != p2)
        if len(diff) >= 2:
            break
    n = int(rng.integers(2, min(len(diff), 30) + 1))
    if rng.random() < 0.5:
        segs = list(np.split(diff, np.sort(rng.choice(len(diff) - 1, n - 1, replace=False) + 1)))
    else:
        lab = np.concatenate([np.arange(n), rng.integers(0, n, len(diff) - n)])
        rng.shuffle(lab)
        segs = [diff[lab == j] for j in range(n)]
    return size, p1, p2, segs, objective(rng, size)


def fresh(f, p1, p2):
    ev = Evaluator(f, 1, "ctx")
    ev.evaluate_batch([p1, p2], phase="init")  # the parents are population members: already evaluated
    return ev


def elite(r):
    return [(g.tobytes(), y[0]) for g, y in r.children]


def test_igc_equals_the_reference_in_every_deterministic_part():
    rng = np.random.default_rng(12345)
    for _ in range(600):
        size, p1, p2, segs, f = random_case(rng)
        y1, y2 = (f(p1),), (f(p2),)
        ref = ref_igc(p1, p2, segs, f, f_p2=y2[0])
        prob = StubProblem(size, segs)
        r = igc(p1, y1, p2, y2, prob, fresh(f, p1, p2), np.random.default_rng(0), step10=False)
        assert r.status == "applied"
        rows = [g for g, _ in r.byproducts]
        assert len(rows) == len(ref["genomes"]) and all(
            np.array_equal(a, b) for a, b in zip(rows, ref["genomes"], strict=True)
        )
        responses = np.array([y[0] for _, y in r.byproducts])
        assert np.array_equal(responses, ref["y"])
        s1, s2 = main_effects(generate_oa(len(segs)), responses)
        assert np.array_equal(s1, ref["S"][:, 0]) and np.array_equal(s2, ref["S"][:, 1])
        assert r.trace["med_factor"] == ref["jmin"]
        (g1, v1), (g2, v2) = r.children
        assert (
            np.array_equal(g1, ref["c1"])
            and v1[0] == ref["y1"]
            and np.array_equal(g2, ref["c2"])
            and v2[0] == ref["y2"]
        )
        # Step 10 with the paper's candidate set (rows, C1, C2 and P2) ...
        paper = [(np.asarray(g, np.uint8).tobytes(), v) for g, v in ref["elite"]]
        assert elite(igc(p1, y1, p2, y2, prob, fresh(f, p1, p2), np.random.default_rng(0), step10=True)) == paper
        # ... and the pre-0.3 candidate set (without P2) when asked for
        old = ref_igc(p1, p2, segs, f, f_p2=np.inf)
        legacy = [(np.asarray(g, np.uint8).tobytes(), v) for g, v in old["elite"]]
        got = igc(p1, y1, p2, y2, prob, fresh(f, p1, p2), np.random.default_rng(0), step10=True, step10_p2=False)
        assert elite(got) == legacy


def test_additive_c1_is_the_brute_force_optimum_over_all_segment_combinations():
    rng = np.random.default_rng(7)
    done = 0
    while done < 300:
        size, p1, p2, segs, _ = random_case(rng)
        if len(segs) > 10:
            continue
        f = Additive(rng, size, bool(rng.random() < 0.5))
        r = igc(
            p1,
            (f(p1),),
            p2,
            (f(p2),),
            StubProblem(size, segs),
            fresh(f, p1, p2),
            np.random.default_rng(0),
            step10=False,
        )
        best = min(f(decode(p1, p2, segs, np.array(lv) + 1)) for lv in itertools.product((0, 1), repeat=len(segs)))
        assert abs(r.children[0][1][0] - best) <= 1e-9 * (1 + abs(best))
        done += 1
