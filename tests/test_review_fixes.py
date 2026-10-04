"""Regression tests for the findings of the independent review of 2026-10-05 (fixed in this version)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import pyiea as ic
from pyiea.benchmarks import QuadraticBinary, WeightedSum
from pyiea.iea import _count
from pyiea.problem import freeze


class Seg3:
    """Three one-bit segments of the differing positions of 000 and 111."""

    n_bits, version = 3, "seg3"

    def divide(self, p1, p2, diff, max_segments, rng):
        return [np.array([0]), np.array([1]), np.array([2])]


def _trap(g):
    return -100.0 if g.tolist() == [1, 1, 1] else float(g.sum())


def _step10(**kw):
    p1, p2 = freeze([0, 0, 0]), freeze([1, 1, 1])
    ev = ic.Evaluator(_trap, 1, "t")
    ev.evaluate_batch([p1, p2])
    return ic.igc(p1, (_trap(p1),), p2, (_trap(p2),), Seg3(), ev, np.random.default_rng(0), **kw)


def test_step10_keeps_the_second_parent_when_it_is_the_best_candidate():
    """Paper p. 527, Step 10: the best two of the n combinations, C1, C2 *and P2* (P1 is combination 1)."""
    got = [(g.tolist(), y[0]) for g, y in _step10(step10=True).children]
    assert got == [([1, 1, 1], -100.0), ([0, 0, 0], 0.0)]
    legacy = [(g.tolist(), y[0]) for g, y in _step10(step10=True, step10_p2=False).children]
    assert [1, 1, 1] not in [g for g, _ in legacy]  # the pre-0.3 behavior this flag reproduces


def test_the_papers_table_ii_end_to_end_through_igc():
    def f(g):  # the paper's Table II response, negated because pyiea minimizes
        return -float(100 * (1 + g[0]) - 10 * (3 + g[1]) - (5 + g[2]))

    p1, p2 = freeze([0, 0, 0]), freeze([1, 1, 1])
    ev = ic.Evaluator(f, 1, "t")
    y1, y2 = (r.objectives for r in ev.evaluate_batch([p1, p2]))  # type: ignore[union-attr]
    r = ic.igc(p1, y1, p2, y2, ic.BinaryProblem(3), ev, np.random.default_rng(0), step10=False)
    assert [-y[0] for _, y in r.byproducts] == [65, 54, 164, 155]
    assert (r.children[0][0] + [1, 3, 5]).tolist() == [2, 3, 5] and (r.children[1][0] + [1, 3, 5]).tolist() == [2, 3, 6]
    assert r.trace["med_factor"] == 2  # x3 has the smallest main-effect difference


class ClockEvaluator(ic.Evaluator):
    """A wall-time limit that expires after ``k`` objective calls (deterministic, no sleeping)."""

    k = 0

    def time_up(self) -> bool:
        return self.counters["objective_calls"] >= self.k


@pytest.mark.parametrize("k", [12, 20, 31, 47, 60, 95, 140, 211])
def test_a_time_limit_inside_an_igc_stops_the_run_instead_of_raising(k):
    ev = ClockEvaluator(QuadraticBinary(64, seed=0), 1, "t", max_seconds=1e9)
    ev.k = k
    res = ic.IEA(ic.BinaryProblem(64), ev, ic.IEAConfig(pop_size=10), seed=0).optimize()
    assert res.stop_reason == "time_limit" and res.best is not None
    assert ev.counters["objective_calls"] >= k


def test_a_time_limit_inside_an_igc_stops_imoea_too():
    def f(g):
        return float(g.sum()), float(len(g) - g.sum() + (g[:5].sum() % 2))

    for k in (25, 40, 77):
        ev = ClockEvaluator(f, 2, "t", max_seconds=1e9)
        ev.k = k
        res = ic.IMOEA(ic.BinaryProblem(32), ev, ic.IMOEAConfig(pop_size=10), seed=0).optimize()
        assert res.stop_reason == "time_limit"


def test_truncation_never_refills_with_individuals_it_dropped_when_ps_is_above_half():
    # one-hot individuals whose objective is their index: the best two have objectives 0 and 1
    problem = ic.BinaryProblem(10)
    pop = [freeze(np.eye(10, dtype=np.uint8)[i]) for i in range(10)]
    ev = ic.Evaluator(lambda g: float(np.argmax(g)), 1, "t")
    it = ic.IEA(problem, ev, ic.IEAConfig(pop_size=10, ps=0.8, pc=0.8, pm=0.0, igc=False), seed=0)
    it.pop, it.pop_y = pop, [None] * 10
    it.evaluate_population()
    assert it.step() is None
    assert {y[0] for y in it.pop_y if y is not None} <= {0.0, 1.0}


def test_truncation_for_ps_up_to_half_is_unchanged():
    problem = ic.BinaryProblem(10)
    ev = ic.Evaluator(lambda g: float(np.argmax(g)), 1, "t")
    it = ic.IEA(problem, ev, ic.IEAConfig(pop_size=10, ps=0.3, pc=0.8, pm=0.0, igc=False), seed=0)
    it.pop, it.pop_y = [freeze(np.eye(10, dtype=np.uint8)[i]) for i in range(10)], [None] * 10
    it.evaluate_population()
    it.step()
    assert sorted(y[0] for y in it.pop_y if y is not None) == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_counts_are_not_fooled_by_float_products():
    assert _count(0.29, 100) == 29 and _count(0.57, 100) == 57 and _count(0.2, 30) == 6 and _count(0.8, 10) == 8


def test_calling_optimize_twice_adds_no_duplicate_generation_entry():
    ev = ic.Evaluator(QuadraticBinary(32, seed=0), 1, "t", max_calls=300)
    it = ic.IEA(ic.BinaryProblem(32), ev, ic.IEAConfig(pop_size=10), seed=0)
    first = it.optimize()
    n = len(first.pop_best)
    second = it.optimize()
    assert len(second.pop_best) == n


def test_without_step10_the_best_so_far_still_never_regresses():
    ev = ic.Evaluator(QuadraticBinary(48, seed=3), 1, "t", max_calls=1200)
    res = ic.IEA(ic.BinaryProblem(48), ev, ic.IEAConfig(pop_size=12, step10=False), seed=3).optimize()
    best = [h["best"] for h in res.history]
    assert best == sorted(best, reverse=True)


# trajectories of pyiea 0.2.0 (before the Step 10 / truncation fixes), pinned with step10_p2=False
GOLDEN = [
    ((0, "bin", "quad", 600, {"pop_size": 10}), ("fd76292fdfd2fdc1", 23)),
    ((1, "card", "sum", 600, {"pop_size": 12, "max_segments": 7}), ("497be09d134609f7", 73)),
    ((2, "bin", "quad", 800, {"pop_size": 10, "ps": 0.4}), ("48be8d68fedd0989", 48)),
]


@pytest.mark.parametrize(("case", "golden"), GOLDEN)
def test_legacy_mode_reproduces_the_pre_fix_trajectories_bit_for_bit(case, golden):
    seed, prob, fn, calls, kw = case
    problem = ic.BinaryProblem(40) if prob == "bin" else ic.FixedCardinalityProblem(40, 13)
    f = QuadraticBinary(40, seed=seed) if fn == "quad" else WeightedSum(40, seed=seed)
    ev = ic.Evaluator(f, 1, "t", is_valid=problem.is_valid, max_calls=calls)
    res = ic.IEA(problem, ev, ic.IEAConfig(**kw, step10_p2=False), seed=seed).optimize()
    digest = hashlib.sha256(repr([(e["genome"], e["phase"], e["objectives"]) for e in ev.ledger]).encode()).hexdigest()[
        :16
    ]
    assert (digest, res.generations) == golden
