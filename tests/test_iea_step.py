"""IEA.step(): optimize() and a caller-driven loop are identical, and the caller may change the objective context."""

import numpy as np
import pytest

import pyiea as ic
from pyiea.benchmarks import QuadraticBinary, WeightedSum


def _drive(iea: ic.IEA) -> str:
    """The manual loop equivalent to optimize() (no checkpoints): evaluate, stop-test, step."""
    cfg = iea.cfg
    iea.pop = [iea.problem.random_genome(iea.rng) for _ in range(cfg.pop_size)]
    iea.pop_y = [None] * cfg.pop_size
    try:
        while True:
            complete = iea.evaluate_population()
            if iea.gen == 0 and iea.best is None:
                iea.evaluator.raise_if_all_failed(len(iea.pop))
            iea.pop_best.append(min(map(iea._key, range(cfg.pop_size))))
            iea._calls_log.append(iea.evaluator.counters["objective_calls"])
            reason = iea._stop_reason(complete)
            if reason:
                return reason
            reason = iea.step()
            if reason:
                return reason
    finally:
        iea.evaluator.close()


def _no_seconds(history):
    return [{k: v for k, v in h.items() if k != "seconds"} for h in history]


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("igc", [True, False])
@pytest.mark.parametrize("max_segments", [None, 7])
@pytest.mark.parametrize("step10", [True, False])
@pytest.mark.parametrize("max_calls", [None, 120])
def test_step_driver_equals_optimize(seed, igc, max_segments, step10, max_calls):
    P, f = ic.BinaryProblem(30), QuadraticBinary(30, seed=seed)
    cfg = ic.IEAConfig(pop_size=10, max_generations=6, max_segments=max_segments, step10=step10, igc=igc)

    def make():
        return ic.IEA(P, ic.Evaluator(f, 1, "t", max_calls=max_calls), cfg, seed=seed)

    a = make()
    res = a.optimize()
    b = make()
    reason = _drive(b)
    assert reason == res.stop_reason
    assert np.array_equal(b.best[0], a.best[0]) and b.best[1] == a.best[1]
    assert b.gen == a.gen == res.generations
    assert _no_seconds(b.history) == _no_seconds(a.history)
    assert b.igc_log == a.igc_log and b.pop_best == a.pop_best
    assert [r["genome"] for r in b.evaluator.ledger] == [r["genome"] for r in a.evaluator.ledger]
    assert [r["phase"] for r in b.evaluator.ledger] == [r["phase"] for r in a.evaluator.ledger]
    assert b.evaluator.counters == a.evaluator.counters
    assert all(np.array_equal(x, y) for x, y in zip(a.pop, b.pop, strict=True))


class _Contexts(ic.Evaluator):
    """An Evaluator with one cache per objective context, switched by set_context()."""

    def __init__(self, objectives, **kw):
        super().__init__(objectives[0], 1, "ctx", **kw)
        self.objectives, self.caches = objectives, {}
        self.set_context(0)

    def set_context(self, i):
        self.objective = self.objectives[i]
        self.cache = self.caches.setdefault(i, {})


def test_caller_can_change_the_objective_context_between_steps():
    P = ic.BinaryProblem(20)
    f0, f1 = WeightedSum(20, seed=1), WeightedSum(20, seed=2)
    ev = _Contexts([f0, f1])
    iea = ic.IEA(P, ev, ic.IEAConfig(pop_size=8, pc=0.5, max_segments=7), seed=0)
    iea.pop = [P.random_genome(iea.rng) for _ in range(8)]
    iea.pop_y = [None] * 8
    iea.evaluate_population()
    assert iea.step() is None
    iea.evaluate_population()  # the mutated individuals, still context 0
    for ctx, f in ((1, f1), (0, f0), (1, f1)):
        calls = ev.counters["objective_calls"]
        ev.set_context(ctx)
        iea.pop_y = [None] * len(iea.pop)  # old fitness is not valid under the new context
        assert iea.evaluate_population()
        assert all(y == (f(g),) for g, y in zip(iea.pop, iea.pop_y, strict=True))
        assert ev.counters["objective_calls"] - calls <= len(iea.pop)
        assert iea.step() is None
        iea.evaluate_population()
    assert ev.counters["objective_calls"] == len(ev.ledger)
    assert iea.gen == 4


def test_step_returns_the_budget_reason_when_an_igc_does_not_fit():
    P, f = ic.BinaryProblem(30), QuadraticBinary(30, seed=0)
    iea = ic.IEA(P, ic.Evaluator(f, 1, "t", max_calls=40), ic.IEAConfig(pop_size=10, max_segments=7), seed=0)
    assert _drive(iea) == "budget_exhausted"
    assert iea.evaluator.counters["objective_calls"] <= 40
    assert iea.igc_log[-1]["status"] == "budget"
