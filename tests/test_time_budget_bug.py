"""Bug-revealing tests (FAIL on pyiea 5f1dc62): a wall-time budget that expires inside an IGC."""

import time

import pyiea as ic
from pyiea.benchmarks import WeightedSum


def _run(monkeypatch, limit):
    clock = [0.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    f = WeightedSum(40, seed=0)

    def ticking(g):
        clock[0] += 1.0  # one second per objective call
        return f(g)

    ev = ic.Evaluator(ticking, 1, "c", max_seconds=limit)
    return ic.IEA(ic.BinaryProblem(40), ev, ic.IEAConfig(pop_size=10), seed=0).optimize()


def test_time_budget_expiring_during_oa_rows_does_not_crash(monkeypatch):
    # seed 0: the third IGC of generation 1 evaluates its OA rows over calls 112..127 and its children at 127..129
    res = _run(monkeypatch, 115.5)  # 5f1dc62: BudgetExhaustedError from the children batch
    assert res.stop_reason == "time_limit"


def test_time_budget_expiring_during_children_is_labelled_time_limit(monkeypatch):
    res = _run(monkeypatch, 128.5)  # 5f1dc62: 'budget_exhausted' with max_calls=None
    assert res.stop_reason == "time_limit"
