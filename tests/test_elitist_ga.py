"""Section V-B baselines: elitist GAs with one-point, two-point and uniform crossover (OEGA, TEGA, UEGA)."""

import numpy as np
import pytest

import pyiea
from pyiea import baselines
from pyiea.benchmarks import PaperBenchmark, WeightedSum
from pyiea.problem import freeze

ZEROS, ONES = freeze(np.zeros(40, dtype=np.uint8)), freeze(np.ones(40, dtype=np.uint8))


@pytest.mark.parametrize(("name", "switches"), [("one_point", 1), ("two_point", 2), ("uniform", None)])
def test_crossover_children_are_complementary(name, switches):
    rng = np.random.default_rng(0)
    for _ in range(50):
        c1, c2 = baselines.CROSSOVERS[name](ZEROS, ONES, rng)
        assert np.array_equal(c1 ^ c2, ONES)  # every gene comes from exactly one parent in each child
        assert not c1.flags.writeable and not c2.flags.writeable
        if switches is not None:
            assert int(np.count_nonzero(np.diff(c1.astype(int)))) == switches  # the cut points


def test_same_initial_population_as_iea():
    bench = PaperBenchmark("f1", 5)
    enc = bench.encoder()
    evs = [pyiea.Evaluator(enc.wrap(bench), 1, "f1", max_calls=10) for _ in range(2)]
    pyiea.IEA(enc.problem(), evs[0], pyiea.IEAConfig(pop_size=10), seed=3).optimize()
    baselines.elitist_ga(enc.problem(), evs[1], seed=3)
    assert [e["genome"] for e in evs[0].ledger] == [e["genome"] for e in evs[1].ledger]


@pytest.mark.parametrize("crossover", ["one_point", "two_point", "uniform"])
def test_elitist_ga_keeps_the_best_and_the_budget(crossover):
    f = WeightedSum(30, seed=2)
    ev = pyiea.Evaluator(f, 1, "w", max_calls=3000)
    res = baselines.elitist_ga(pyiea.BinaryProblem(30), ev, seed=1, crossover=crossover)
    best = [h["best"] for h in res.history]
    assert best == sorted(best, reverse=True)  # the elite never gets worse
    assert res.accounting["objective_calls"] <= 3000 and res.stop_reason == "budget_exhausted"
    assert res.best == pytest.approx(min(e["objectives"][0] for e in ev.ledger))
    assert res.best <= f.optimum() + 1.0  # additive and easy: close to the optimum


def test_elitist_ga_stalls_on_an_exhausted_space_and_rejects_unknown_crossover():
    res = baselines.elitist_ga(pyiea.BinaryProblem(3), pyiea.Evaluator(WeightedSum(3), 1, "w"), seed=0)
    assert res.stop_reason == "stalled" and res.accounting["objective_calls"] <= 8
    with pytest.raises(ValueError, match="unknown crossover"):
        baselines.elitist_ga(pyiea.BinaryProblem(3), pyiea.Evaluator(WeightedSum(3), 1, "w"), crossover="blx")
