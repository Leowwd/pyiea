"""Public API: optimize(), configs, encoders, operators, baselines, packaging."""

import itertools

import numpy as np
import pytest

import pyiea
from pyiea import baselines
from pyiea.benchmarks import FixedCardWeighted, KeepDropBiObjective, PaperBenchmark, WeightedSum


def test_public_api_is_exported():
    for name in pyiea.__all__:
        assert hasattr(pyiea, name), name
    assert pyiea.__version__ != "0.0.0"


def test_optimize_single_objective_finds_additive_optimum():
    f = WeightedSum(30, seed=1)
    res = pyiea.optimize(f, n_bits=30, max_calls=3000, seed=1)
    assert isinstance(res, pyiea.IEAResult)
    assert res.best == pytest.approx(f.optimum())
    assert res.best == pytest.approx(f(res.best_genome))


def test_optimize_multi_objective_returns_front():
    f = KeepDropBiObjective(10, seed=0)
    res = pyiea.optimize(f, n_bits=10, mode="multi_objective", n_objectives=2, max_calls=600)
    assert isinstance(res, pyiea.IMOEAResult) and res.front
    assert pyiea.coverage(f.exact_front(), [y for _, y in res.archive]) == 1.0


def test_optimize_accepts_numpy_array_objectives():
    res = pyiea.optimize(
        lambda g: np.array([float(g.sum()), -float(g.sum())]), n_bits=6, mode="multi_objective", max_calls=100
    )
    assert res.accounting["failed_evaluations"] == 0


def test_optimize_with_custom_problem():
    f = FixedCardWeighted(30, 10, seed=2)
    res = pyiea.optimize(f, problem=pyiea.FixedCardinalityProblem(30, 10), max_calls=3000)
    assert int(res.best_genome.sum()) == 10
    assert res.best == pytest.approx(f.optimum())


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({}, ValueError),  # neither n_bits nor problem
        ({"n_bits": 4, "max_calls": None}, ValueError),  # no stopping condition
        ({"n_bits": 4, "mode": "pareto"}, ValueError),
        ({"n_bits": 4, "config": pyiea.IMOEAConfig()}, TypeError),
        ({"n_bits": 4, "mode": "multi_objective", "config": pyiea.IEAConfig()}, TypeError),
    ],
)
def test_optimize_rejects_bad_arguments(kwargs, exc):
    with pytest.raises(exc):
        pyiea.optimize(lambda g: 0.0, **kwargs)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (pyiea.IEAConfig, {"pop_size": 1}),
        (pyiea.IEAConfig, {"ps": 1.5}),
        (pyiea.IEAConfig, {"pc": 0.01}),
        (pyiea.IEAConfig, {"max_segments": 1}),
        (pyiea.IMOEAConfig, {"elite_capacity": 0}),
        (pyiea.IMOEAConfig, {"pm": -0.1}),
    ],
)
def test_config_validation(cls, kwargs):
    with pytest.raises(ValueError, match="must"):
        cls(**kwargs)


def test_problem_and_evaluator_validation():
    with pytest.raises(ValueError, match="n_bits"):
        pyiea.BinaryProblem(0)
    with pytest.raises(ValueError, match="0 < k < n_bits"):
        pyiea.FixedCardinalityProblem(5, 5)
    with pytest.raises(ValueError, match="n_objectives"):
        pyiea.Evaluator(lambda g: 0.0, 0, "c")
    with pytest.raises(ValueError, match="single-objective"):
        pyiea.IEA(pyiea.BinaryProblem(4), pyiea.Evaluator(lambda g: (0.0, 0.0), 2, "c"))


def test_unpicklable_objective_with_workers_gives_clear_error():
    ev = pyiea.Evaluator(lambda g: 0.0, 1, "c", workers=2)
    gs = [pyiea.freeze([0, 1]), pyiea.freeze([1, 0])]
    with pytest.raises(TypeError, match="picklable"):
        ev.evaluate_batch(gs)


def test_real_encoder_decodes_bounds_and_wraps():
    enc = pyiea.RealEncoder([(-1.0, 1.0), (3.0, 13.0)], bits=4)
    assert enc.n_bits == 8
    assert enc.decode(np.zeros(8, np.uint8)).tolist() == [-1.0, 3.0]
    assert enc.decode(np.ones(8, np.uint8)).tolist() == [1.0, 13.0]
    obj = enc.wrap(lambda x: float(x.sum()))
    assert obj(np.ones(8, np.uint8)) == 14.0
    with pytest.raises(ValueError, match="low < high"):
        pyiea.RealEncoder([(1.0, 0.0)])


def test_gray_decoding_makes_neighbours_one_bit_apart():
    enc = pyiea.RealEncoder([(0.0, 7.0)], bits=3, gray=True)
    gray = [np.array([(k ^ k >> 1) >> s & 1 for s in (2, 1, 0)], np.uint8) for k in range(8)]
    assert [enc.decode(g)[0] for g in gray] == list(range(8))
    assert all(int((a != b).sum()) == 1 for a, b in itertools.pairwise(gray))


def test_paper_f1_reaches_near_optimum():
    # f1 of Table IV with D = 10 (10 bits per parameter, as in the paper); optimum 1.21598 * D
    b = PaperBenchmark("f1", 10)
    enc = b.encoder()
    res = pyiea.optimize(enc.wrap(b), problem=enc.problem(), max_calls=10_000, seed=0)
    assert b.paper_value(res.best) > 0.99 * b.optimum


def test_truncation_refill_copies_the_best():
    # after Step 3 the worst ps*Npop slots hold copies of the best ps*Npop (engineering choice)
    P = pyiea.BinaryProblem(12)
    ev = pyiea.Evaluator(WeightedSum(12, seed=3), 1, "c")
    opt = pyiea.IEA(P, ev, pyiea.IEAConfig(pop_size=10, ps=0.2, pc=0.2, pm=0.0, max_generations=1), seed=0)
    seen = []
    real = pyiea.iea.igc

    def spy(p1, y1, p2, y2, *a, **k):
        seen.append(list(opt.pop_y))
        return real(p1, y1, p2, y2, *a, **k)

    pyiea.iea.igc = spy
    try:
        opt.optimize()
    finally:
        pyiea.iea.igc = real
    ys = [y[0] for y in seen[0]]
    assert ys[:8] == sorted(ys[:8]) and ys[8:] == ys[:2]


def test_swap_mutation_keeps_cardinality_with_per_bit_rate():
    P, rng = pyiea.FixedCardinalityProblem(40, 10), np.random.default_rng(0)
    g = P.random_genome(rng)
    swaps = []
    for _ in range(500):
        m = P.mutate(g, 0.2, rng)
        assert P.is_valid(m)
        assert (m is g) == bool((m == g).all())
        swaps.append(int((m != g).sum()) // 2)
    assert np.mean(swaps) == pytest.approx(0.2 * 10, rel=0.15)  # pm per 1-bit


def test_baselines_return_standard_results():
    P, f = pyiea.BinaryProblem(20), WeightedSum(20)
    for fn in (baselines.random_search, baselines.vanilla_ga):
        r = fn(P, pyiea.Evaluator(f, 1, "c", max_calls=200), seed=0)
        assert isinstance(r, pyiea.IEAResult) and r.best is not None and r.accounting["objective_calls"] <= 200
    g = KeepDropBiObjective(20)
    for fn in (baselines.random_search, baselines.nsga2):
        r = fn(P, pyiea.Evaluator(g, 2, "c", max_calls=200), seed=0)
        assert isinstance(r, pyiea.IMOEAResult) and r.front


def test_evaluator_context_manager_closes_pool():
    with pyiea.Evaluator(WeightedSum(4), 1, "c", workers=2) as ev:
        ev.evaluate_batch([pyiea.freeze([0, 0, 0, 1]), pyiea.freeze([1, 0, 0, 0])])
        assert ev._pool is not None
    assert ev._pool is None
