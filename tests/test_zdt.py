"""ZDT1-ZDT6 (IMOEA paper, Section VI-B, eq. 10): hand-computed values, fronts and encodings."""

import numpy as np
import pytest

import pyiea as ic
from pyiea import baselines
from pyiea.benchmarks import ZDT, ZDT_PROBLEMS
from pyiea.problem import freeze


@pytest.mark.parametrize(
    ("name", "x", "expected"),
    [
        ("zdt1", [0.25, 0.0, 0.0], (0.25, 0.5)),
        ("zdt1", [0.25, 1.0, 1.0], (0.25, 10 * (1 - np.sqrt(0.025)))),
        ("zdt2", [0.5, 0.0, 0.0], (0.5, 0.75)),
        ("zdt3", [0.5, 0.0, 0.0], (0.5, 1 - np.sqrt(0.5))),
        ("zdt4", [0.25, 0.0, 0.0], (0.25, 0.5)),
        ("zdt4", [0.25, 0.5, 0.0], (0.25, 1.25 * (1 - np.sqrt(0.2)))),
        ("zdt6", [0.0, 0.0, 0.0], (1.0, 0.0)),
        ("zdt6", [1 / 12, 0.0, 0.0], (1 - np.exp(-1 / 3), 1 - (1 - np.exp(-1 / 3)) ** 2)),
    ],
)
def test_real_zdt_by_hand(name, x, expected):
    assert ZDT(name, m=3)(np.array(x)) == pytest.approx(expected)


def test_zdt5_unitation_by_hand():
    z = ZDT("zdt5", m=3)
    g = np.zeros(40, dtype=np.uint8)
    g[:3] = 1  # u(x1) = 3 -> f1 = 4
    g[30:35] = 1  # u(x2) = 5 -> v = 1; u(x3) = 0 -> v = 2 (the deceptive attractor)
    assert z(g) == (4.0, 3 / 4)
    assert z.problem().n_bits == 40 and z.problem().widths.tolist() == [30, 5, 5]
    with pytest.raises(ValueError, match="no Gray"):
        z.problem(gray=True)
    with pytest.raises(ValueError, match="defined on bits"):
        z.encoder()


def test_paper_sizes_m63():
    assert ZDT("zdt1", m=63).problem().n_bits == 63 * 30
    assert ZDT("zdt5", m=63).problem().n_bits == 30 + 62 * 5
    assert ZDT("zdt4", m=63).bounds[1] == (-5.0, 5.0) and ZDT("zdt4", m=63).bounds[0] == (0.0, 1.0)
    with pytest.raises(ValueError, match="unknown"):
        ZDT("zdt7")


@pytest.mark.parametrize("gray", [False, True])
@pytest.mark.parametrize("name", ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6"])
def test_encoded_objective_decodes_then_evaluates(name, gray):
    z = ZDT(name, m=4, bits=8)
    rng = np.random.default_rng(0)
    P, f, enc = z.problem(gray), z.objective(gray), z.encoder(gray)
    for _ in range(5):
        g = P.random_genome(rng)
        assert f(g) == z(enc.decode(g))
    assert P.version.endswith("-gray") == gray


def test_fronts_are_analytic_optima():
    # ZDT1/2/4/6: g >= 1 and f2 = g h(f1/g) grows with g, so the front bounds every point from below
    rng = np.random.default_rng(1)
    for name in ("zdt1", "zdt2", "zdt4", "zdt6"):
        z = ZDT(name, m=5)
        F = z.pareto_front(2000)
        for _ in range(200):
            x = np.array([lo + (hi - lo) * rng.random() for lo, hi in z.bounds])
            f1, f2 = z(x)
            ref = np.interp(f1, F[:, 0], F[:, 1]) if f1 >= F[0, 0] else np.inf
            assert f2 >= ref - 1e-3 or f1 < F[0, 0], name
    assert ZDT("zdt6").pareto_front()[0, 0] == pytest.approx(0.2807753191, abs=1e-6)
    z5 = ZDT("zdt5", m=63).pareto_front()
    assert z5[0].tolist() == [1.0, 62.0] and z5[-1].tolist() == [31.0, 2.0]
    z3 = ZDT("zdt3").pareto_front()
    # two objectives: mutually non-dominated <=> f1 strictly increasing and f2 strictly decreasing
    assert np.all(np.diff(z3[:, 0]) > 0) and np.all(np.diff(z3[:, 1]) < 0) and z3[0, 0] == 0.0
    sample = z3[:: len(z3) // 300]
    assert not ic.dominance_matrix(sample).any()  # the generic check agrees on a subsample
    assert len(np.flatnonzero(np.diff(z3[:, 0]) > 0.05)) == 4  # five disconnected pieces


def test_optimal_genome_lies_on_front():
    z = ZDT("zdt1", m=5, bits=10)
    g = np.zeros(50, dtype=np.uint8)
    g[:10] = 1  # x1 = 1, x2..x5 = 0
    assert z.objective()(freeze(g)) == (1.0, 0.0)
    g[:10] = 0
    assert z.objective()(freeze(g)) == (0.0, 1.0)


def test_imoea_approaches_zdt1_front():
    z = ZDT("zdt1", m=8, bits=10)
    ev = ic.Evaluator(z.objective(True), 2, "zdt1-small", max_calls=4000)
    res = ic.IMOEA(z.problem(True), ev, ic.IMOEAConfig(max_segments=8), seed=0).optimize()
    ev_r = ic.Evaluator(z.objective(True), 2, "zdt1-small", max_calls=4000)
    rnd = baselines.random_search(z.problem(True), ev_r, seed=0)
    front = z.pareto_front()
    assert ic.igd([y for _, y in res.archive], front) < ic.igd([y for _, y in rnd.archive], front)


def test_all_problems_listed():
    assert ZDT_PROBLEMS == ("zdt1", "zdt2", "zdt3", "zdt4", "zdt5", "zdt6")
