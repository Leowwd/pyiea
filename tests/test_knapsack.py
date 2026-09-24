"""Multi-objective 0/1 knapsack of the IMOEA paper (Section VI-A, eq. 9) and the cover metric (eq. 8)."""

import itertools

import numpy as np
import pytest

import pyiea as ic
from pyiea.benchmarks import MultiKnapsack
from pyiea.problem import freeze

ALL_12 = [freeze(np.array(b, dtype=np.uint8)) for b in itertools.product((0, 1), repeat=12)]


def _zt_text(p, w, c):
    lines = [f"knapsack problem specification ({len(c)} knapsacks, {len(p[0])} items)"]
    for i in range(len(c)):
        lines += ["=", f"knapsack {i + 1}:", f" capacity: +{c[i]}"]
        for j in range(len(p[0])):
            lines += [f" item {j + 1}:", f"  weight: +{w[i][j]}", f"  profit: +{p[i][j]}"]
    return "\n".join(lines) + "\n"


def test_reads_zitzler_thiele_format(tmp_path):
    p, w, c = [[10, 20, 30], [40, 50, 60]], [[11, 12, 13], [14, 15, 16]], [18, 22]
    (tmp_path / "k").write_text(_zt_text(p, w, c))
    k = MultiKnapsack.from_zt_file(tmp_path / "k")
    assert k.p.tolist() == p and k.w.tolist() == w and k.c.tolist() == c
    (tmp_path / "bad").write_text(_zt_text(p, w, c).replace("(2 knapsacks, 3 items)", "(2 knapsacks, 4 items)"))
    with pytest.raises(ValueError, match="expected"):
        MultiKnapsack.from_zt_file(tmp_path / "bad")
    (tmp_path / "other").write_text("hello")
    with pytest.raises(ValueError, match="not a Zitzler-Thiele"):
        MultiKnapsack.from_zt_file(tmp_path / "other")


def test_generated_instance_follows_eq9():
    k = MultiKnapsack.random(300, 3, seed=5)
    assert k.p.shape == k.w.shape == (3, 300)
    assert k.p.min() >= 10 and k.p.max() <= 100 and k.w.min() >= 10 and k.w.max() <= 100
    assert k.c.tolist() == (k.w.sum(1) // 2).tolist()
    assert np.array_equal(MultiKnapsack.random(300, 3, seed=5).p, k.p)
    with pytest.raises(ValueError, match="capacities"):
        MultiKnapsack([[1, 2]], [[1, 2]], [1, 2])
    with pytest.raises(ValueError, match="positive"):
        MultiKnapsack([[1, 2]], [[0, 2]], [1])


def test_repair_removes_the_shortest_greedy_prefix():
    k = MultiKnapsack.random(12, 2, seed=3)
    q = (k.p / k.w).max(0)
    assert np.all(np.diff(q[k.removal_order]) >= 0)
    for s in ALL_12:
        x = k.repair(s)
        assert k.is_feasible(x) and np.all(x <= s)
        if k.is_feasible(s):
            assert np.array_equal(x, s)
            continue
        chosen = [j for j in k.removal_order if s[j]]
        removed = [j for j in chosen if not x[j]]
        assert removed == chosen[: len(removed)]  # items leave in increasing q order
        y = x.copy()
        y[removed[-1]] = 1
        assert not k.is_feasible(y)  # one removal fewer is still infeasible
    assert not s.flags.writeable  # the genome is never rewritten


def test_objective_is_negated_profit_of_repaired_solution():
    k = MultiKnapsack.random(12, 2, seed=3)
    s = ALL_12[-1]
    assert k(s) == tuple(-float(v) for v in k.p @ k.repair(s))


def test_repair_image_front_is_the_true_pareto_front():
    # every feasible x repairs to itself, so the non-dominated repaired values are the true front
    k = MultiKnapsack.random(12, 2, seed=3)
    Y = np.unique(np.array([k(s) for s in ALL_12]), axis=0)
    assert Y[ic.nondominated_mask(Y)].tolist() == k.exact_front().tolist()


def test_imoea_on_enumerable_knapsack():
    k = MultiKnapsack.random(12, 2, seed=3)
    exact = {tuple(y) for y in k.exact_front()}
    ev = ic.Evaluator(k, 2, "knap12", max_calls=1500)
    cfg = ic.IMOEAConfig(pop_size=20, elite_capacity=20, ps=0.2, pc=0.8, pm=0.01, max_segments=15)
    res = ic.IMOEA(ic.BinaryProblem(12), ev, cfg, seed=0).optimize()
    arch = {tuple(y) for _, y in res.archive}
    # the archive is exactly the non-dominated set of everything evaluated (independent recomputation)
    Y = np.array([e["objectives"] for e in ev.ledger if e["objectives"]])
    assert {tuple(y) for y in Y[ic.nondominated_mask(Y)]} == arch
    assert arch == exact  # 1500 calls on 4096 genomes find the whole true front
    for g, y in res.archive:
        assert k.is_feasible(k.repair(g)) and y == k(g)


def test_elite_unique_objectives_keeps_distinct_vectors():
    k = MultiKnapsack.random(40, 2, seed=1)
    cfg = ic.IMOEAConfig(pop_size=20, elite_capacity=20, max_segments=15, elite_unique_objectives=True)
    res = ic.IMOEA(ic.BinaryProblem(40), ic.Evaluator(k, 2, "k40", max_calls=3000), cfg, seed=0).optimize()
    Y = [y for _, y in res.front]
    assert len(Y) == len(set(Y))


def test_cover_metric_eq8_uses_weak_dominance():
    A = [(1, 3), (3, 1)]
    assert ic.coverage(A, [(1, 3)]) == 1.0  # an equal point is weakly dominated
    assert ic.coverage(A, [(2, 3), (0, 0), (3, 1), (4, 4)]) == 0.75
    assert ic.coverage([(0, 0)], A) == 1.0 and ic.coverage(A, [(0, 0)]) == 0.0  # not symmetric
    assert ic.coverage(A, []) == 0.0
