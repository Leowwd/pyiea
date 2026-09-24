"""Multi-objective correctness gates: dominance, GPSIFF, elite sets, IMOEA, archive."""

import numpy as np
import pytest

import pyiea as ic
from pyiea.benchmarks import KeepDropBiObjective
from pyiea.pareto import ParetoSet
from pyiea.problem import freeze


def test_gpsiff_hand_example():
    S = [(1, 1), (2, 2), (0, 3), (3, 0)]
    assert ic.gpsiff(S).tolist() == [5, 3, 4, 4]


def test_gpsiff_paper_fig1_formula():
    # individual dominating p=3 and dominated by q=2 among 12 participants scores 13
    A = (5, 5)
    pts = [A, (6, 6), (7, 6), (6, 7), (4, 4), (3, 3)] + [(0, 20 + i) for i in range(6)]
    assert ic.gpsiff(pts)[0] == 3 - 2 + 12


def test_equal_vectors_do_not_dominate():
    D = ic.dominance_matrix([(1, 2), (1, 2), (2, 1)])
    assert not D.any()


def test_new_point_changes_gpsiff_not_cache():
    ev = ic.Evaluator(lambda g: (float(g[0]), float(g[1])), 2, "t")
    gs = [freeze(np.array(v)) for v in ([0, 1], [1, 0])]
    ys = [r.objectives for r in ev.evaluate_batch(gs)]
    before = ic.gpsiff(ys).tolist()
    ys2 = ys + [r.objectives for r in ev.evaluate_batch([freeze(np.array([0, 0]))])]
    after = ic.gpsiff(ys2).tolist()
    assert before == [2, 2] and after == [2, 2, 5]
    assert [r.objectives for r in ev.evaluate_batch(gs)] == ys  # objective cache unchanged


def test_elite_set_dedup_dominance_and_seeded_truncation():
    E = ParetoSet()
    g = [freeze(np.array([i], dtype=np.uint8)) for i in range(6)]
    E.update([(g[0], (1, 5)), (g[1], (2, 4)), (g[2], (3, 3)), (g[3], (3, 5)), (g[0], (1, 5))])
    assert len(E) == 3  # (3,5) dominated, duplicate genome kept once
    E.update([(g[4], (0, 0))])
    assert [y for _, y in E.items()] == [(0, 0)]
    for cap_seed in (0, 0):
        F = ParetoSet()
        F.update([(g[i], (i, 5 - i)) for i in range(6)])
        F.truncate(3, np.random.default_rng(cap_seed))
        assert len(F) == 3
    a, b = ParetoSet(), ParetoSet()
    for s in (a, b):
        s.update([(g[i], (i, 5 - i)) for i in range(6)])
        s.truncate(3, np.random.default_rng(7))
    assert [y for _, y in a.items()] == [y for _, y in b.items()]


@pytest.mark.parametrize("k", [2, 4])
def test_incremental_update_equals_filtering_the_union(k):
    # brute force: add every genome once (first wins), keep rows no other row dominates, in insertion order
    rng = np.random.default_rng(k)
    g = [freeze(np.array([i // 256, i % 256], dtype=np.uint8)) for i in range(600)]
    E, seen = ParetoSet(), {}
    for step in range(40):
        idx = rng.integers(0, 600, rng.integers(0, 25))
        batch = [(g[i], tuple(rng.integers(0, 6, k).tolist())) for i in idx]
        E.update(batch)
        for gi, y in batch:
            seen.setdefault(gi.tobytes(), (gi, tuple(float(v) for v in y)))
        vals = list(seen.values())
        nd = ic.nondominated_mask([y for _, y in vals]) if vals else []
        seen = {gi.tobytes(): (gi, y) for (gi, y), m in zip(vals, nd) if m}
        assert [(gi.tobytes(), y) for gi, y in E.items()] == [(gi.tobytes(), y) for gi, y in seen.values()], step
        assert E.objectives().tolist() == [list(y) for _, y in seen.values()]
        if step % 10 == 9:  # cached objectives stay in sync after the other mutations
            E.truncate(5, np.random.default_rng(step))
            seen = {gi.tobytes(): (gi, y) for gi, y in E.items()}


def test_drop_duplicate_objectives_keeps_first_genome():
    g = [freeze(np.array([i], dtype=np.uint8)) for i in range(4)]
    E = ParetoSet([(g[0], (1, 2)), (g[1], (1, 2)), (g[2], (2, 1)), (g[3], (2, 1))])
    E.drop_duplicate_objectives()
    assert [(int(x[0]), y) for x, y in E.items()] == [(0, (1, 2)), (2, (2, 1))]
    assert E.objectives().tolist() == [[1, 2], [2, 1]]


@pytest.mark.parametrize("n", [1, 2, 7, 300])
def test_two_objective_fast_path_matches_pairwise_dominance(n):
    rng = np.random.default_rng(n)
    F = rng.integers(0, 6, (n, 2)).astype(float)  # many ties and duplicate rows
    F = np.vstack([F, F[: n // 3]])
    assert ic.nondominated_mask(F).tolist() == (~ic.dominance_matrix(F).any(0)).tolist()


def test_hypervolume_and_igd():
    assert ic.hypervolume_2d([(1, 1)], (2, 2)) == 1
    assert ic.hypervolume_2d([(0, 1), (1, 0)], (2, 2)) == 3
    assert ic.igd([(0, 0)], [(0, 0), (3, 4)]) == 2.5


def _run(seed, n_bits=10, calls=600, **cfg):
    f = KeepDropBiObjective(n_bits, seed=seed)
    ev = ic.Evaluator(f, 2, "t", max_calls=calls)
    res = ic.IMOEA(ic.BinaryProblem(n_bits), ev, ic.IMOEAConfig(**cfg), seed=seed).optimize()
    return f, ev, res


def test_imoea_archive_matches_independent_oracle():
    f, ev, res = _run(1)
    exact = f.exact_front()
    arch = np.array([y for _, y in res.archive])
    # every archive point is really non-dominated among all evaluated points, and
    # every archive point that is Pareto-optimal must appear in the oracle front
    all_y = np.array([e["objectives"] for e in ev.ledger if e["objectives"]])
    assert not ic.dominance_matrix(np.vstack([arch, all_y]))[:, : len(arch)].any()
    assert ic.coverage(exact, arch) == 1.0
    for g, y in res.archive:
        assert y == pytest.approx(f(g))
    assert ic.hypervolume_2d(arch, (10, 1.1)) >= 0.9 * ic.hypervolume_2d(exact, (10, 1.1))


def test_byproducts_reach_temporary_and_final_sets():
    _f, ev, res = _run(2, calls=400)
    oa_points = {tuple(e["objectives"]) for e in ev.ledger if e["phase"] == "oa"}
    arch = {tuple(y) for _, y in res.archive}
    assert arch & oa_points  # some non-dominated solutions came from OA by-products


def test_working_elite_is_bounded_and_separate_from_archive():
    _f, _ev, res = _run(3, n_bits=14, calls=1500, elite_capacity=8)
    assert len(res.front) <= 8
    assert len(res.archive) >= len(res.front)


def test_archive_hypervolume_is_monotone():
    f = KeepDropBiObjective(12, seed=4)
    ev = ic.Evaluator(f, 2, "t", max_calls=800)
    opt = ic.IMOEA(ic.BinaryProblem(12), ev, ic.IMOEAConfig(), seed=4)
    hv, orig = [], opt._add_to_archive

    def spy(items):
        orig(items)
        hv.append(ic.hypervolume_2d(opt.archive.objectives(), (10, 1.1)))

    opt._add_to_archive = spy
    opt.optimize()
    assert hv == sorted(hv)


def test_imoea_budget_and_checkpoint_resume(tmp_path):
    f = KeepDropBiObjective(12, seed=5)
    cfg = ic.IMOEAConfig(max_generations=6, checkpoint_every=3)
    full = ic.IMOEA(ic.BinaryProblem(12), ic.Evaluator(f, 2, "t"), cfg, seed=5).optimize()

    cut = ic.IMOEAConfig(**{**cfg.__dict__, "max_generations": 3})
    ic.IMOEA(ic.BinaryProblem(12), ic.Evaluator(f, 2, "t"), cut, seed=5).optimize(tmp_path / "c.pkl")
    res = ic.IMOEA(ic.BinaryProblem(12), ic.Evaluator(f, 2, "t"), cfg, seed=5).optimize(tmp_path / "c.pkl", resume=True)
    assert [y for _, y in res.front] == [y for _, y in full.front]
    assert [y for _, y in res.archive] == [y for _, y in full.archive]
    assert res.accounting["objective_calls"] == full.accounting["objective_calls"]

    ev = ic.Evaluator(f, 2, "t", max_calls=137)
    r = ic.IMOEA(ic.BinaryProblem(12), ev, ic.IMOEAConfig(), seed=5).optimize()
    assert ev.counters["objective_calls"] <= 137 and r.stop_reason == "budget_exhausted"
    # every completed evaluation (OA by-products included) reached the archive before the stop
    Y = np.array([e["objectives"] for e in ev.ledger if e["objectives"]])
    assert {tuple(y) for y in Y[ic.nondominated_mask(Y)]} == {tuple(y) for _, y in r.archive}
