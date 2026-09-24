"""Correctness gates for OA, division, IGC, evaluator and IEA."""

import itertools
import math

import numpy as np
import pytest

import pyiea as ic
from pyiea.benchmarks import Costly, FixedCardWeighted, QuadraticBinary, WeightedSum
from pyiea.igc import decode, main_effects
from pyiea.problem import freeze, paper_n_segments


# --- OA ----------------------------------------------------------------------------
@pytest.mark.parametrize("N", [1, 3, 7, 10, 15])
def test_oa_balance(N):
    oa = ic.generate_oa(N)
    n = 2 ** math.ceil(math.log2(N + 1))
    assert oa.shape == (n, N) and set(np.unique(oa)) <= {1, 2}
    assert (oa[0] == 1).all()  # row 1 is parent 1
    assert ((oa == 1).sum(0) == n // 2).all()
    for a, b in itertools.combinations(range(N), 2):
        for la, lb in itertools.product((1, 2), repeat=2):
            assert ((oa[:, a] == la) & (oa[:, b] == lb)).sum() == n // 4


def test_oa_matches_paper_L4():
    assert ic.generate_oa(3).tolist() == [[1, 1, 1], [1, 2, 2], [2, 1, 2], [2, 2, 1]]


def test_paper_table2_main_effects():
    # Table II: y = 100 x1 - 10 x2 - x3 (maximize) on L4(2^3)
    oa = ic.generate_oa(3)
    levels = {0: (1, 2), 1: (3, 4), 2: (5, 6)}
    y = np.array([100 * levels[0][r[0] - 1] - 10 * levels[1][r[1] - 1] - levels[2][r[2] - 1] for r in oa])
    assert y.tolist() == [65, 54, 164, 155]
    s1, s2 = main_effects(oa, y)
    assert s1.tolist() == [119, 229, 220] and s2.tolist() == [319, 209, 218]


# --- division / decoding --------------------------------------------------------------
def test_segmentation_covers_diff_exactly_once():
    rng = np.random.default_rng(0)
    P = ic.BinaryProblem(40)
    for _ in range(50):
        p1, p2 = P.random_genome(rng), P.random_genome(rng)
        diff = np.flatnonzero(p1 != p2)
        segs = P.divide(p1, p2, diff, None, rng)
        if paper_n_segments(len(diff)) < 2:
            assert segs == []
            continue
        assert len(segs) == paper_n_segments(len(diff)) and all(len(s) for s in segs)
        assert sorted(np.concatenate(segs).tolist()) == diff.tolist()
        for r in ic.generate_oa(len(segs)):
            g = decode(p1, p2, segs, r)
            same = p1 == p2
            assert (g[same] == p1[same]).all()


def test_bounded_segments_cap():
    rng = np.random.default_rng(1)
    P = ic.BinaryProblem(64)
    p1, p2 = freeze(np.zeros(64)), freeze(np.ones(64))
    assert len(P.divide(p1, p2, np.arange(64), 7, rng)) == 7


def test_decode_by_hand():
    p1, p2 = freeze(np.array([1, 1, 0, 0, 1, 0])), freeze(np.array([0, 1, 1, 1, 0, 0]))
    segs = [np.array([0]), np.array([2, 3]), np.array([4])]
    assert decode(p1, p2, segs, [2, 1, 2]).tolist() == [0, 1, 0, 0, 0, 0]
    assert decode(p1, p2, segs, [1, 2, 1]).tolist() == [1, 1, 1, 1, 1, 0]


def test_fixed_cardinality_division_keeps_all_rows_legal():
    rng = np.random.default_rng(2)
    P = ic.FixedCardinalityProblem(30, 12)
    for _ in range(50):
        p1, p2 = P.random_genome(rng), P.random_genome(rng)
        segs = P.divide(p1, p2, np.flatnonzero(p1 != p2), None, rng)
        for r in ic.generate_oa(max(len(segs), 1)) if segs else []:
            assert P.is_valid(decode(p1, p2, segs, r))
        assert P.is_valid(P.mutate(p1, 0.5, rng)) and P.is_valid(P.crossover(p1, p2, rng))


# --- IGC -------------------------------------------------------------------------------------
def _ev(f, n_obj=1, **kw):
    return ic.Evaluator(f, n_obj, "test", **kw)


def _parents(f, P, rng, ev):
    p1, p2 = P.random_genome(rng), P.random_genome(rng)
    y1, y2 = (r.objectives for r in ev.evaluate_batch([p1, p2]))
    return p1, y1, p2, y2


def test_additive_child1_is_best_segment_combination():
    rng = np.random.default_rng(3)
    P, f = ic.BinaryProblem(24), WeightedSum(24, seed=3)
    for _ in range(20):
        ev = _ev(f)
        p1, y1, p2, y2 = _parents(f, P, rng, ev)
        state = rng.bit_generator.state
        r = ic.igc(p1, y1, p2, y2, P, ev, rng, step10=False)
        if r.status != "applied":
            continue
        rng2 = np.random.default_rng()
        rng2.bit_generator.state = state
        segs = P.divide(p1, p2, np.flatnonzero(p1 != p2), None, rng2)  # same draw as IGC
        best = min(f(decode(p1, p2, segs, lv)) for lv in itertools.product((1, 2), repeat=len(segs)))
        assert r.children[0][1][0] == pytest.approx(best)


def test_epistatic_children_really_evaluated_and_step10_keeps_best_row():
    rng = np.random.default_rng(4)
    P, f = ic.BinaryProblem(16), QuadraticBinary(16, seed=4)
    for _ in range(30):
        ev = _ev(f)
        p1, y1, p2, y2 = _parents(f, P, rng, ev)
        r = ic.igc(p1, y1, p2, y2, P, ev, rng, step10=True)
        if r.status != "applied":
            continue
        for g, y in r.children:
            assert y[0] == pytest.approx(f(g))  # actual fitness, not a main-effect estimate
        cands = [*sorted(y[0] for _, y in r.byproducts), r.trace["c1"], r.trace["c2"]]
        assert r.children[0][1][0] == pytest.approx(min(cands))
        assert r.children[0][1][0] <= r.trace["best_row"]


def test_identical_and_one_bit_parents_cost_nothing():
    P, f = ic.BinaryProblem(8), WeightedSum(8)
    ev = _ev(f)
    p = freeze(np.zeros(8))
    q = freeze(np.eye(8, dtype=np.uint8)[0])
    y = ev.evaluate_batch([p, q])
    calls = ev.counters["objective_calls"]
    rng = np.random.default_rng(0)
    assert ic.igc(p, y[0].objectives, p, y[0].objectives, P, ev, rng).status == "identical"
    assert ic.igc(p, y[0].objectives, q, y[1].objectives, P, ev, rng).status == "not_divisible"
    assert ev.counters["objective_calls"] == calls


def test_parents_immutable():
    p = ic.BinaryProblem(8).random_genome(np.random.default_rng(0))
    with pytest.raises(ValueError, match="read-only"):
        p[0] = 1 - p[0]


def test_main_effect_tie_prefers_parent1_reproducibly():
    P = ic.BinaryProblem(6)
    p1, p2 = freeze(np.zeros(6)), freeze(np.ones(6))
    out = []
    for _ in range(2):
        ev = _ev(lambda g: 0.0)
        r = ic.igc(p1, (0.0,), p2, (0.0,), P, ev, np.random.default_rng(9), step10=False)
        out.append(r.children[0][0].tolist())
    assert out[0] == out[1] == [0] * 6  # all ties -> every factor at level 1 (parent 1)


def test_failed_row_never_enters_main_effects():
    P = ic.BinaryProblem(8)
    p1, p2 = freeze(np.zeros(8)), freeze(np.ones(8))

    def f(g):
        return float("nan") if g[0] == 1 and g[7] == 0 else float(g.sum())

    ev = _ev(f)
    r = ic.igc(p1, (0.0,), p2, (8.0,), P, ev, np.random.default_rng(0))
    assert r.status == "failed_row" and [g for g, _ in r.children] == [p1, p2]
    assert ev.counters["failed_evaluations"] >= 1


# --- evaluator --------------------------------------------------------------------------------
class CountingObjective:
    def __init__(self):
        self.calls = 0

    def __call__(self, g):
        self.calls += 1
        return float(g.sum())


def test_accounting_matches_mock_calls_and_ledger():
    f = CountingObjective()
    P = ic.BinaryProblem(20)
    ev = _ev(f, is_valid=P.is_valid, max_calls=300)
    res = ic.IEA(P, ev, ic.IEAConfig(pop_size=10, max_generations=5), seed=0).optimize()
    acc = res.accounting
    assert f.calls == acc["objective_calls"] == len(ev.ledger) == len(ev.cache)
    assert acc["objective_calls"] <= 300
    assert {e["phase"] for e in ev.ledger} >= {"init", "oa", "child"}
    assert sum(r["calls_after"] - r["calls_before"] for r in res.igc_log if "calls_after" in r) == sum(
        e["phase"] in ("oa", "child") for e in ev.ledger
    )


def test_budget_never_exceeded_and_no_partial_oa():
    f = CountingObjective()
    P = ic.BinaryProblem(64)
    ev = _ev(f, max_calls=77)
    res = ic.IEA(P, ev, ic.IEAConfig(pop_size=20), seed=1).optimize()
    assert f.calls <= 77 and res.stop_reason == "budget_exhausted"
    for r in res.igc_log:
        if r["status"] == "applied":
            assert r["calls_after"] - r["calls_before"] <= r["n_rows"] + 2
    with pytest.raises(ic.BudgetExhaustedError):
        ev.evaluate_batch([P.random_genome(np.random.default_rng(s)) for s in range(99)])


def test_exceptions_and_nonfinite_are_failures():
    def f(g):
        if g[0]:
            raise RuntimeError("boom")
        return math.inf if g[1] else 1.0

    ev = _ev(f)
    a, b, c = (freeze(np.array(v)) for v in ([1, 0], [0, 1], [0, 0]))
    ra, rb, rc = ev.evaluate_batch([a, b, c])
    assert (ra.status, rb.status, rc.status) == ("failed", "failed", "ok")
    assert ev.counters["failed_evaluations"] == 2 and "boom" in ev.ledger[0]["error"]


def test_invalid_candidates_skip_objective():
    f = CountingObjective()
    P = ic.FixedCardinalityProblem(6, 3)
    ev = _ev(f, is_valid=P.is_valid)
    (r,) = ev.evaluate_batch([freeze(np.ones(6))])
    assert r.status == "invalid" and f.calls == 0 and ev.counters["invalid_candidates"] == 1


def test_cache_and_context_guard():
    f = CountingObjective()
    ev = _ev(f)
    g = freeze(np.ones(5))
    ev.evaluate_batch([g, g])
    ev.evaluate_batch([g])
    assert f.calls == 1 and ev.counters["cache_hits"] == 2
    other = ic.Evaluator(f, 1, "other-context")
    with pytest.raises(ic.CheckpointError):
        other.load_state_dict(ev.state_dict())


def test_serial_and_parallel_backends_agree():
    rng = np.random.default_rng(0)
    gs = [ic.BinaryProblem(16).random_genome(rng) for _ in range(12)]
    f = Costly(QuadraticBinary(16, seed=1), work=5)
    serial = _ev(f).evaluate_batch(gs)
    ev = _ev(f, workers=2)
    par = ev.evaluate_batch(gs)
    ev.close()
    assert [r.objectives for r in serial] == [r.objectives for r in par]


# --- IEA --------------------------------------------------------------------------------------
def test_iea_best_so_far_never_regresses():
    P, f = ic.BinaryProblem(30), QuadraticBinary(30, seed=5)
    res = ic.IEA(P, _ev(f, max_calls=1500), ic.IEAConfig(), seed=5).optimize()
    bests = [h["best"] for h in res.history]
    assert bests == sorted(bests, reverse=True)
    # truncation, Step 10 (row 1 = I_best) and the mutation exemption keep the population best
    assert res.pop_best == sorted(res.pop_best, reverse=True)
    assert res.best == pytest.approx(f(res.best_genome))


def test_iea_fixed_cardinality_stays_legal_and_solves_additive():
    P, f = ic.FixedCardinalityProblem(40, 15), FixedCardWeighted(40, 15, seed=6)
    ev = _ev(f, is_valid=P.is_valid, max_calls=4000)
    res = ic.IEA(P, ev, ic.IEAConfig(), seed=6).optimize()
    assert ev.counters["invalid_candidates"] == 0
    assert res.best == pytest.approx(f.optimum())


def test_iea_maximization_by_documented_negation():
    # objectives are minimized; a maximization objective is passed as its negation
    P, f = ic.BinaryProblem(20), WeightedSum(20, seed=7)
    res = ic.IEA(P, _ev(lambda g: -f(g), max_calls=1500), ic.IEAConfig(), seed=7).optimize()
    assert -res.best == pytest.approx(f.w[f.w > 0].sum())


def test_iea_checkpoint_resume_equals_uninterrupted(tmp_path):
    P, f = ic.BinaryProblem(24), QuadraticBinary(24, seed=8)
    cfg = ic.IEAConfig(pop_size=10, max_generations=8, checkpoint_every=4)
    full = ic.IEA(P, _ev(f), cfg, seed=8).optimize(checkpoint_path=tmp_path / "a.pkl")

    cut = ic.IEAConfig(**{**cfg.__dict__, "max_generations": 4})
    ic.IEA(P, _ev(f), cut, seed=8).optimize(checkpoint_path=tmp_path / "b.pkl")  # saves at gen 4
    assert ic.load_checkpoint(tmp_path / "b.pkl")["gen"] == 4
    # extending a run with a different stopping condition is allowed
    resumed = ic.IEA(P, _ev(f), cfg, seed=8).optimize(checkpoint_path=tmp_path / "b.pkl", resume=True)
    assert resumed.best == full.best and resumed.history == [
        {**h, "seconds": resumed.history[i]["seconds"]} for i, h in enumerate(full.history)
    ]
    for k in ("objective_calls", "cache_hits", "proposals"):
        assert resumed.accounting[k] == full.accounting[k]


def test_exhausted_search_space_stops_as_stalled():
    P, f = ic.BinaryProblem(3), WeightedSum(3)
    res = ic.IEA(P, _ev(f), ic.IEAConfig(pop_size=6, max_stall_generations=5), seed=0).optimize()
    assert res.stop_reason == "stalled" and res.accounting["objective_calls"] <= 8
    from pyiea.benchmarks import KeepDropBiObjective

    r = ic.IMOEA(
        P, _ev(KeepDropBiObjective(3), 2), ic.IMOEAConfig(pop_size=6, max_stall_generations=5), seed=0
    ).optimize()
    assert r.stop_reason == "stalled"


def test_checkpoint_rejects_other_algorithm_settings_seed_or_context(tmp_path):
    P, f = ic.BinaryProblem(16), QuadraticBinary(16, seed=9)
    ic.IEA(P, _ev(f), ic.IEAConfig(pop_size=10, max_generations=2, checkpoint_every=1), seed=1).optimize(
        checkpoint_path=tmp_path / "c.pkl"
    )
    with pytest.raises(ic.CheckpointError):
        ic.IEA(P, _ev(f), ic.IEAConfig(pop_size=10, pm=0.1), seed=1).optimize(tmp_path / "c.pkl", resume=True)
    with pytest.raises(ic.CheckpointError):
        ic.IEA(P, _ev(f), ic.IEAConfig(pop_size=10), seed=2).optimize(tmp_path / "c.pkl", resume=True)
    with pytest.raises(ic.CheckpointError):
        ic.IEA(P, ic.Evaluator(f, 1, "other"), ic.IEAConfig(pop_size=10), seed=1).optimize(
            tmp_path / "c.pkl", resume=True
        )
