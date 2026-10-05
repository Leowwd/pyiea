"""Tests proposed by the 2026-10-05 mutation review; each closes a gap found by a surviving mutant."""

import numpy as np
import pytest

import pyiea as ic
from pyiea.problem import freeze

LEVELS = ((1, 2), (3, 4), (5, 6))  # Table II: level k of factor j


class OneBitFactors(ic.BinaryProblem):
    """Every differing bit is its own factor, in order (lets a test fix the division)."""

    def divide(self, p1, p2, diff, max_segments, rng):
        return [diff[i : i + 1] for i in range(len(diff))]


def table2(g):  # maximize y = 100 x1 - 10 x2 - x3, passed negated
    x = [LEVELS[j][int(g[j])] for j in range(3)]
    return -(100 * x[0] - 10 * x[1] - x[2])


@pytest.mark.parametrize("step10", [False, True])
def test_igc_reproduces_paper_table2_children(step10):
    """Comb1 = (2,3,5) with y=165; Comb2 flips x3, the factor with the smallest MED, giving (2,3,6), y=164."""
    P, ev = OneBitFactors(3), ic.Evaluator(table2, 1, "t2")
    p1, p2 = freeze([0, 0, 0]), freeze([1, 1, 1])
    r = ic.igc(p1, (table2(p1),), p2, (table2(p2),), P, ev, np.random.default_rng(0), step10=step10)
    assert r.status == "applied" and r.trace["med_factor"] == 2
    assert sorted(-y[0] for _, y in r.byproducts) == [54, 65, 155, 164]
    assert [g.tolist() for g, _ in r.children] == [[1, 0, 0], [1, 0, 1]]
    assert [-y[0] for _, y in r.children] == [165, 164]
    assert ev.counters["objective_calls"] == 4 + 1  # C2 equals OA row 3: a cache hit, not a call


def test_a_failed_child_returns_the_parents_and_keeps_the_rows():
    """C1 = (1,0,0) is not an OA row of Table II; if it fails the IGC must hand back the parents."""

    def f(g):
        return float("nan") if g.tolist() == [1, 0, 0] else table2(g)

    P, ev = OneBitFactors(3), ic.Evaluator(f, 1, "t2-fail")
    p1, p2 = freeze([0, 0, 0]), freeze([1, 1, 1])
    r = ic.igc(p1, (table2(p1),), p2, (table2(p2),), P, ev, np.random.default_rng(0))
    assert r.status == "failed_row"
    assert [g.tolist() for g, _ in r.children] == [[0, 0, 0], [1, 1, 1]]
    assert len(r.byproducts) == 4 and ev.counters["failed_evaluations"] == 1
    assert r.trace["calls_after"] - r.trace["calls_before"] == 5


def test_a_failed_row_records_its_calls_in_the_trace():
    P = ic.BinaryProblem(8)
    p1, p2 = freeze(np.zeros(8)), freeze(np.ones(8))
    ev = ic.Evaluator(lambda g: float("nan") if g[0] == 1 and g[7] == 0 else float(g.sum()), 1, "fr")
    r = ic.igc(p1, (0.0,), p2, (8.0,), P, ev, np.random.default_rng(0))
    assert r.status == "failed_row" and r.trace["calls_after"] - r.trace["calls_before"] == r.trace["n_rows"]


def test_trace_best_row_and_child_beats_rows():
    P, ev = OneBitFactors(3), ic.Evaluator(table2, 1, "t2-trace")
    p1, p2 = freeze([0, 0, 0]), freeze([1, 1, 1])
    t = ic.igc(p1, (table2(p1),), p2, (table2(p2),), P, ev, np.random.default_rng(0)).trace
    assert (t["best_row"], t["c1"], t["c2"], t["child_beats_rows"]) == (-164, -165, -164, True)
    # children equal to the best row do not beat it
    ev = ic.Evaluator(lambda g: 0.0, 1, "flat")
    t = ic.igc(p1, (0.0,), p2, (0.0,), P, ev, np.random.default_rng(0)).trace
    assert t["best_row"] == t["c1"] == 0.0 and t["child_beats_rows"] is False


def test_config_boundaries():
    ic.IEAConfig(pop_size=2, pc=1.0)  # the smallest legal population
    ic.IEAConfig(max_segments=2)  # the smallest legal cap
    with pytest.raises(ValueError, match="pair"):
        ic.IEAConfig(pop_size=10, pc=0.1)  # pc * pop_size == 1 selects no pair


def test_oa_needs_at_least_one_factor():
    with pytest.raises(ValueError, match="n_factors"):
        ic.generate_oa(0)


def test_paper_division_size_values():
    """N = 2**floor(log2(M+1)) - 1 (Section III-A), checked against the formula's values, not against itself."""
    from pyiea.problem import paper_n_segments

    expect = {1: 1, 2: 1, 3: 3, 6: 3, 7: 7, 14: 7, 15: 15, 30: 15, 31: 31, 127: 127, 128: 127}
    assert {m: paper_n_segments(m) for m in expect} == expect


def test_divide_contract_two_segments_and_empty_list():
    rng = np.random.default_rng(0)
    p1, p2 = freeze(np.zeros(16)), freeze(np.ones(16))
    segs = ic.BinaryProblem(16).divide(p1, p2, np.arange(16), 2, rng)  # max_segments=2 is legal
    assert len(segs) == 2 and sorted(np.concatenate(segs).tolist()) == list(range(16))
    one = freeze(np.eye(16, dtype=np.uint8)[0])
    assert ic.BinaryProblem(16).divide(p1, one, np.array([0]), None, rng) == []  # M=1: N=1 -> []
    a, b = freeze([1, 1, 0, 0]), freeze([0, 0, 1, 1])  # one balanced block only
    assert ic.FixedCardinalityProblem(4, 2).divide(a, b, np.arange(4), None, rng) == []
    # the paper's PAP example (Section III-A): P1 = 10 10 111000, P2 = 01 01 010101 -> N = 3
    p, q = freeze([1, 0, 1, 0, 1, 1, 1, 0, 0, 0]), freeze([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    segs = ic.FixedCardinalityProblem(10, 5).divide(p, q, np.flatnonzero(p != q), None, rng)
    assert [s.tolist() for s in segs] == [[0, 1], [2, 3], [4, 6, 7, 9]]


@pytest.mark.parametrize("cap", [2, 3, 5])
def test_fixed_cardinality_bounded_segments(cap):
    """With max_segments the division keeps min(natural, cap) non-empty, balanced, contiguous segments."""
    rng = np.random.default_rng(cap)
    P = ic.FixedCardinalityProblem(60, 20)
    capped_seen = 0
    for _ in range(60):
        p1, p2 = P.random_genome(rng), P.random_genome(rng)
        diff = np.flatnonzero(p1 != p2)
        natural = len(P.divide(p1, p2, diff, None, rng)) or 1
        segs = P.divide(p1, p2, diff, cap, rng)
        if min(natural, cap) < 2:
            assert segs == []
            continue
        capped_seen += natural > cap
        assert len(segs) == min(natural, cap) and all(len(s) for s in segs)
        assert np.concatenate(segs).tolist() == diff.tolist()
        assert all(int(p1[s].sum()) == int(p2[s].sum()) for s in segs)
    assert capped_seen > 10
    a, b = freeze([1, 0, 1, 0]), freeze([0, 1, 0, 1])  # two natural segments, a cap of 3 leaves them alone
    assert len(ic.FixedCardinalityProblem(4, 2).divide(a, b, np.arange(4), 3, rng)) == 2


def test_history_records_strict_improvements_only():
    res = ic.IEA(
        ic.BinaryProblem(20), ic.Evaluator(lambda g: 1.0, 1, "flat", max_calls=300), ic.IEAConfig(pop_size=10), seed=0
    ).optimize()
    assert len(res.history) == 1


def test_ledger_phase_labels():
    ev = ic.Evaluator(lambda g: float(g.sum()), 1, "ph", max_calls=400)
    ic.IEA(ic.BinaryProblem(40), ev, ic.IEAConfig(pop_size=10), seed=0).optimize()
    phases = [e["phase"] for e in ev.ledger]
    assert phases[:10] == ["init"] * 10 and "init" not in phases[10:]
    assert {"oa", "child", "mutation"} <= set(phases[10:])


def test_resume_needs_a_checkpoint_path():
    with pytest.raises(ValueError, match="checkpoint_path"):
        ic.IEA(ic.BinaryProblem(4), ic.Evaluator(lambda g: 0.0, 1, "c"), ic.IEAConfig(pop_size=4, pc=0.5)).optimize(
            resume=True
        )


def _recorded_pairs(monkeypatch, seed, ps=0.0, pop=10, pc=0.8):
    import pyiea.iea as iea_module

    calls = []
    real = iea_module.igc

    def spy(p1, y1, p2, y2, *a, **k):
        calls.append((p1, y1, p2, y2))
        return real(p1, y1, p2, y2, *a, **k)

    monkeypatch.setattr(iea_module, "igc", spy)
    f = lambda g: float(np.random.default_rng(int.from_bytes(g.tobytes()[:8], "little")).normal())  # noqa: E731
    it = ic.IEA(ic.BinaryProblem(64), ic.Evaluator(f, 1, "sel"), ic.IEAConfig(pop_size=pop, ps=ps, pc=pc), seed=seed)
    it.pop = [it.problem.random_genome(it.rng) for _ in range(pop)]
    it.pop_y = [None] * pop
    it.evaluate_population()
    ys = sorted(y[0] for y in it.pop_y)
    it.step()
    return calls, ys


@pytest.mark.parametrize("seed", range(5))
def test_step4_pairs_ibest_first_and_draws_distinct_parents(monkeypatch, seed):
    calls, ys = _recorded_pairs(monkeypatch, seed)
    assert len(calls) == int(0.8 * 10) // 2  # pc * N_pop parents -> 4 IGCs
    assert calls[0][1][0] == ys[0]  # I_best is parent 1 of the first pair
    parents = [y[0] for c in calls for y in (c[1], c[3])]
    assert len(set(parents)) == len(parents)  # ps = 0: no copies, so distinct values = distinct individuals


def test_step4_can_draw_the_second_best(monkeypatch):
    seen = 0
    for seed in range(20):
        calls, ys = _recorded_pairs(monkeypatch, seed)
        seen += ys[1] in [y[0] for c in calls for y in (c[1], c[3])]
    assert seen > 0


def test_individuals_without_objective_are_never_recombined(monkeypatch):
    import pyiea.iea as iea_module

    real = iea_module.igc

    def guarded(p1, y1, p2, y2, *a, **k):
        assert y1 is not None and y2 is not None
        return real(p1, y1, p2, y2, *a, **k)

    monkeypatch.setattr(iea_module, "igc", guarded)
    f = lambda g: float("nan") if g[0] else float(g.sum())  # noqa: E731  (half the genomes fail)
    res = ic.IEA(
        ic.BinaryProblem(30), ic.Evaluator(f, 1, "nan", max_calls=600), ic.IEAConfig(pop_size=10), seed=1
    ).optimize()
    assert res.accounting["failed_evaluations"] > 0


def test_step5_mutates_everyone_but_the_best():
    it = ic.IEA(
        ic.BinaryProblem(30),
        ic.Evaluator(lambda g: float(g @ np.arange(30)), 1, "m"),
        ic.IEAConfig(pop_size=10, pm=1.0, igc=False),
        seed=0,
    )
    it.pop = [it.problem.random_genome(it.rng) for _ in range(10)]
    it.pop_y = [None] * 10
    it.evaluate_population()
    best = min(zip(it.pop_y, it.pop), key=lambda t: t[0][0])
    it.step()
    kept = [i for i, y in enumerate(it.pop_y) if y is not None]
    assert len(kept) == 1 and it.pop[kept[0]].tolist() == best[1].tolist()


def test_budget_is_used_exactly_and_labelled():
    # igc=False: 10 initial calls, then 9 mutated individuals per generation (pm=0.5 on 40 bits never repeats)
    for max_calls, gens in ((91, 9), (92, 10)):
        ev = ic.Evaluator(lambda g: float(g @ np.arange(40)), 1, "b", max_calls=max_calls)
        res = ic.IEA(ic.BinaryProblem(40), ev, ic.IEAConfig(pop_size=10, pm=0.5, igc=False), seed=0).optimize()
        assert (res.stop_reason, ev.counters["objective_calls"], res.generations) == (
            "budget_exhausted",
            max_calls,
            gens,
        )


def test_time_limit_stops_with_its_own_reason(monkeypatch):
    import time

    clock = [0.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])

    def f(g):
        clock[0] += 1.0  # one second per call
        return float(g.sum())

    ev = ic.Evaluator(f, 1, "t", max_seconds=10.5)
    res = ic.IEA(ic.BinaryProblem(40), ev, ic.IEAConfig(pop_size=10, pm=0.5, igc=False), seed=0).optimize()
    assert res.stop_reason == "time_limit" and ev.counters["objective_calls"] == 19


def test_no_call_once_the_time_is_up():
    ev = ic.Evaluator(lambda g: 0.0, 1, "t0", max_seconds=0.0)
    assert ev.evaluate_batch([freeze([0, 1])], partial_ok=True) == [None] and ev.counters["objective_calls"] == 0


@pytest.mark.parametrize("target", [-20.0, -3.0])  # -20 is the optimum: reaching it needs best <= target
def test_target_stops_the_run(target):
    f = lambda g: -float(g.sum())  # noqa: E731
    ev = ic.Evaluator(f, 1, "tg", max_calls=5000)
    res = ic.IEA(ic.BinaryProblem(20), ev, ic.IEAConfig(pop_size=10, target=target), seed=0).optimize()
    assert res.stop_reason == "target_reached" and res.best <= target
    assert ev.counters["objective_calls"] < 5000


def test_max_generations_is_exact():
    res = ic.IEA(
        ic.BinaryProblem(40),
        ic.Evaluator(lambda g: float(g.sum()), 1, "g"),
        ic.IEAConfig(pop_size=10, max_generations=5),
        seed=0,
    ).optimize()
    assert res.stop_reason == "max_generations" and res.generations == 5 and len(res.pop_best) == 6


def test_stall_rule_fires_exactly_after_k_generations_without_calls():
    k = 5
    it = ic.IEA(
        ic.BinaryProblem(4),
        ic.Evaluator(lambda g: float(g.sum()), 1, "s"),
        ic.IEAConfig(pop_size=6, max_stall_generations=k),
        seed=0,
    )
    res = it.optimize()
    log = it._calls_log
    assert res.stop_reason == "stalled" and log[-1] == log[-1 - k]
    assert all(log[i] != log[i - k] for i in range(k, len(log) - 1))
    # k-1 generations without a call followed by a new call is not a stall
    it._calls_log = [10] * k + [11]
    assert it._stop_reason(True) is None


class Tokens:
    tokens_per_call = 7

    def __call__(self, g):
        return float(g.sum())


def test_evaluator_counters():
    ev = ic.Evaluator(Tokens(), 1, "cnt", is_valid=lambda g: g[0] == 0)
    a, b, bad = freeze([0, 1]), freeze([0, 0]), freeze([1, 1])
    ev.evaluate_batch([a, a, b, bad])
    ev.evaluate_batch([a])
    c = ev.counters
    assert (c["proposals"], c["unique_candidates"], c["objective_calls"], c["cache_hits"]) == (5, 3, 2, 2)
    assert (c["invalid_candidates"], c["evaluated_tokens"]) == (1, 14)
    plain = ic.Evaluator(lambda g: 0.0, 1, "plain")
    plain.evaluate_batch([a, b])
    assert plain.counters["evaluated_tokens"] == 0  # no tokens_per_call attribute: no tokens


def test_all_invalid_initial_population_names_the_cause():
    ev = ic.Evaluator(lambda g: 0.0, 1, "inv", is_valid=lambda g: False, max_calls=100)
    with pytest.raises(ic.EvaluationError, match="invalid genome"):
        ic.IEA(ic.BinaryProblem(8), ev, ic.IEAConfig(pop_size=4, pc=0.5), seed=0).optimize()
