"""Comparison baselines. These are **not** IEA or IMOEA and must never be reported as such.

All of them use the problem's initialization, legality, mutation and crossover
and the same :class:`~pyiea.Evaluator`, so budgets, caching and accounting are
identical to the IEA/IMOEA runs they are compared with. They return the same
result types (:class:`~pyiea.IEAResult`, :class:`~pyiea.IMOEAResult`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from .evaluator import Evaluator, Objectives
from .iea import IEAResult
from .imoea import IMOEAResult
from .pareto import ParetoSet, dominance_matrix
from .problem import Genome, Problem, freeze

__all__ = ["CROSSOVERS", "elitist_ga", "nsga2", "random_search", "vanilla_ga"]


class CrossoverProblem(Problem, Protocol):
    def crossover(self, p1: Genome, p2: Genome, rng: np.random.Generator) -> Genome: ...


def _eval_all(ev: Evaluator, genomes: Sequence[Genome], phase: str) -> tuple[list[Objectives | None], bool]:
    res = ev.evaluate_batch(genomes, phase=phase, partial_ok=True)
    return [r.objectives if r is not None else None for r in res], all(r is not None for r in res)


def _budget_left(ev: Evaluator) -> bool:
    return ev.remaining_calls() > 0 and not ev.time_up()


class _Stall:
    """The IEA/IMOEA ``max_stall_generations`` rule: stop after ``k`` generations without a new call."""

    def __init__(self, ev: Evaluator, k: int = 50) -> None:
        self.ev, self.k, self.calls, self.n = ev, k, -1, 0

    def __call__(self) -> bool:
        c = self.ev.counters["objective_calls"]
        self.n = self.n + 1 if c == self.calls else 0
        self.calls = c
        return self.n >= self.k


def _tournament(rng: np.random.Generator, keys: Sequence[Any]) -> int:
    """Binary tournament: the smaller key of two uniformly drawn individuals wins."""
    a, b = (int(i) for i in rng.integers(0, len(keys), 2))
    return a if keys[a] <= keys[b] else b


def _stop(ev: Evaluator, stalled: bool) -> str:
    if stalled:
        return "stalled"
    return "time_limit" if ev.time_up() else "budget_exhausted"


class _Best:
    def __init__(self, ev: Evaluator) -> None:
        self.ev = ev
        self.best: tuple[Genome, Objectives] | None = None
        self.history: list[dict[str, Any]] = []

    def observe(self, gs: Sequence[Genome], ys: Sequence[Objectives | None]) -> None:
        for g, y in zip(gs, ys, strict=True):
            if y is not None and (self.best is None or y[0] < self.best[1][0]):
                self.best = (g, y)
                self.history.append(
                    {"calls": self.ev.counters["objective_calls"], "seconds": self.ev.elapsed(), "best": y[0]}
                )

    def result(self, generations: int, reason: str) -> IEAResult:
        return IEAResult(
            best_genome=self.best[0] if self.best else None,
            best=self.best[1][0] if self.best else None,
            generations=generations,
            stop_reason=reason,
            history=self.history,
            pop_best=[],
            igc_log=[],
            accounting=self.ev.summary(),
        )


def random_search(problem: Problem, evaluator: Evaluator, seed: int = 0, batch: int = 32) -> IEAResult | IMOEAResult:
    """Uniform samples from ``problem.random_genome`` until the budget is spent.

    Returns an :class:`IMOEAResult` (front = archive) when ``evaluator.n_objectives > 1``.
    """
    rng = np.random.default_rng(seed)
    multi = evaluator.n_objectives > 1
    tracker, archive = _Best(evaluator), ParetoSet()
    stall, stalled, rounds = _Stall(evaluator), False, 0
    while _budget_left(evaluator) and not (stalled := stall()):
        gs = [problem.random_genome(rng) for _ in range(batch)]
        ys, _ = _eval_all(evaluator, gs, "random")
        rounds += 1
        if multi:
            archive.update([(g, y) for g, y in zip(gs, ys, strict=True) if y is not None])
        else:
            tracker.observe(gs, ys)
    reason = _stop(evaluator, stalled)
    if multi:
        return IMOEAResult(
            front=archive.items(),
            archive=archive.items(),
            generations=rounds,
            stop_reason=reason,
            accounting=evaluator.summary(),
        )
    return tracker.result(rounds, reason)


def vanilla_ga(
    problem: CrossoverProblem,
    evaluator: Evaluator,
    seed: int = 0,
    pop_size: int = 30,
    pc: float = 0.8,
    pm: float = 0.05,
) -> IEAResult:
    """Generational GA: binary tournament, ``problem.crossover``, ``problem.mutate``, one elite."""
    rng = np.random.default_rng(seed)
    pop = [problem.random_genome(rng) for _ in range(pop_size)]
    ys, _ = _eval_all(evaluator, pop, "init")
    tracker = _Best(evaluator)
    tracker.observe(pop, ys)

    def key(y: Objectives | None) -> float:
        return np.inf if y is None else y[0]

    stall, stalled, gen = _Stall(evaluator), False, 0
    while _budget_left(evaluator) and not (stalled := stall()):
        keys = [key(y) for y in ys]
        e = int(np.argmin(keys))
        children = [pop[e]]
        while len(children) < pop_size:
            p1, p2 = pop[_tournament(rng, keys)], pop[_tournament(rng, keys)]
            c = problem.crossover(p1, p2, rng) if rng.random() < pc else p1
            children.append(problem.mutate(c, pm, rng))
        cys, complete = _eval_all(evaluator, children[1:], "ga")
        pop, ys = children, [ys[e], *cys]
        tracker.observe(children[1:], cys)
        gen += 1
        if not complete:
            break
    return tracker.result(gen, _stop(evaluator, stalled))


def _one_point(p1: Genome, p2: Genome, rng: np.random.Generator) -> tuple[Genome, Genome]:
    c = int(rng.integers(1, len(p1)))
    return freeze(np.r_[p1[:c], p2[c:]]), freeze(np.r_[p2[:c], p1[c:]])


def _two_point(p1: Genome, p2: Genome, rng: np.random.Generator) -> tuple[Genome, Genome]:
    a, b = sorted(int(v) for v in rng.choice(np.arange(1, len(p1)), 2, replace=False))
    return freeze(np.r_[p1[:a], p2[a:b], p1[b:]]), freeze(np.r_[p2[:a], p1[a:b], p2[b:]])


def _uniform(p1: Genome, p2: Genome, rng: np.random.Generator) -> tuple[Genome, Genome]:
    take = rng.random(len(p1)) < 0.5
    return freeze(np.where(take, p1, p2)), freeze(np.where(take, p2, p1))


CROSSOVERS = {"one_point": _one_point, "two_point": _two_point, "uniform": _uniform}
"""Two-child bit-string crossovers of :func:`elitist_ga`: OEGA, TEGA and UEGA of the IEA paper."""


def elitist_ga(
    problem: Problem,
    evaluator: Evaluator,
    seed: int = 0,
    crossover: str = "one_point",
    pop_size: int = 10,
    ps: float = 0.2,
    pc: float = 0.8,
    pm: float = 0.05,
    max_stall_generations: int = 50,
) -> IEAResult:
    """Elitist GA with a conventional crossover: the OEGA, TEGA and UEGA of Section V-B.

    The paper compares IGC with one-point, two-point and uniform crossover inside
    "EAs with elitist strategy" that share N_pop, ps, pc and pm. This loop mirrors
    IEA Steps 1-6 (:class:`pyiea.IEA`) with IGC replaced by ``crossover``:

    1. random population; 2. evaluate the new or changed individuals;
    3. truncation: the worst ``int(ps * N_pop)`` are replaced by copies of the best;
    4. ``int(pc * N_pop)`` (rounded down to even) parents drawn without replacement from
       every individual except the elite (index 0 after sorting), paired in draw order;
       each pair is replaced by its two children, which are evaluated in step 2;
    5. bit-inverse mutation (``problem.mutate``) of everything except the elite.

    The elite is never recombined or mutated, so the best value never gets worse.
    Crossover works on unconstrained bit strings; an infeasible child is an invalid
    candidate. Stops when the budget runs out or after ``max_stall_generations``
    generations without a new objective call.

    Args:
        problem: Search space (initialization and mutation).
        evaluator: Objective, budget and accounting shared with the IEA run it is compared with.
        seed: Seed of the run's random generator.
        crossover: ``"one_point"`` (OEGA), ``"two_point"`` (TEGA) or ``"uniform"`` (UEGA).
        pop_size: ``N_pop``.
        ps: Fraction of the population replaced by copies of the best (truncation).
        pc: Fraction of the population recombined per generation.
        pm: Per-bit mutation probability.
        max_stall_generations: Stop after this many generations without a new objective call.
    """
    if crossover not in CROSSOVERS:
        raise ValueError(f"unknown crossover {crossover!r}; choose from {tuple(CROSSOVERS)}")
    cross = CROSSOVERS[crossover]
    rng = np.random.default_rng(seed)
    pop = [problem.random_genome(rng) for _ in range(pop_size)]
    ys: list[Objectives | None] = [None] * pop_size
    tracker = _Best(evaluator)
    n_replace = int(ps * pop_size)
    n_parents = min(int(pc * pop_size) // 2 * 2, (pop_size - 1) // 2 * 2)
    stall, stalled, gen = _Stall(evaluator, max_stall_generations), False, 0

    def key(i: int) -> float:
        y = ys[i]
        return np.inf if y is None else y[0]

    while True:
        todo = [i for i in range(pop_size) if ys[i] is None]
        new, complete = _eval_all(evaluator, [pop[i] for i in todo], "init" if gen == 0 else "ga")
        for i, y in zip(todo, new, strict=True):
            ys[i] = y
        tracker.observe([pop[i] for i in todo], new)
        if not complete or not _budget_left(evaluator) or (stalled := stall()):
            break
        order = sorted(range(pop_size), key=key)
        idx = order[: pop_size - n_replace] + order[:n_replace]
        pop, ys = [pop[i] for i in idx], [ys[i] for i in idx]  # [0] is the elite
        chosen = rng.choice(np.arange(1, pop_size), n_parents, replace=False).tolist()
        for a, b in zip(chosen[0::2], chosen[1::2], strict=True):
            (pop[a], pop[b]), ys[a], ys[b] = cross(pop[a], pop[b], rng), None, None
        for i in range(1, pop_size):
            g = problem.mutate(pop[i], pm, rng)
            if g is not pop[i]:
                pop[i], ys[i] = g, None
        gen += 1
    return tracker.result(gen, _stop(evaluator, stalled))


def _nd_sort_crowding(Y: Sequence[Objectives]) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """NSGA-II non-domination ranks and crowding distances."""
    A = np.asarray(Y, dtype=float)
    D = dominance_matrix(A)
    rank = np.full(len(A), -1)
    dominated_by = D.sum(0)
    r, front = 0, np.flatnonzero(dominated_by == 0)
    while len(front):
        rank[front] = r
        dominated_by = dominated_by - D[front].sum(0)
        dominated_by[rank >= 0] = -1
        front = np.flatnonzero(dominated_by == 0)
        r += 1
    crowd = np.zeros(len(A))
    for r in range(rank.max() + 1):
        idx = np.flatnonzero(rank == r)
        for m in range(A.shape[1]):
            o = idx[np.argsort(A[idx, m])]
            crowd[o[[0, -1]]] = np.inf
            span = A[o[-1], m] - A[o[0], m]
            if len(o) > 2 and span > 0:
                crowd[o[1:-1]] += (A[o[2:], m] - A[o[:-2], m]) / span
    return rank, crowd


def nsga2(
    problem: CrossoverProblem,
    evaluator: Evaluator,
    seed: int = 0,
    pop_size: int = 30,
    pc: float = 0.9,
    pm: float | None = None,
) -> IMOEAResult:
    """NSGA-II (Deb et al. 2002) with the problem's crossover and mutation (``pm`` defaults to 1/n_bits)."""
    rng = np.random.default_rng(seed)
    pm = 1.0 / problem.n_bits if pm is None else pm
    pop = [problem.random_genome(rng) for _ in range(pop_size)]
    ys, _ = _eval_all(evaluator, pop, "init")
    archive = ParetoSet((g, y) for g, y in zip(pop, ys, strict=True) if y is not None)
    stall, stalled, gen = _Stall(evaluator), False, 0
    while _budget_left(evaluator) and not (stalled := stall()):
        valid = [(g, y) for g, y in zip(pop, ys, strict=True) if y is not None]
        pop, oks = [g for g, _ in valid], [y for _, y in valid]
        rank, crowd = _nd_sort_crowding(oks)
        keys = [(float(r), -float(c)) for r, c in zip(rank, crowd, strict=True)]
        kids: list[Genome] = []
        while len(kids) < pop_size:
            p1, p2 = pop[_tournament(rng, keys)], pop[_tournament(rng, keys)]
            c = problem.crossover(p1, p2, rng) if rng.random() < pc else p1
            kids.append(problem.mutate(c, pm, rng))
        kys, complete = _eval_all(evaluator, kids, "nsga2")
        new = [(g, y) for g, y in zip(kids, kys, strict=True) if y is not None]
        archive.update(new)
        allg, ally = pop + [g for g, _ in new], oks + [y for _, y in new]
        rank, crowd = _nd_sort_crowding(ally)
        order = sorted(range(len(allg)), key=lambda i: (rank[i], -crowd[i]))[:pop_size]
        pop, ys = [allg[i] for i in order], [ally[i] for i in order]
        gen += 1
        if not complete:
            break
    front = ParetoSet((g, y) for g, y in zip(pop, ys, strict=True) if y is not None)
    return IMOEAResult(
        front=front.items(),
        archive=archive.items(),
        generations=gen,
        stop_reason=_stop(evaluator, stalled),
        accounting=evaluator.summary(),
    )
