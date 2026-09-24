"""Intelligent gene collector (IGC), Section III-C Steps 1-10.

Single-objective: the response ``y_t`` of combination ``t`` is its objective (minimized).
Multi-objective (IMOEA): the response is GPSIFF computed over the ``n`` OA
combinations of this IGC (maximized). See ``docs/paper_map.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from .evaluator import Evaluator, Objectives
from .oa import generate_oa
from .pareto import gpsiff
from .problem import Genome, Problem, freeze

__all__ = ["IGCResult", "decode", "igc", "main_effects"]

Individual = tuple[Genome, Objectives]


@dataclass
class IGCResult:
    """Outcome of one IGC operation.

    Attributes:
        status: ``"applied"``, ``"identical"`` (parents equal), ``"not_divisible"``
            (fewer than two segments), ``"budget"`` (the OA plus two children do not
            fit the budget; nothing was evaluated) or ``"failed_row"`` (a combination
            or child failed; parents are returned).
        children: Two evaluated ``(genome, objectives)`` pairs.
        byproducts: Evaluated OA combinations.
        trace: Diagnostics (``M``, ``N``, ``n_rows``, call counts, ...).
    """

    status: str
    children: list[Individual]
    byproducts: list[Individual] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


def decode(p1: Genome, p2: Genome, segments: Sequence[npt.NDArray[np.intp]], levels: npt.ArrayLike) -> Genome:
    """Build a combination: level 1 of factor ``j`` takes segment ``j`` from ``p1``, level 2 from ``p2``."""
    g = p1.copy()
    for seg, lv in zip(segments, np.asarray(levels), strict=True):
        if lv == 2:
            g[seg] = p2[seg]
    return freeze(g)


def main_effects(
    oa: npt.NDArray[np.uint8], response: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Eq. (2): ``S_jk = sum_t y_t [level of factor j in combination t is k]`` for ``k = 1, 2``."""
    r = np.asarray(response, dtype=float)
    return ((oa == 1) * r[:, None]).sum(0), ((oa == 2) * r[:, None]).sum(0)


def igc(
    p1: Genome,
    y1: Objectives,
    p2: Genome,
    y2: Objectives,
    problem: Problem,
    evaluator: Evaluator,
    rng: np.random.Generator,
    *,
    multi_objective: bool = False,
    max_segments: int | None = None,
    step10: bool = True,
) -> IGCResult:
    """Recombine two evaluated parents with one IGC operation.

    Children are always really evaluated; main effects only choose them. Nothing
    is evaluated unless the whole OA plus both children fit the remaining budget.

    Args:
        p1: First parent (level 1 of every factor; also OA row 1).
        y1: Objective vector of ``p1``.
        p2: Second parent (level 2).
        y2: Objective vector of ``p2``.
        problem: Supplies the division into gene segments.
        evaluator: Performs and accounts every objective call.
        rng: Random generator (division cut points).
        multi_objective: Use GPSIFF over the OA combinations as the response.
        max_segments: Cap on the number of segments (``bounded_segments`` variant).
        step10: Single-objective elitist Step 10: return the best two of the OA
            combinations, C1 and C2.
    """
    parents = [(p1, y1), (p2, y2)]
    diff = np.flatnonzero(p1 != p2)  # Step 1
    if len(diff) == 0:
        return IGCResult("identical", parents)
    segments = problem.divide(p1, p2, diff, max_segments, rng)  # Step 2
    if len(segments) < 2:
        return IGCResult("not_divisible", parents, trace={"M": len(diff)})
    oa = generate_oa(len(segments))  # Step 3
    rows = [decode(p1, p2, segments, r) for r in oa]  # Step 4
    trace: dict[str, Any] = {
        "M": len(diff),
        "N": len(segments),
        "n_rows": len(oa),
        "calls_before": evaluator.counters["objective_calls"],
    }
    if not evaluator.can_afford(evaluator.n_uncached(rows) + 2):
        return IGCResult("budget", parents, trace=trace)

    res = evaluator.evaluate_batch(rows, phase="oa")  # Step 5
    byproducts = [
        (g, r.objectives) for g, r in zip(rows, res, strict=True) if r is not None and r.objectives is not None
    ]
    if len(byproducts) < len(rows):
        # a failed or invalid combination has no response: never put inf into main effects
        trace["calls_after"] = evaluator.counters["objective_calls"]
        return IGCResult("failed_row", parents, byproducts, trace)

    Y = np.array([y for _, y in byproducts], dtype=float)
    # lower is better: the objective itself, or minus GPSIFF over this IGC's combinations
    response = -gpsiff(Y).astype(float) if multi_objective else Y[:, 0]
    s1, s2 = main_effects(oa, response)  # Step 6
    best = np.where(s1 <= s2, 1, 2)  # Step 7; a tie picks level 1 (engineering choice)
    j = int(np.argmin(np.abs(s1 - s2)))  # smallest MED; first index on ties
    alt = best.copy()
    alt[j] = 3 - alt[j]
    c1, c2 = decode(p1, p2, segments, best), decode(p1, p2, segments, alt)  # Steps 8, 9
    rc1, rc2 = evaluator.evaluate_batch([c1, c2], phase="child")
    trace.update(calls_after=evaluator.counters["objective_calls"], med_factor=j)
    if rc1 is None or rc2 is None or rc1.objectives is None or rc2.objectives is None:
        return IGCResult("failed_row", parents, byproducts, trace)
    children = [(c1, rc1.objectives), (c2, rc2.objectives)]

    if not multi_objective:
        best_row = float(Y[:, 0].min())
        trace.update(
            best_row=best_row,
            c1=rc1.objectives[0],
            c2=rc2.objectives[0],
            child_beats_rows=min(rc1.objectives[0], rc2.objectives[0]) < best_row,
        )
        if step10:  # Step 10: best two of the n combinations (row 1 = P1), C1 and C2
            pool: list[Individual] = []
            seen: set[bytes] = set()
            for g, y in byproducts + children:
                if g.tobytes() not in seen:
                    seen.add(g.tobytes())
                    pool.append((g, y))
            order = sorted(range(len(pool)), key=lambda i: pool[i][1][0])  # stable: ties keep order
            children = [pool[i] for i in order[:2]]
    return IGCResult("applied", children, byproducts, trace)
