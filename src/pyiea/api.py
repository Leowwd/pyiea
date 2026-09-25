"""One-call entry point for the common case."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, Literal, overload

from .encoding import RealEncoder
from .evaluator import Evaluator, Objective
from .iea import IEA, IEAConfig, IEAResult
from .imoea import IMOEA, IMOEAConfig, IMOEAResult
from .problem import BinaryProblem, Problem

__all__ = ["optimize"]

Mode = Literal["single_objective", "multi_objective"]
Fitness = Callable[[Any], float | Sequence[float]]
"""``f(genome)`` for bit strings, ``f(x)`` (a float array) with ``bounds``."""


@overload
def optimize(
    fitness: Fitness,
    n_bits: int | None = ...,
    *,
    bounds: Sequence[tuple[float, float]] | None = ...,
    bits: int = ...,
    gray: bool = ...,
    mode: Literal["single_objective"] | None = ...,
    n_objectives: Literal[1] | None = ...,
    problem: Problem | None = ...,
    config: IEAConfig | None = ...,
    max_calls: int | None = ...,
    max_seconds: float | None = ...,
    seed: int = ...,
    workers: int = ...,
    context: str | None = ...,
) -> IEAResult: ...


@overload
def optimize(
    fitness: Fitness,
    n_bits: int | None = ...,
    *,
    bounds: Sequence[tuple[float, float]] | None = ...,
    bits: int = ...,
    gray: bool = ...,
    mode: Mode | None = ...,
    n_objectives: int | None = ...,
    problem: Problem | None = ...,
    config: IMOEAConfig | None = ...,
    max_calls: int | None = ...,
    max_seconds: float | None = ...,
    seed: int = ...,
    workers: int = ...,
    context: str | None = ...,
) -> IMOEAResult: ...


def optimize(
    fitness: Fitness,
    n_bits: int | None = None,
    *,
    bounds: Sequence[tuple[float, float]] | None = None,
    bits: int = 10,
    gray: bool = True,
    mode: Mode | None = None,
    n_objectives: int | None = None,
    problem: Problem | None = None,
    config: IEAConfig | IMOEAConfig | None = None,
    max_calls: int | None = 1000,
    max_seconds: float | None = None,
    seed: int = 0,
    workers: int = 1,
    context: str | None = None,
) -> IEAResult | IMOEAResult:
    """Minimize ``fitness`` with IEA (one objective) or IMOEA (several objectives).

    Describe the search space in one of three ways:

    * ``n_bits=n``: ``fitness`` receives a read-only 0/1 ``uint8`` array of length ``n``.
    * ``bounds=[(low, high), ...]``: ``fitness`` receives a float array with one value
      per parameter. Each parameter is encoded with ``bits`` bits (Gray code by
      default), and the result's ``best_x`` / ``front_x`` hold the decoded values.
    * ``problem=...``: a custom search space, such as :class:`FixedCardinalityProblem`.

    One objective runs IEA and returns :class:`IEAResult`. With ``n_objectives=k``
    (``k >= 2``), ``fitness`` returns ``k`` values and IMOEA returns an
    :class:`IMOEAResult` with a Pareto front. Everything is minimized, so negate
    what you want to maximize. NaN, inf or an exception marks a failed evaluation.

    Args:
        fitness: The objective, ``f(genome)``, or ``f(x)`` with ``bounds``.
        n_bits: Genome length of an unconstrained bit string.
        bounds: ``(low, high)`` of every real parameter.
        bits: Bits per real parameter (with ``bounds``).
        gray: Gray-code the real parameters (with ``bounds``); see ``docs/paper_map.md``.
        mode: ``"single_objective"`` or ``"multi_objective"``. Usually unnecessary,
            because it follows from ``n_objectives``.
        n_objectives: Number of values ``fitness`` returns (default 1, or 2 with
            ``mode="multi_objective"`` or an :class:`IMOEAConfig`).
        problem: A custom search space instead of ``n_bits`` or ``bounds``.
        config: :class:`IEAConfig` or :class:`IMOEAConfig`; defaults are the paper's.
        max_calls: Objective-call budget (``None`` for unlimited; then set another stop).
        max_seconds: Wall-time budget.
        seed: Random seed; the same seed reproduces the run.
        workers: Worker processes (``fitness`` must then be picklable).
        context: Evaluator context; defaults to ``problem.version``.

    Returns:
        :class:`IEAResult` or :class:`IMOEAResult`.

    Raises:
        EvaluationError: if ``fitness`` fails on every genome of the initial population.

    Example:
        >>> import numpy as np, pyiea
        >>> w = np.arange(8) - 3.0
        >>> pyiea.optimize(lambda g: float(w @ g), n_bits=8, max_calls=300).best
        -6.0
        >>> res = pyiea.optimize(lambda x: float(((x - 1) ** 2).sum()), bounds=[(-4, 4)] * 2, max_calls=2000)
        >>> np.round(res.best_x, 1).tolist()
        [1.0, 1.0]
    """
    problem, fitness, encoder = _search_space(fitness, n_bits, bounds, bits, gray, problem)
    if max_calls is None and max_seconds is None and getattr(config, "max_generations", None) is None:
        raise ValueError("set max_calls, max_seconds or config.max_generations so the run can stop")
    multi, k = _resolve_mode(mode, n_objectives, config)
    ev = Evaluator(
        fitness,
        k,
        context or problem.version,
        is_valid=problem.is_valid,
        max_calls=max_calls,
        max_seconds=max_seconds,
        workers=workers,
    )
    if multi:
        if config is not None and not isinstance(config, IMOEAConfig):
            raise TypeError(f"{k} objectives run IMOEA, which needs an IMOEAConfig, not {type(config).__name__}")
        return dataclasses.replace(IMOEA(problem, ev, config, seed).optimize(), encoder=encoder)
    if config is not None and not isinstance(config, IEAConfig):
        raise TypeError(f"one objective runs IEA, which needs an IEAConfig, not {type(config).__name__}")
    return dataclasses.replace(IEA(problem, ev, config, seed).optimize(), encoder=encoder)


def _search_space(
    fitness: Fitness,
    n_bits: int | None,
    bounds: Sequence[tuple[float, float]] | None,
    bits: int,
    gray: bool,
    problem: Problem | None,
) -> tuple[Problem, Objective, RealEncoder | None]:
    """The problem, the objective on genomes and, for real parameters, the encoder."""
    if bounds is not None:
        if n_bits is not None or problem is not None:
            raise ValueError("give only one of n_bits, bounds and problem")
        encoder = RealEncoder(bounds, bits=bits, gray=gray)
        return encoder.problem(), encoder.wrap(fitness), encoder
    if problem is not None:
        if n_bits is not None and n_bits != problem.n_bits:
            raise ValueError(f"n_bits={n_bits} does not match problem.n_bits={problem.n_bits}")
        return problem, fitness, None
    if n_bits is None:
        raise ValueError("describe the search space: n_bits=..., bounds=[(low, high), ...] or problem=...")
    return BinaryProblem(n_bits), fitness, None


def _resolve_mode(mode: str | None, n_objectives: int | None, config: object) -> tuple[bool, int]:
    """``(multi_objective, n_objectives)`` from whichever of mode, n_objectives and config the caller gave."""
    if mode not in (None, "single_objective", "multi_objective"):
        raise ValueError(f"unknown mode {mode!r}; use 'single_objective' or 'multi_objective' (or just n_objectives)")
    if n_objectives is not None and n_objectives < 1:
        raise ValueError(f"n_objectives must be >= 1, got {n_objectives}")
    if mode == "single_objective":
        if n_objectives not in (None, 1):
            raise ValueError(f"mode='single_objective' means one objective, but n_objectives={n_objectives}")
        return False, 1
    if mode == "multi_objective" or (mode is None and n_objectives is None and isinstance(config, IMOEAConfig)):
        k = 2 if n_objectives is None else n_objectives
        if k < 2:
            raise ValueError("multi-objective optimization needs n_objectives >= 2")
        return True, k
    k = 1 if n_objectives is None else n_objectives
    return k > 1, k
