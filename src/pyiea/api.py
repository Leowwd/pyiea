"""One-call entry point for the common case."""

from __future__ import annotations

from typing import Literal, overload

from .evaluator import Evaluator, Objective
from .iea import IEA, IEAConfig, IEAResult
from .imoea import IMOEA, IMOEAConfig, IMOEAResult
from .problem import BinaryProblem, Problem

__all__ = ["optimize"]


@overload
def optimize(
    fitness: Objective,
    n_bits: int | None = ...,
    *,
    mode: Literal["single_objective"] = ...,
    n_objectives: int = ...,
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
    fitness: Objective,
    n_bits: int | None = ...,
    *,
    mode: Literal["multi_objective"],
    n_objectives: int = ...,
    problem: Problem | None = ...,
    config: IMOEAConfig | None = ...,
    max_calls: int | None = ...,
    max_seconds: float | None = ...,
    seed: int = ...,
    workers: int = ...,
    context: str | None = ...,
) -> IMOEAResult: ...


def optimize(
    fitness: Objective,
    n_bits: int | None = None,
    *,
    mode: Literal["single_objective", "multi_objective"] = "single_objective",
    n_objectives: int = 2,
    problem: Problem | None = None,
    config: IEAConfig | IMOEAConfig | None = None,
    max_calls: int | None = 1000,
    max_seconds: float | None = None,
    seed: int = 0,
    workers: int = 1,
    context: str | None = None,
) -> IEAResult | IMOEAResult:
    """Minimize ``fitness`` over bit strings with IEA or IMOEA.

    Args:
        fitness: ``f(genome) -> float`` (single objective) or a sequence of floats
            (multi-objective). Everything is minimized; negate what you maximize.
            NaN, inf or an exception marks a failed evaluation.
        n_bits: Genome length for the default unconstrained :class:`BinaryProblem`.
        mode: ``"single_objective"`` (IEA) or ``"multi_objective"`` (IMOEA).
        n_objectives: Objective count in multi-objective mode.
        problem: A custom search space instead of ``BinaryProblem(n_bits)``.
        config: :class:`IEAConfig` or :class:`IMOEAConfig`; defaults are the paper's.
        max_calls: Objective-call budget (``None`` for unlimited; then set another stop).
        max_seconds: Wall-time budget.
        seed: Random seed.
        workers: Worker processes (``fitness`` must then be picklable).
        context: Evaluator context; defaults to ``problem.version``.

    Returns:
        :class:`IEAResult` or :class:`IMOEAResult`.

    Example:
        >>> import numpy as np, pyiea
        >>> w = np.arange(8) - 3.0
        >>> res = pyiea.optimize(lambda g: float(w @ g), n_bits=8, max_calls=300)
        >>> res.best
        -6.0
    """
    if problem is None:
        if n_bits is None:
            raise ValueError("give n_bits or a problem")
        problem = BinaryProblem(n_bits)
    if max_calls is None and max_seconds is None and getattr(config, "max_generations", None) is None:
        raise ValueError("set max_calls, max_seconds or config.max_generations so the run can stop")
    multi = mode == "multi_objective"
    if mode not in ("single_objective", "multi_objective"):
        raise ValueError(f"unknown mode {mode!r}")
    ev = Evaluator(
        fitness,
        n_objectives if multi else 1,
        context or problem.version,
        is_valid=problem.is_valid,
        max_calls=max_calls,
        max_seconds=max_seconds,
        workers=workers,
    )
    if multi:
        if config is not None and not isinstance(config, IMOEAConfig):
            raise TypeError("multi_objective mode needs an IMOEAConfig")
        return IMOEA(problem, ev, config, seed).optimize()
    if config is not None and not isinstance(config, IEAConfig):
        raise TypeError("single_objective mode needs an IEAConfig")
    return IEA(problem, ev, config, seed).optimize()
