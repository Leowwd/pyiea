"""Objective evaluation with caching, budgets, a ledger and an optional process pool.

Every objective is **minimized**. An objective is a callable
``f(genome) -> float | Sequence[float]``; pass maximization objectives negated.
Returning NaN/inf or raising marks the candidate as a failed evaluation, which is
never treated as a good fitness. An objective may define an integer attribute
``tokens_per_call`` to have tokens counted.

One *objective call* is one real call of ``f`` on a genome that is not cached.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import os
import pickle
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

from .exceptions import BudgetExhaustedError, CheckpointError, EvaluationError
from .problem import Genome

__all__ = ["COUNTERS", "EvalResult", "Evaluator", "Objective", "genome_hash", "genome_key"]

Objective: TypeAlias = Callable[[Genome], float | Sequence[float]]
Objectives: TypeAlias = tuple[float, ...]

COUNTERS = (
    "proposals",
    "unique_candidates",
    "objective_calls",
    "cache_hits",
    "invalid_candidates",
    "failed_evaluations",
    "evaluated_tokens",
)
"""Accounting counters kept by :class:`Evaluator`."""


@dataclass(frozen=True)
class EvalResult:
    """Outcome of evaluating one genome.

    Attributes:
        objectives: Objective vector, or ``None`` unless ``status == "ok"``.
        status: ``"ok"``, ``"failed"`` (exception or non-finite value) or ``"invalid"``.
        seconds: Wall time of the objective call.
    """

    objectives: Objectives | None
    status: str
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _call(objective: Objective, genome: Genome) -> tuple[Any, str, float]:
    t0 = time.perf_counter()
    try:
        raw = objective(genome)
        y = tuple(float(v) for v in np.atleast_1d(np.asarray(raw, dtype=float)))  # scalar, list or array
        status = "ok" if all(math.isfinite(v) for v in y) else "failed"
        out: Any = y
    except Exception as e:  # noqa: BLE001 - any objective error is an evaluation failure
        out, status = repr(e), "failed"
    return out, status, time.perf_counter() - t0


_WORKER_OBJECTIVE: Objective | None = None


def _worker_init(objective: Objective) -> None:
    global _WORKER_OBJECTIVE
    _WORKER_OBJECTIVE = objective


def _worker_call(genome: Genome) -> tuple[Any, str, float]:
    assert _WORKER_OBJECTIVE is not None
    return _call(_WORKER_OBJECTIVE, genome)


def _objective_count_message(got: int, expected: int) -> str:
    msg = f"the objective returned {got} value{'s' if got != 1 else ''} for a genome, but n_objectives={expected}."
    if expected == 1:
        return msg + f" For {got} objectives use IMOEA, e.g. pyiea.optimize(..., n_objectives={got})."
    if got == 1:
        return msg + " Return one value per objective, e.g. `return distortion, size`."
    return msg + f" Pass n_objectives={got}, or return {expected} values."


def genome_key(g: Genome) -> bytes:
    """Cache key of a genome."""
    return g.tobytes()


def genome_hash(g: Genome) -> str:
    """Short, stable identifier of a genome for ledgers and logs."""
    return hashlib.sha1(g.tobytes(), usedforsecurity=False).hexdigest()[:16]


class Evaluator:
    """The single place where objective calls happen.

    Args:
        objective: The fitness function (see module docstring). With ``workers > 1``
            it must be picklable (a module-level function or class instance).
        n_objectives: Length of the objective vector.
        context: Identifies everything the objective depends on (problem version,
            data, model revision, settings). The cache is valid only within one
            context, and checkpoints refuse to load into a different one.
        is_valid: Optional legality check; invalid genomes never reach the objective.
        max_calls: Objective-call budget (``None`` for unlimited).
        max_seconds: Wall-time budget in seconds (``None`` for unlimited).
        workers: Worker processes for batch evaluation; ``1`` evaluates serially.
    """

    def __init__(
        self,
        objective: Objective,
        n_objectives: int,
        context: str,
        is_valid: Callable[[Genome], bool] | None = None,
        max_calls: int | None = None,
        max_seconds: float | None = None,
        workers: int = 1,
    ) -> None:
        if n_objectives < 1:
            raise ValueError(f"n_objectives must be >= 1, got {n_objectives}")
        if max_calls is not None and max_calls < 0:
            raise ValueError(f"max_calls must be >= 0, got {max_calls}")
        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        self.objective = objective
        self.n_objectives = n_objectives
        self.context = context
        self.is_valid = is_valid
        self.max_calls = max_calls
        self.max_seconds = max_seconds
        self.workers = workers
        self.cache: dict[bytes, EvalResult] = {}
        self.counters: dict[str, int] = dict.fromkeys(COUNTERS, 0)
        self.ledger: list[dict[str, Any]] = []
        self._elapsed_before = 0.0
        self._t0 = time.perf_counter()
        self._pool: ProcessPoolExecutor | None = None
        self._first_failure: str | None = None  # why the earliest failed or invalid candidate failed

    def __enter__(self) -> Evaluator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- budget -----------------------------------------------------------------
    def elapsed(self) -> float:
        """Wall time since creation, including time before a checkpoint resume."""
        return self._elapsed_before + time.perf_counter() - self._t0

    def remaining_calls(self) -> float:
        """Objective calls left (``inf`` without a call budget)."""
        return math.inf if self.max_calls is None else self.max_calls - self.counters["objective_calls"]

    def time_up(self) -> bool:
        return self.max_seconds is not None and self.elapsed() >= self.max_seconds

    def n_uncached(self, genomes: Sequence[Genome]) -> int:
        """Objective calls that evaluating ``genomes`` would need."""
        keys = {genome_key(g) for g in genomes if self.is_valid is None or self.is_valid(g)}
        return sum(k not in self.cache for k in keys)

    def can_afford(self, n_calls: int) -> bool:
        return n_calls <= self.remaining_calls() and not self.time_up()

    # --- evaluation ---------------------------------------------------------------
    def evaluate_batch(
        self, genomes: Sequence[Genome], phase: str = "", partial_ok: bool = False
    ) -> list[EvalResult | None]:
        """Evaluate ``genomes``; results are aligned with the input order.

        Duplicates within a batch cost one call. If the uncached calls exceed the
        remaining budget, ``partial_ok=True`` evaluates the first affordable ones
        and returns ``None`` for the rest; otherwise :class:`BudgetExhaustedError` is
        raised before any call is made.

        Args:
            genomes: Candidates to evaluate.
            phase: Label stored in the ledger (for example ``"oa"`` or ``"child"``).
            partial_ok: Allow a partially evaluated batch.
        """
        self.counters["proposals"] += len(genomes)
        keys = [genome_key(g) for g in genomes]
        todo: list[tuple[bytes, Genome]] = []
        todo_keys: set[bytes] = set()
        new: set[bytes] = set()
        for g, k in zip(genomes, keys, strict=True):
            if k in self.cache or k in todo_keys:
                continue
            if self.is_valid is not None and not self.is_valid(g):
                self._first_failure = self._first_failure or "an invalid genome (rejected by is_valid)"
                self.cache[k] = EvalResult(None, "invalid")
                self.counters["invalid_candidates"] += 1
                self.counters["unique_candidates"] += 1
                new.add(k)
                continue
            todo.append((k, g))
            todo_keys.add(k)
        room = 0.0 if self.time_up() else self.remaining_calls()
        if len(todo) > room:
            if not partial_ok:
                raise BudgetExhaustedError(f"batch needs {len(todo)} objective calls, {room} left")
            todo = todo[: int(room)]
        for (k, g), (y, status, sec) in zip(todo, self._run([g for _, g in todo]), strict=True):
            if isinstance(y, tuple) and len(y) != self.n_objectives:
                raise ValueError(_objective_count_message(len(y), self.n_objectives))
            ok = status == "ok"
            if not ok and self._first_failure is None:
                self._first_failure = f"{y}" if isinstance(y, str) else f"a non-finite value {y}"
            self.cache[k] = EvalResult(y if ok else None, status, sec)
            self.counters["objective_calls"] += 1
            self.counters["unique_candidates"] += 1
            self.counters["failed_evaluations"] += not ok
            self.counters["evaluated_tokens"] += int(getattr(self.objective, "tokens_per_call", 0))
            self.ledger.append(
                {
                    "call": self.counters["objective_calls"],
                    "phase": phase,
                    "genome": genome_hash(g),
                    "objectives": list(y) if ok else None,
                    "status": status,
                    "error": None if ok else y,
                    "seconds": sec,
                    "elapsed": self.elapsed(),
                }
            )
        new |= {k for k, _ in todo}
        out: list[EvalResult | None] = []
        for k in keys:
            r = self.cache.get(k)
            if r is not None and k not in new:
                self.counters["cache_hits"] += 1
            new.discard(k)  # a later duplicate in the same batch is a hit
            out.append(r)
        return out

    def raise_if_all_failed(self, n_genomes: int) -> None:
        """Raise :class:`~pyiea.EvaluationError` if any candidate failed; call it when none succeeded.

        IEA and IMOEA call this when no genome of the initial population has a
        valid objective. Without any failure (for example a zero budget) it does nothing.
        """
        if self._first_failure is None:
            return
        reason = self._first_failure
        hint = (
            " Genomes are read-only; work on a copy (g.copy()) if the objective needs to change one."
            if "read-only" in reason
            else ""
        )
        raise EvaluationError(
            f"the objective failed on all {n_genomes} genomes of the initial population; "
            f"the first failure was {reason}.{hint}"
        )

    def _run(self, genomes: list[Genome]) -> list[tuple[Any, str, float]]:
        if self.workers <= 1 or len(genomes) <= 1:
            return [_call(self.objective, g) for g in genomes]
        if self._pool is None:
            try:
                pickle.dumps(self.objective)
            except Exception as e:
                raise TypeError(
                    "workers > 1 needs a picklable objective (a module-level function or class instance, "
                    f"not a lambda or closure): {e}"
                ) from e
            # one native thread per worker; the parent's already-loaded BLAS is unaffected
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            # forkserver/spawn: never fork a process that may hold CUDA or BLAS threads
            ctx = mp.get_context("forkserver" if os.name == "posix" else "spawn")
            self._pool = ProcessPoolExecutor(
                self.workers, mp_context=ctx, initializer=_worker_init, initargs=(self.objective,)
            )
        return list(self._pool.map(_worker_call, genomes))

    def close(self) -> None:
        """Shut down the worker pool, if any. Safe to call more than once."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    # --- checkpoint ---------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "cache": dict(self.cache),
            "counters": dict(self.counters),
            "ledger": list(self.ledger),
            "elapsed": self.elapsed(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore cache, counters, ledger and elapsed time.

        Raises:
            CheckpointError: if the checkpoint was written under another context.
        """
        if state["context"] != self.context:
            raise CheckpointError(f"checkpoint context {state['context']!r} != evaluator context {self.context!r}")
        self.cache = dict(state["cache"])
        self.counters = dict(state["counters"])
        self.ledger = list(state["ledger"])
        self._elapsed_before, self._t0 = state["elapsed"], time.perf_counter()

    def summary(self) -> dict[str, float]:
        """Counters plus total wall time and summed objective time."""
        return {
            **self.counters,
            "wall_seconds": self.elapsed(),
            "objective_seconds": sum(e["seconds"] for e in self.ledger),
        }
