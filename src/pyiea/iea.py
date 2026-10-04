"""Single-objective IEA (Ho, Shu, Chen 2004, Section IV-C, Steps 1-6)."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from .checkpoint import algorithm_config, load_checkpoint, save_checkpoint
from .encoding import RealEncoder
from .evaluator import Evaluator, Objectives
from .exceptions import CheckpointError
from .igc import igc
from .problem import Genome, Problem

__all__ = ["IEA", "IEAConfig", "IEAResult"]

logger = logging.getLogger(__name__)


def _count(p: float, n: int) -> int:
    """``floor(p * n)`` that is not fooled by float products such as ``0.29 * 100 == 28.999999999999996``."""
    return math.floor(p * n + 1e-9)


def _check_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass
class IEAConfig:
    """IEA parameters. Defaults are the paper's Section V settings (N_pop = 30 as in Table III).

    Attributes:
        pop_size: Population size ``N_pop`` (>= 2).
        ps: Selection probability.
        pc: Recombination probability.
        pm: Mutation probability passed to ``Problem.mutate``.
        max_segments: ``None`` uses the paper's division size; an integer caps it
            (the ``bounded_segments`` variant, which must be reported as such).
        step10: Use IGC's optional elitist Step 10.
        step10_p2: Step 10 chooses the best two of the OA combinations, C1, C2 **and P2**, as in the paper.
            ``False`` restores the pre-0.3 behavior without P2 (an IEA-based variant); it exists to reproduce
            earlier results.
        igc: ``False`` skips Step 4 (the intelligent gene collector): selection, elitism and mutation are
            unchanged but no parents are recombined. This is an ablation, an IEA-based variant that must not
            be reported as IEA. ``pc`` is still validated and is then unused.
        max_generations: Stop after this many generations.
        target: Stop once the best objective is ``<= target``.
        max_stall_generations: Stop after this many generations without a new
            objective call (``0`` disables); prevents looping over a fully cached space.
        checkpoint_every: Write a checkpoint every this many generations (``0`` never).
    """

    pop_size: int = 30
    ps: float = 0.2
    pc: float = 0.8
    pm: float = 0.05
    max_segments: int | None = None
    step10: bool = True
    step10_p2: bool = True
    igc: bool = True
    max_generations: int | None = None
    target: float | None = None
    max_stall_generations: int = 50
    checkpoint_every: int = 0

    def __post_init__(self) -> None:
        if self.pop_size < 2:
            raise ValueError(f"pop_size must be >= 2, got {self.pop_size}")
        for name in ("ps", "pc", "pm"):
            _check_probability(name, getattr(self, name))
        if _count(self.pc, self.pop_size) < 2:
            raise ValueError("pc * pop_size must select at least one pair of parents")
        if self.max_segments is not None and self.max_segments < 2:
            raise ValueError(f"max_segments must be >= 2 or None, got {self.max_segments}")


@dataclass(frozen=True)
class IEAResult:
    """What :meth:`IEA.optimize` returns.

    Attributes:
        best_genome: Best genome among every evaluated candidate (OA combinations included).
        best: Its objective value.
        generations: Completed generations.
        stop_reason: ``budget_exhausted``, ``time_limit``, ``target_reached``,
            ``max_generations`` or ``stalled``.
        history: Best-so-far improvements with call counts and elapsed seconds.
        pop_best: Best objective in the population after each evaluation step.
        igc_log: One trace per IGC operation.
        accounting: Evaluator counters, wall time and summed objective time.
        encoder: The :class:`RealEncoder` of a run over real parameters (``optimize(bounds=...)``).
    """

    best_genome: Genome | None
    best: float | None
    generations: int
    stop_reason: str
    history: list[dict[str, Any]] = field(repr=False)
    pop_best: list[float] = field(repr=False)
    igc_log: list[dict[str, Any]] = field(repr=False)
    accounting: dict[str, float] = field(repr=False)
    encoder: RealEncoder | None = field(default=None, repr=False)

    @property
    def best_x(self) -> npt.NDArray[np.float64] | None:
        """Real parameters of ``best_genome`` for a run over real parameters, else ``None``."""
        if self.encoder is None or self.best_genome is None:
            return None
        return self.encoder.decode(self.best_genome)

    def __repr__(self) -> str:
        found = "no valid evaluation" if self.best is None else f"best={self.best:.6g}"
        where = f", best_x={np.array2string(self.best_x, precision=4, threshold=8)}" if self.best_x is not None else ""
        calls = int(self.accounting.get("objective_calls", 0))
        return (
            f"IEAResult({found}{where}, stop_reason={self.stop_reason!r}, "
            f"generations={self.generations}, objective_calls={calls})"
        )


class IEA:
    """Intelligent evolutionary algorithm with IGC recombination.

    Args:
        problem: Search space (see :class:`pyiea.Problem`).
        evaluator: Objective, budget and accounting; ``n_objectives`` must be 1.
        config: Algorithm parameters.
        seed: Seed of the run's random generator.
    """

    def __init__(self, problem: Problem, evaluator: Evaluator, config: IEAConfig | None = None, seed: int = 0) -> None:
        if evaluator.n_objectives != 1:
            raise ValueError("IEA is single-objective; use IMOEA for n_objectives > 1")
        self.problem, self.evaluator, self.cfg, self.seed = problem, evaluator, config or IEAConfig(), seed
        self.rng = np.random.default_rng(seed)
        self.gen = 0
        self.pop: list[Genome] = []
        self.pop_y: list[Objectives | None] = []  # None: not evaluated yet, or failed
        self.best: tuple[Genome, Objectives] | None = None
        self.history: list[dict[str, Any]] = []
        self.igc_log: list[dict[str, Any]] = []
        self.pop_best: list[float] = []
        self._calls_log: list[int] = []

    # --- helpers ------------------------------------------------------------------
    def _observe(self, items: list[tuple[Genome, Objectives | None]]) -> None:
        for g, y in items:
            if y is not None and (self.best is None or y[0] < self.best[1][0]):
                self.best = (g, y)
                self.history.append(
                    {
                        "generation": self.gen,
                        "calls": self.evaluator.counters["objective_calls"],
                        "seconds": self.evaluator.elapsed(),
                        "best": y[0],
                    }
                )

    def _key(self, i: int) -> float:
        y = self.pop_y[i]  # failed/unevaluated individuals rank last but never enter main effects
        return math.inf if y is None else y[0]

    def evaluate_population(self) -> bool:
        """Evaluate every individual whose objective is not known (``pop_y`` entry ``None``) and update ``best``.

        Returns ``False`` if the budget stopped the batch before every individual was evaluated. A caller that
        drives the loop with :meth:`step` can set ``pop_y`` entries to ``None`` and call this to re-score the
        population under another objective context.
        """
        return self._evaluate_population()

    def _evaluate_population(self) -> bool:
        idx = [i for i, y in enumerate(self.pop_y) if y is None]
        res = self.evaluator.evaluate_batch(
            [self.pop[i] for i in idx], phase="init" if self.gen == 0 else "mutation", partial_ok=True
        )
        for i, r in zip(idx, res, strict=True):
            self.pop_y[i] = r.objectives if r is not None else None
        self._observe([(self.pop[i], self.pop_y[i]) for i in idx])
        return all(r is not None for r in res)

    # --- checkpointing --------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "IEA",
            "config": asdict(self.cfg),
            "seed": self.seed,
            "gen": self.gen,
            "rng": self.rng.bit_generator.state,
            "pop": self.pop,
            "pop_y": self.pop_y,
            "best": self.best,
            "history": self.history,
            "igc_log": self.igc_log,
            "pop_best": self.pop_best,
            "calls_log": self._calls_log,
            "evaluator": self.evaluator.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a run. Stopping conditions may differ from the checkpoint (to extend a run).

        Raises:
            CheckpointError: if the kind, seed, algorithm parameters or evaluator context differ.
        """
        if (
            state.get("kind") != "IEA"
            or state["seed"] != self.seed
            or algorithm_config(state["config"]) != algorithm_config(asdict(self.cfg))
        ):
            raise CheckpointError("checkpoint does not match this IEA configuration and seed")
        self.evaluator.load_state_dict(state["evaluator"])
        self.gen, self.pop, self.pop_y, self.best = state["gen"], state["pop"], state["pop_y"], state["best"]
        self.rng.bit_generator.state = state["rng"]
        self.history, self.igc_log = state["history"], state["igc_log"]
        self.pop_best, self._calls_log = state["pop_best"], state["calls_log"]

    # --- main loop ------------------------------------------------------------------
    def optimize(self, checkpoint_path: str | os.PathLike[str] | None = None, resume: bool = False) -> IEAResult:
        """Run until a stopping condition holds.

        Args:
            checkpoint_path: Where to write checkpoints (``config.checkpoint_every``).
            resume: Continue from ``checkpoint_path`` instead of starting fresh; the
                result equals that of an uninterrupted run with the same seed.
        """
        cfg, P = self.cfg, self.problem
        if resume:
            if checkpoint_path is None:
                raise ValueError("resume=True needs checkpoint_path")
            self.load_state_dict(load_checkpoint(checkpoint_path))
        elif not self.pop:
            self.pop = [P.random_genome(self.rng) for _ in range(cfg.pop_size)]  # Step 1
            self.pop_y = [None] * cfg.pop_size
        reason: str | None = None

        try:
            while True:
                if cfg.checkpoint_every and checkpoint_path and self.gen % cfg.checkpoint_every == 0:
                    save_checkpoint(checkpoint_path, self.state_dict())
                complete = self._evaluate_population()  # Step 2
                if self.gen == 0 and self.best is None:
                    self.evaluator.raise_if_all_failed(len(self.pop))  # fail fast instead of burning the budget
                if len(self.pop_best) <= self.gen:  # one entry per generation, also when optimize() is called again
                    self.pop_best.append(min(map(self._key, range(cfg.pop_size))))
                    self._calls_log.append(self.evaluator.counters["objective_calls"])
                logger.debug(
                    "IEA gen %d: best %s, calls %d",
                    self.gen,
                    self.best and self.best[1][0],
                    self.evaluator.counters["objective_calls"],
                )
                reason = self._stop_reason(complete)  # Step 6, tested right after evaluation
                if reason:
                    break
                reason = self.step()  # Steps 3-5
                if reason:
                    break
        finally:
            self.evaluator.close()

        assert reason is not None
        return IEAResult(
            best_genome=self.best[0] if self.best else None,
            best=self.best[1][0] if self.best else None,
            generations=self.gen,
            stop_reason=reason,
            history=self.history,
            pop_best=self.pop_best,
            igc_log=self.igc_log,
            accounting=self.evaluator.summary(),
        )

    def step(self) -> str | None:
        """Run Steps 3-5 once: truncation selection, parent pairing with IGC, mutation; then ``gen += 1``.

        The population must be fully evaluated (:meth:`evaluate_population`) before the call, and the mutated
        individuals are left unevaluated (``pop_y`` entry ``None``); :meth:`optimize` evaluates them at the top of
        its next iteration. Returns ``"budget_exhausted"`` or ``"time_limit"`` if an IGC does not fit the budget
        (the generation then stops before mutation and ``gen`` is not advanced), otherwise ``None``. In that
        stopped case the population has already been truncated and may hold the children of earlier IGCs of the
        generation, so a caller that keeps stepping after raising the budget starts Step 3 from that state.

        A caller that drives IEA with ``step`` may change the objective context between generations: set the
        ``pop_y`` entries to ``None`` and call :meth:`evaluate_population` under the new context. ``best`` and
        ``history`` then mix contexts, so such a caller must keep its own bookkeeping.
        """
        cfg, P = self.cfg, self.problem
        n_replace = min(_count(cfg.ps, cfg.pop_size), cfg.pop_size - 1)  # at least one individual survives
        n_parents = _count(cfg.pc, cfg.pop_size) // 2 * 2

        # Step 3: truncation selection. The paper leaves open how N_pop is restored;
        # engineering choice: the worst ps*Npop are replaced by copies of the best individuals, cycling through the
        # survivors (for ps <= 0.5 these are the best ps*Npop, as before; for ps > 0.5 never an individual that
        # truncation dropped)
        order = sorted(range(cfg.pop_size), key=self._key)
        keep = cfg.pop_size - n_replace
        idx = order[:keep] + [order[i % keep] for i in range(n_replace)]
        self.pop, self.pop_y = [self.pop[i] for i in idx], [self.pop_y[i] for i in idx]  # [0] = I_best

        # Step 4: pc*Npop parents including I_best, which is parent 1 of the first pair
        # (skipped in the igc=False ablation, which then draws no parents either)
        chosen = (
            [0, *self.rng.choice(np.arange(1, cfg.pop_size), n_parents - 1, replace=False).tolist()] if cfg.igc else []
        )
        for a, b in zip(chosen[0::2], chosen[1::2], strict=True):
            ya, yb = self.pop_y[a], self.pop_y[b]
            if ya is None or yb is None:
                continue  # never recombine an individual without a valid objective
            r = igc(
                self.pop[a],
                ya,
                self.pop[b],
                yb,
                P,
                self.evaluator,
                self.rng,
                max_segments=cfg.max_segments,
                step10=cfg.step10,
                step10_p2=cfg.step10_p2,
            )
            self.igc_log.append({"generation": self.gen, "status": r.status, **r.trace})
            self._observe([*r.byproducts, *r.children])
            if r.status == "budget":
                return "time_limit" if self.evaluator.time_up() else "budget_exhausted"
            (self.pop[a], self.pop_y[a]), (self.pop[b], self.pop_y[b]) = r.children

        # Step 5: mutation, never applied to the current best individual
        ib = min(range(cfg.pop_size), key=self._key)
        for i in range(cfg.pop_size):
            if i != ib:
                g = P.mutate(self.pop[i], cfg.pm, self.rng)
                if g is not self.pop[i]:
                    self.pop[i], self.pop_y[i] = g, None
        self.gen += 1
        return None

    def _stop_reason(self, complete: bool) -> str | None:
        cfg, ev = self.cfg, self.evaluator
        if not complete or ev.remaining_calls() <= 0:
            return "budget_exhausted"
        if ev.time_up():
            return "time_limit"
        if cfg.target is not None and self.best and self.best[1][0] <= cfg.target:
            return "target_reached"
        if cfg.max_generations is not None and self.gen >= cfg.max_generations:
            return "max_generations"
        k = cfg.max_stall_generations
        if k and len(self._calls_log) > k and self._calls_log[-1] == self._calls_log[-1 - k]:
            return "stalled"
        return None
