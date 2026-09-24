"""IMOEA (Ho, Shu, Chen 2004, Section IV-D, Steps 1-7) with GPSIFF fitness (Section IV-B)."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from .checkpoint import algorithm_config, load_checkpoint, save_checkpoint
from .evaluator import Evaluator, Objectives
from .exceptions import CheckpointError
from .iea import _check_probability
from .igc import Individual, igc
from .pareto import ParetoSet, gpsiff, nondominated_mask
from .problem import Genome, Problem

__all__ = ["IMOEA", "IMOEAConfig", "IMOEAResult"]

logger = logging.getLogger(__name__)


@dataclass
class IMOEAConfig:
    """IMOEA parameters. Defaults are the paper's ZDT settings (Table VIII).

    Attributes:
        pop_size: Population size ``N_pop`` (>= 2).
        elite_capacity: Capacity ``N_Emax`` of the working elite set E.
        ps: Fraction of the new population drawn from E.
        pc: Recombination probability.
        pm: Mutation probability passed to ``Problem.mutate``.
        max_segments: ``None`` uses the paper's division size; an integer caps it
            (``bounded_segments`` variant, unless the paper fixes ``N`` for the experiment).
        elite_unique_objectives: Keep one genome per objective vector in E before the
            random truncation of Step 3 (engineering option, off by default). Useful when
            many genomes decode to the same solution, e.g. under a repair decoder.
        max_generations: Stop after this many generations.
        max_stall_generations: Stop after this many generations without a new
            objective call (``0`` disables).
        checkpoint_every: Write a checkpoint every this many generations (``0`` never).
    """

    pop_size: int = 30
    elite_capacity: int = 30
    ps: float = 0.2
    pc: float = 0.6
    pm: float = 0.01
    max_segments: int | None = None
    elite_unique_objectives: bool = False
    max_generations: int | None = None
    max_stall_generations: int = 50
    checkpoint_every: int = 0

    def __post_init__(self) -> None:
        if self.pop_size < 2:
            raise ValueError(f"pop_size must be >= 2, got {self.pop_size}")
        if self.elite_capacity < 1:
            raise ValueError(f"elite_capacity must be >= 1, got {self.elite_capacity}")
        for name in ("ps", "pc", "pm"):
            _check_probability(name, getattr(self, name))
        if self.max_segments is not None and self.max_segments < 2:
            raise ValueError(f"max_segments must be >= 2 or None, got {self.max_segments}")


@dataclass(frozen=True)
class IMOEAResult:
    """What :meth:`IMOEA.optimize` returns.

    Attributes:
        front: Final working elite set E as ``(genome, objectives)`` pairs.
        archive: Every evaluated non-dominated candidate (never used for selection).
        generations: Completed generations.
        stop_reason: ``budget_exhausted``, ``time_limit``, ``max_generations`` or ``stalled``.
        history: Per-generation call counts and set sizes.
        igc_log: One trace per IGC operation.
        accounting: Evaluator counters, wall time and summed objective time.
        archive_bytes: Memory held by archived genomes.
    """

    front: list[Individual]
    archive: list[Individual] = field(repr=False)
    generations: int = 0
    stop_reason: str = ""
    history: list[dict[str, Any]] = field(default_factory=list, repr=False)
    igc_log: list[dict[str, Any]] = field(default_factory=list, repr=False)
    accounting: dict[str, float] = field(default_factory=dict, repr=False)
    archive_bytes: int = 0


class IMOEA:
    """Intelligent multi-objective EA with GPSIFF fitness, elite sets and IGC.

    Args:
        problem: Search space (see :class:`pyiea.Problem`).
        evaluator: Objectives (all minimized), budget and accounting.
        config: Algorithm parameters.
        seed: Seed of the run's random generator.
    """

    def __init__(
        self, problem: Problem, evaluator: Evaluator, config: IMOEAConfig | None = None, seed: int = 0
    ) -> None:
        self.problem, self.evaluator, self.cfg, self.seed = problem, evaluator, config or IMOEAConfig(), seed
        self.rng = np.random.default_rng(seed)
        self.gen = 0
        self.pop: list[Genome] = []
        self.pop_y: list[Objectives | None] = []
        self.elite = ParetoSet()  # E: bounded, used by selection
        self.temp = ParetoSet()  # E'
        self.archive = ParetoSet()  # every evaluated non-dominated candidate; never used by selection
        self.history: list[dict[str, Any]] = []
        self.igc_log: list[dict[str, Any]] = []
        self._calls_log: list[int] = []

    def _add_to_archive(self, items: list[tuple[Genome, Objectives | None]]) -> None:
        self.archive.update([(g, y) for g, y in items if y is not None])

    def _fitness(self) -> npt.NDArray[np.float64]:
        """GPSIFF over individuals with a valid objective; failed ones score 0 (the minimum is 1)."""
        ok = [(i, y) for i, y in enumerate(self.pop_y) if y is not None]
        fit = np.zeros(len(self.pop))
        if ok:
            fit[[i for i, _ in ok]] = gpsiff([y for _, y in ok])
        return fit

    def _evaluate_population(self) -> bool:
        idx = [i for i, y in enumerate(self.pop_y) if y is None]
        res = self.evaluator.evaluate_batch(
            [self.pop[i] for i in idx], phase="init" if self.gen == 0 else "mutation", partial_ok=True
        )
        for i, r in zip(idx, res, strict=True):
            self.pop_y[i] = r.objectives if r is not None else None
        self._add_to_archive([(self.pop[i], self.pop_y[i]) for i in idx])
        return all(r is not None for r in res)

    def _update_elite(self) -> None:
        """Step 3: non-dominated members of the population and E' join E, E' is emptied,
        dominated members of E are dropped and any excess over N_Emax is discarded at random.
        """
        ok = [(g, y) for g, y in zip(self.pop, self.pop_y, strict=True) if y is not None]
        if ok:
            nd = nondominated_mask([y for _, y in ok])
            self.elite.update([it for it, m in zip(ok, nd, strict=True) if m])
        self.elite.update(self.temp.items())
        self.temp.clear()
        if self.cfg.elite_unique_objectives:
            self.elite.drop_duplicate_objectives()
        self.elite.truncate(self.cfg.elite_capacity, self.rng)

    # --- checkpointing --------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "IMOEA",
            "config": asdict(self.cfg),
            "seed": self.seed,
            "gen": self.gen,
            "rng": self.rng.bit_generator.state,
            "pop": self.pop,
            "pop_y": self.pop_y,
            "elite": self.elite.items(),
            "temp": self.temp.items(),
            "archive": self.archive.items(),
            "history": self.history,
            "igc_log": self.igc_log,
            "calls_log": self._calls_log,
            "evaluator": self.evaluator.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a run. Stopping conditions may differ from the checkpoint (to extend a run).

        Raises:
            CheckpointError: if the kind, seed, algorithm parameters or evaluator context differ.
        """
        if (
            state.get("kind") != "IMOEA"
            or state["seed"] != self.seed
            or algorithm_config(state["config"]) != algorithm_config(asdict(self.cfg))
        ):
            raise CheckpointError("checkpoint does not match this IMOEA configuration and seed")
        self.evaluator.load_state_dict(state["evaluator"])
        self.gen, self.pop, self.pop_y = state["gen"], state["pop"], state["pop_y"]
        self.rng.bit_generator.state = state["rng"]
        self.elite, self.temp = ParetoSet.restore(state["elite"]), ParetoSet.restore(state["temp"])
        self.archive = ParetoSet.restore(state["archive"])
        self.history, self.igc_log, self._calls_log = state["history"], state["igc_log"], state["calls_log"]

    # --- main loop ------------------------------------------------------------------
    def optimize(self, checkpoint_path: str | os.PathLike[str] | None = None, resume: bool = False) -> IMOEAResult:
        """Run until a stopping condition holds (see :meth:`pyiea.IEA.optimize` for the arguments)."""
        cfg, P, ev = self.cfg, self.problem, self.evaluator
        if resume:
            if checkpoint_path is None:
                raise ValueError("resume=True needs checkpoint_path")
            self.load_state_dict(load_checkpoint(checkpoint_path))
        elif not self.pop:
            self.pop = [P.random_genome(self.rng) for _ in range(cfg.pop_size)]  # Step 1
            self.pop_y = [None] * cfg.pop_size
        n_pairs = int(cfg.pc * cfg.pop_size) // 2
        reason: str | None = None

        try:
            while True:
                if cfg.checkpoint_every and checkpoint_path and self.gen % cfg.checkpoint_every == 0:
                    save_checkpoint(checkpoint_path, self.state_dict())
                complete = self._evaluate_population()  # Step 2 (GPSIFF is computed where used, Step 4)
                self._update_elite()  # Step 3
                self._calls_log.append(ev.counters["objective_calls"])
                self.history.append(
                    {
                        "generation": self.gen,
                        "calls": ev.counters["objective_calls"],
                        "seconds": ev.elapsed(),
                        "elite_size": len(self.elite),
                        "archive_size": len(self.archive),
                    }
                )
                logger.debug(
                    "IMOEA gen %d: |E| %d, |archive| %d, calls %d",
                    self.gen,
                    len(self.elite),
                    len(self.archive),
                    ev.counters["objective_calls"],
                )
                reason = self._stop_reason(complete)  # Step 7
                if reason:
                    break

                # Step 4: Npop - Nps by binary tournament on GPSIFF, Nps drawn at random from E
                fit = self._fitness()
                n_ps = min(int(cfg.pop_size * cfg.ps), len(self.elite))
                new_pop: list[Genome] = []
                new_y: list[Objectives | None] = []
                for _ in range(cfg.pop_size - n_ps):
                    a, b = self.rng.integers(0, cfg.pop_size, 2)
                    w = a if fit[a] > fit[b] or (fit[a] == fit[b] and self.rng.random() < 0.5) else b
                    new_pop.append(self.pop[w])
                    new_y.append(self.pop_y[w])
                elite = self.elite.items()
                for i in self.rng.choice(len(elite), n_ps, replace=False):
                    new_pop.append(elite[i][0])
                    new_y.append(elite[i][1])
                self.pop, self.pop_y = new_pop, new_y

                # Step 5: IGC on pc*Npop random parents; non-dominated by-products and children go to E'
                chosen = self.rng.choice(cfg.pop_size, 2 * n_pairs, replace=False)
                for a, b in zip(chosen[0::2], chosen[1::2], strict=True):
                    ya, yb = self.pop_y[a], self.pop_y[b]
                    if ya is None or yb is None:
                        continue
                    r = igc(
                        self.pop[a],
                        ya,
                        self.pop[b],
                        yb,
                        P,
                        ev,
                        self.rng,
                        multi_objective=True,
                        max_segments=cfg.max_segments,
                    )
                    self.igc_log.append({"generation": self.gen, "status": r.status, **r.trace})
                    produced = r.byproducts + (r.children if r.status == "applied" else [])
                    self._add_to_archive(list(produced))
                    if produced:
                        nd = nondominated_mask([y for _, y in produced])
                        self.temp.update([it for it, m in zip(produced, nd, strict=True) if m])
                    if r.status == "budget":
                        reason = "time_limit" if ev.time_up() else "budget_exhausted"
                        break
                    (self.pop[a], self.pop_y[a]), (self.pop[b], self.pop_y[b]) = r.children
                if reason:
                    break

                # Step 6: mutation of the whole population (IMOEA has no best-individual exemption)
                for i in range(cfg.pop_size):
                    g = P.mutate(self.pop[i], cfg.pm, self.rng)
                    if g is not self.pop[i]:
                        self.pop[i], self.pop_y[i] = g, None
                self.gen += 1
        finally:
            ev.close()

        self._update_elite()  # completed temporary candidates join the output; unevaluated ones cannot
        assert reason is not None
        return IMOEAResult(
            front=self.elite.items(),
            archive=self.archive.items(),
            generations=self.gen,
            stop_reason=reason,
            history=self.history,
            igc_log=self.igc_log,
            accounting=ev.summary(),
            archive_bytes=sum(g.nbytes for g, _ in self.archive.items()),
        )

    def _stop_reason(self, complete: bool) -> str | None:
        ev, cfg = self.evaluator, self.cfg
        if not complete or ev.remaining_calls() <= 0:
            return "budget_exhausted"
        if ev.time_up():
            return "time_limit"
        if cfg.max_generations is not None and self.gen >= cfg.max_generations:
            return "max_generations"
        k = cfg.max_stall_generations
        if k and len(self._calls_log) > k and self._calls_log[-1] == self._calls_log[-1 - k]:
            return "stalled"
        return None
