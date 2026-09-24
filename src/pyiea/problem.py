"""Binary search spaces: initialization, legality, mutation and IGC division.

A genome is a 1-D ``uint8`` array of 0/1 values that is made read-only on
creation (:func:`freeze`), so parents can never be modified in place.

To add a constrained search space, subclass :class:`BinaryProblem` and keep
every operator feasible by construction (see :class:`FixedCardinalityProblem`).
That is far cheaper than rejecting infeasible genomes in the fitness function.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

__all__ = [
    "BinaryProblem",
    "FixedCardinalityProblem",
    "Genome",
    "Problem",
    "freeze",
    "paper_n_segments",
]

Genome = npt.NDArray[np.uint8]
"""A read-only 1-D ``uint8`` array of 0/1 values."""


def freeze(g: npt.ArrayLike) -> Genome:
    """Return ``g`` as a contiguous, read-only ``uint8`` genome."""
    out = np.array(g, dtype=np.uint8, copy=True)
    out.flags.writeable = False
    return out


def paper_n_segments(m: int) -> int:
    """Low-epistasis division size of Section III-A: ``N = 2**floor(log2(M + 1)) - 1``."""
    return int((1 << ((m + 1).bit_length() - 1)) - 1)


@runtime_checkable
class Problem(Protocol):
    """What IEA and IMOEA need from a search space.

    Implementations must keep genomes feasible by construction: ``random_genome``,
    ``mutate`` and every OA combination built from ``divide`` must be legal.
    ``version`` identifies the problem definition in evaluator contexts and checkpoints.
    """

    n_bits: int
    version: str

    def random_genome(self, rng: np.random.Generator) -> Genome: ...

    def is_valid(self, g: Genome) -> bool: ...

    def mutate(self, g: Genome, pm: float, rng: np.random.Generator) -> Genome:
        """Return a mutated genome, or ``g`` itself (same object) if nothing changed."""
        ...

    def divide(
        self,
        p1: Genome,
        p2: Genome,
        diff: npt.NDArray[np.intp],
        max_segments: int | None,
        rng: np.random.Generator,
    ) -> list[npt.NDArray[np.intp]]:
        """Split the differing positions ``diff`` into gene segments (IGC Step 2).

        Return an empty list when fewer than two segments are possible.
        """
        ...


class BinaryProblem:
    """Unconstrained bit string (Section III-A, a 0/1 problem with low epistasis).

    Args:
        n_bits: Genome length.
        version: Identifier stored in evaluator contexts and checkpoints.
    """

    def __init__(self, n_bits: int, version: str = "binary-v1") -> None:
        if n_bits < 1:
            raise ValueError(f"n_bits must be >= 1, got {n_bits}")
        self.n_bits = n_bits
        self.version = version

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_bits={self.n_bits}, version={self.version!r})"

    def random_genome(self, rng: np.random.Generator) -> Genome:
        """Uniformly random bits."""
        return freeze(rng.integers(0, 2, self.n_bits, dtype=np.uint8))

    def is_valid(self, g: Genome) -> bool:
        return g.shape == (self.n_bits,) and bool(np.all(g <= 1))

    def mutate(self, g: Genome, pm: float, rng: np.random.Generator) -> Genome:
        """Bit-inverse mutation: every bit flips independently with probability ``pm``."""
        flip = rng.random(self.n_bits) < pm
        return freeze(g ^ flip) if flip.any() else g

    def crossover(self, p1: Genome, p2: Genome, rng: np.random.Generator) -> Genome:
        """Uniform crossover. Only the baselines use it; IEA/IMOEA recombine with IGC."""
        return freeze(np.where(rng.random(self.n_bits) < 0.5, p1, p2))

    def divide(
        self,
        p1: Genome,
        p2: Genome,
        diff: npt.NDArray[np.intp],
        max_segments: int | None,
        rng: np.random.Generator,
    ) -> list[npt.NDArray[np.intp]]:
        """``N = 2**floor(log2(M+1)) - 1`` contiguous segments, cut at ``N - 1`` random points.

        ``max_segments`` caps ``N`` (the ``bounded_segments`` variant).
        """
        m = len(diff)
        n = paper_n_segments(m)
        if max_segments is not None:
            n = min(n, max_segments)
        if n < 2:
            return []
        cuts = np.sort(rng.choice(m - 1, n - 1, replace=False) + 1)
        return list(np.split(diff, cuts))


class FixedCardinalityProblem(BinaryProblem):
    """Bit strings with exactly ``k`` ones.

    The division follows the feasibility-preserving criterion of the IEA paper's
    polygonal-approximation example (Section III-A), and mutation swaps bits, so
    every candidate stays feasible.
    """

    def __init__(self, n_bits: int, k: int, version: str = "fixedcard-v1") -> None:
        super().__init__(n_bits, version)
        if not 0 < k < n_bits:
            raise ValueError(f"k must satisfy 0 < k < n_bits, got k={k}, n_bits={n_bits}")
        self.k = k

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_bits={self.n_bits}, k={self.k}, version={self.version!r})"

    def random_genome(self, rng: np.random.Generator) -> Genome:
        g = np.zeros(self.n_bits, dtype=np.uint8)
        g[rng.choice(self.n_bits, self.k, replace=False)] = 1
        return freeze(g)

    def is_valid(self, g: Genome) -> bool:
        return super().is_valid(g) and int(g.sum()) == self.k

    def mutate(self, g: Genome, pm: float, rng: np.random.Generator) -> Genome:
        """Swap mutation: each 1-bit, with probability ``pm``, swaps with a random 0-bit.

        ``pm`` is a per-bit probability, as in :meth:`BinaryProblem.mutate`.
        """
        ones, zeros = np.flatnonzero(g == 1), np.flatnonzero(g == 0)
        pick = ones[rng.random(len(ones)) < pm][: len(zeros)]
        if len(pick) == 0:
            return g
        out = g.copy()
        out[pick] = 0
        out[rng.choice(zeros, len(pick), replace=False)] = 1
        return freeze(out)

    def crossover(self, p1: Genome, p2: Genome, rng: np.random.Generator) -> Genome:
        """Cardinality-preserving uniform crossover (baselines only)."""
        out = (p1 & p2).astype(np.uint8)
        diff = np.flatnonzero(p1 != p2)
        out[rng.choice(diff, self.k - int(out.sum()), replace=False)] = 1
        return freeze(out)

    def divide(
        self,
        p1: Genome,
        p2: Genome,
        diff: npt.NDArray[np.intp],
        max_segments: int | None,
        rng: np.random.Generator,
    ) -> list[npt.NDArray[np.intp]]:
        """Maximal contiguous segments holding equal numbers of 1s in both parents.

        A cut is placed after every prefix of ``diff`` where parent 1's surplus of
        ones returns to zero. With ``max_segments``, a random subset of those cuts is kept.
        """
        balance = np.cumsum(np.where(p1[diff] == 1, 1, -1))
        cuts = np.flatnonzero(balance[:-1] == 0) + 1
        if max_segments is not None and len(cuts) + 1 > max_segments:
            cuts = np.sort(rng.choice(cuts, max_segments - 1, replace=False)) if max_segments > 1 else cuts[:0]
        if len(cuts) + 1 < 2:
            return []
        return list(np.split(diff, cuts))
