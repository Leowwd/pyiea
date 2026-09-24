"""Pareto dominance, GPSIFF (Section IV-B, eq. 4), non-dominated sets and front metrics.

Objective vectors are minimized; GPSIFF scores are maximized.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt

from .evaluator import Objectives, genome_key
from .problem import Genome

__all__ = [
    "ParetoSet",
    "coverage",
    "dominance_matrix",
    "gpsiff",
    "hypervolume_2d",
    "igd",
    "nondominated_mask",
]

Points = npt.ArrayLike


def dominance_matrix(F: Points) -> npt.NDArray[np.bool_]:
    """``D[i, j]`` is True iff row ``i`` dominates row ``j`` (all ``<=``, at least one ``<``)."""
    A = np.asarray(F, dtype=float)
    le = np.asarray((A[:, None, :] <= A[None, :, :]).all(-1), dtype=bool)
    lt = np.asarray((A[:, None, :] < A[None, :, :]).any(-1), dtype=bool)
    return np.asarray(le & lt, dtype=bool)


def gpsiff(F: Points) -> npt.NDArray[np.int64]:
    """``GPSIFF(X) = p - q + c``: dominated count minus dominating count plus ``c = |S|``."""
    D = dominance_matrix(F)
    out: npt.NDArray[np.int64] = (D.sum(1) - D.sum(0) + len(D)).astype(np.int64)
    return out


def nondominated_mask(F: Points) -> npt.NDArray[np.bool_]:
    """True for rows of ``F`` that no other row dominates. Equal rows are all kept.

    Two objectives take an ``O(n log n)`` sort; more objectives compare all pairs.
    """
    A = np.asarray(F, dtype=float)
    if len(A) == 0:
        return np.zeros(0, dtype=bool)
    if A.ndim == 2 and A.shape[1] == 2:
        return _nondominated_mask_2d(A)
    dominated = np.asarray(dominance_matrix(A).any(0), dtype=bool)
    return ~dominated


def _nondominated_mask_2d(A: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    # Distinct rows in lexicographic order: every earlier row has f1' <= f1 and differs, so it
    # dominates this row iff f2' <= f2. Equal rows share one verdict through `inverse`.
    U, inverse = np.unique(A, axis=0, return_inverse=True)
    best_before = np.minimum.accumulate(np.r_[np.inf, U[:-1, 1]])
    keep = np.asarray(best_before > U[:, 1], dtype=bool)
    return np.asarray(keep[np.ravel(inverse)], dtype=bool)


def _dominates_any(A: npt.NDArray[np.float64], B: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """For each row of ``B``: is it dominated by some row of ``A``?"""
    if len(A) == 0 or len(B) == 0:
        return np.zeros(len(B), dtype=bool)
    le = (A[:, None, :] <= B[None, :, :]).all(-1)
    lt = (A[:, None, :] < B[None, :, :]).any(-1)
    return np.asarray((le & lt).any(0), dtype=bool)


class ParetoSet:
    """Non-dominated ``(genome, objectives)`` pairs, deduplicated by genome.

    Equal objective vectors from different genomes do not dominate each other and
    are all kept. Items are immutable, so population changes cannot alter them.
    """

    def __init__(self, items: Iterable[tuple[Genome, Objectives]] = ()) -> None:
        self._items: dict[bytes, tuple[Genome, Objectives]] = {}
        self._Y: npt.NDArray[np.float64] | None = None  # objectives in item order; None = recompute
        self.update(items)

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> list[tuple[Genome, Objectives]]:
        return list(self._items.values())

    def objectives(self) -> npt.NDArray[np.float64]:
        return self._objectives().copy()

    def _objectives(self) -> npt.NDArray[np.float64]:
        if self._Y is None:
            self._Y = np.array([y for _, y in self._items.values()], dtype=float).reshape(len(self), -1)
        return self._Y

    @classmethod
    def restore(cls, items: Sequence[tuple[Genome, Objectives]]) -> ParetoSet:
        """Rebuild a set from ``items()`` output, keeping order and skipping re-filtering."""
        s = cls()
        s._items = {genome_key(g): (g, tuple(y)) for g, y in items}
        s._Y = None
        return s

    def update(self, items: Iterable[tuple[Genome, Sequence[float]]]) -> None:
        """Add items, then drop every dominated member.

        Genomes already in the set are ignored. Only the new items are compared with
        the members and with each other, so an update costs ``O(len(self) * len(items))``;
        the result is the same as filtering the union, because the members never
        dominate each other.
        """
        new: dict[bytes, tuple[Genome, Objectives]] = {}
        for g, y in items:
            k = genome_key(g)
            if k not in self._items and k not in new:
                new[k] = (g, tuple(float(v) for v in y))
        if not new:
            return
        old = self._objectives() if self._items else None
        N = np.array([y for _, y in new.values()], dtype=float).reshape(len(new), -1)
        keep_new = nondominated_mask(N)
        if old is not None:
            keep_new &= ~_dominates_any(old, N)
            keep_old = ~_dominates_any(N[keep_new], old)
            if keep_old.all():
                kept, Y = self._items, np.vstack([old, N[keep_new]])
            else:
                kept = {k: v for (k, v), m in zip(self._items.items(), keep_old, strict=True) if m}
                Y = np.vstack([old[keep_old], N[keep_new]])
        else:
            kept, Y = {}, N[keep_new]
        kept.update({k: v for (k, v), m in zip(new.items(), keep_new, strict=True) if m})
        self._items, self._Y = kept, Y

    def drop_duplicate_objectives(self) -> None:
        """Keep only the first-inserted genome of every objective vector."""
        seen: set[Objectives] = set()
        keep: dict[bytes, tuple[Genome, Objectives]] = {}
        for k, (g, y) in self._items.items():
            if y not in seen:
                seen.add(y)
                keep[k] = (g, y)
        self._items, self._Y = keep, None

    def truncate(self, capacity: int, rng: np.random.Generator) -> None:
        """Randomly discard members until at most ``capacity`` remain (IMOEA Step 3)."""
        if len(self) > capacity:
            keys = list(self._items)
            keep = sorted(rng.choice(len(keys), capacity, replace=False))
            self._items = {keys[i]: self._items[keys[i]] for i in keep}
            self._Y = None

    def clear(self) -> None:
        self._items, self._Y = {}, None


def hypervolume_2d(F: Points, ref: Sequence[float]) -> float:
    """Area dominated by the points of ``F`` and bounded by ``ref`` (both objectives minimized)."""
    A: npt.NDArray[np.float64] = np.asarray(F, dtype=float).reshape(-1, 2)
    A = A[(np.asarray(ref) > A).all(1)]
    if len(A) == 0:
        return 0.0
    A = A[nondominated_mask(A)]
    A = A[np.argsort(A[:, 0])]
    xs = np.append(A[1:, 0], ref[0])
    return float(np.sum((xs - A[:, 0]) * (ref[1] - A[:, 1])))


def igd(F: Points, reference_front: Points) -> float:
    """Inverted generational distance: mean distance from each reference point to ``F``."""
    A, R = np.asarray(F, dtype=float), np.asarray(reference_front, dtype=float)
    return float(np.linalg.norm(R[:, None, :] - A[None, :, :], axis=-1).min(1).mean())


def coverage(A: Points, B: Points) -> float:
    """Cover metric ``C(A, B)`` of eq. (8): fraction of ``B`` weakly dominated by ``A``."""
    a, b = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    if len(b) == 0:
        return 0.0
    return float((a[:, None, :] <= b[None, :, :]).all(-1).any(0).mean())
