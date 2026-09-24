"""Seeded, picklable test objectives with independently checkable optima (all minimized)."""

from __future__ import annotations

import itertools
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .encoding import ParameterProblem, RealEncoder
from .evaluator import Objective
from .pareto import nondominated_mask
from .problem import Genome

__all__ = [
    "Costly",
    "FixedCardWeighted",
    "KeepDropBiObjective",
    "MultiKnapsack",
    "PAPER_FUNCTIONS",
    "PaperBenchmark",
    "QuadraticBinary",
    "WeightedSum",
    "ZDT",
    "ZDT_PROBLEMS",
]


class WeightedSum:
    """``f(g) = w . g`` with standard-normal weights. Additive; optimum keeps the negative weights."""

    def __init__(self, n_bits: int, seed: int = 0) -> None:
        self.w = np.random.default_rng(seed).normal(size=n_bits)

    def __call__(self, g: Genome) -> float:
        return float(self.w @ g)

    def optimum(self) -> float:
        return float(self.w[self.w < 0].sum())


class FixedCardWeighted(WeightedSum):
    """``f(g) = w . g`` subject to ``sum(g) = k``. Optimum: the ``k`` smallest weights."""

    def __init__(self, n_bits: int, k: int, seed: int = 0) -> None:
        super().__init__(n_bits, seed)
        self.k = k

    def optimum(self) -> float:
        return float(np.sort(self.w)[: self.k].sum())


class QuadraticBinary:
    """``f(g) = g^T Q g`` with a random symmetric ``Q``: pairwise interactions (epistatic)."""

    def __init__(self, n_bits: int, seed: int = 0) -> None:
        a = np.random.default_rng(seed).normal(size=(n_bits, n_bits))
        self.Q = (a + a.T) / 2

    def __call__(self, g: Genome) -> float:
        x = g.astype(float)
        return float(x @ self.Q @ x)

    def optimum(self) -> float:
        """Exhaustive search; only practical for ``n_bits <= ~24``."""
        return min(self(np.array(b, dtype=np.uint8)) for b in itertools.product((0, 1), repeat=len(self.Q)))


class KeepDropBiObjective:
    """Toy compression trade-off ``(D, R)`` for keep bits ``g`` (1 = keep).

    ``D`` is the cost of the dropped units plus extra cost for dropping interacting
    pairs; ``R`` is the retained size ratio including a fixed, always-kept part.
    """

    def __init__(self, n_bits: int, seed: int = 0, fixed: float = 0.1) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(0.1, 1.0, n_bits)
        a = rng.uniform(0, 0.3, (n_bits, n_bits)) * (rng.random((n_bits, n_bits)) < 0.3)
        self.Q = np.triu(a, 1)
        self.c = rng.uniform(0.5, 1.5, n_bits)
        self.fixed = fixed * self.c.sum()

    def __call__(self, g: Genome) -> tuple[float, float]:
        d = 1.0 - g.astype(float)
        return float(self.w @ d + d @ self.Q @ d), float((self.fixed + self.c @ g) / (self.fixed + self.c.sum()))

    def exact_front(self) -> npt.NDArray[np.float64]:
        """True Pareto front by enumeration; only practical for ``n_bits <= ~16``."""
        Y = np.array([self(np.array(b, dtype=np.uint8)) for b in itertools.product((0, 1), repeat=len(self.w))])
        return Y[nondominated_mask(Y)]


def _f1(x: npt.NDArray[np.float64]) -> float:
    return float(-np.sum(np.sin(x) + np.sin(2 * x / 3)))


def _f2(x: npt.NDArray[np.float64]) -> float:
    a, b = x[:-1], x[1:]
    return float(-np.sum(np.sin(a + b) + np.sin(2 * a * b / 3)))


def _f3(x: npt.NDArray[np.float64]) -> float:
    return float(np.sum(np.floor(x + 0.5) ** 2))


def _f4(x: npt.NDArray[np.float64]) -> float:
    return float(np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10))


def _f5(x: npt.NDArray[np.float64]) -> float:
    return float(np.sum(x**2))


def _f6(x: npt.NDArray[np.float64]) -> float:
    return float(np.sum(x * np.sin(10 * np.pi * x)))


def _f7(x: npt.NDArray[np.float64]) -> float:
    z = 10 * x * np.pi
    return float(np.sum(np.abs(np.sinc(z / np.pi))))  # |sin(z)/z|, equal to 1 at z = 0


def _f8(x: npt.NDArray[np.float64]) -> float:
    d = len(x)
    return float(20 + np.e - 20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / d))


def _f9(x: npt.NDArray[np.float64]) -> float:
    return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def _f10(x: npt.NDArray[np.float64]) -> float:
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))


def _f11(x: npt.NDArray[np.float64]) -> float:
    return float(6 * len(x) + np.sum(np.floor(x)))


def _f12(x: npt.NDArray[np.float64]) -> float:
    i = np.arange(1, len(x) + 1)
    return float(np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(i))) + 1)


# name: (function, (low, high), maximize, bits per parameter, optimum as a function of D)
_PAPER: dict[
    str, tuple[Callable[[npt.NDArray[np.float64]], float], tuple[float, float], bool, int, Callable[[int], float]]
] = {
    "f1": (_f1, (3.0, 13.0), True, 10, lambda d: 1.21598 * d),
    "f2": (_f2, (3.0, 13.0), True, 10, lambda d: 2.0 * d),  # "approximately 2D" in Table IV
    "f3": (_f3, (-100.0, 100.0), False, 10, lambda d: 0.0),
    "f4": (_f4, (-5.12, 5.12), False, 10, lambda d: 0.0),
    "f5": (_f5, (-5.12, 5.12), False, 10, lambda d: 0.0),
    "f6": (_f6, (-1.0, 2.0), True, 10, lambda d: 1.85 * d),
    "f7": (_f7, (-0.5, 0.5), False, 10, lambda d: 0.0),
    "f8": (_f8, (-30.0, 30.0), False, 10, lambda d: 0.0),
    "f9": (_f9, (-500.0, 500.0), False, 24, lambda d: 0.0),
    "f10": (_f10, (-5.12, 5.12), False, 10, lambda d: 0.0),
    "f11": (_f11, (-5.12, 5.12), False, 10, lambda d: 0.0),
    "f12": (_f12, (-600.0, 600.0), False, 24, lambda d: 0.0),
}
PAPER_FUNCTIONS = tuple(_PAPER)
"""Names of the twelve benchmark functions of Table IV of the IEA paper."""


class PaperBenchmark:
    """Benchmark ``f1`` ... ``f12`` of Table IV of the IEA paper on real parameters ``x``.

    Calling returns the value to **minimize** (maximized functions are negated);
    :meth:`paper_value` converts back to the paper's sign. Each parameter is
    encoded with 10 bits, or 24 bits for ``f9`` and ``f12`` (Section V-B); use
    :meth:`encoder` and ``encoder.problem()`` so that IGC never splits a parameter.
    The encoder is Gray-coded: the paper does not name its encoding, and with plain
    binary pyiea stays well short of Table III and Section V-A (engineering choice,
    see ``docs/paper_map.md``).

    Example:
        >>> b = PaperBenchmark("f5", dims=3)
        >>> b(np.zeros(3)), b.optimum
        (0.0, 0.0)
    """

    def __init__(self, name: str, dims: int) -> None:
        if name not in _PAPER:
            raise ValueError(f"unknown benchmark {name!r}; choose from {PAPER_FUNCTIONS}")
        self.name, self.dims = name, dims
        self._fn, self.bounds, self.maximize, self.bits, opt = _PAPER[name]
        self.optimum = opt(dims)

    def __call__(self, x: npt.NDArray[np.float64]) -> float:
        v = self._fn(np.asarray(x, dtype=float))
        return -v if self.maximize else v

    def paper_value(self, minimized: float) -> float:
        """Objective in the paper's sign convention."""
        return -minimized if self.maximize else minimized

    def encoder(self, bits: int | None = None, gray: bool = True) -> RealEncoder:
        """Encoder over ``dims`` parameters with the paper's bits per parameter unless ``bits`` is given."""
        return RealEncoder([self.bounds] * self.dims, bits=bits or self.bits, gray=gray)


# --- IMOEA paper, Section VI-A: multi-objective 0/1 knapsack (eq. 9) ---------------------------------------------


class MultiKnapsack:
    """Multi-objective 0/1 knapsack of eq. (9) with the greedy repair of Zitzler & Thiele (1999).

    ``I`` knapsacks and ``J`` items: maximize ``f_i(x) = sum_j p[i, j] x_j`` for every
    knapsack ``i`` subject to ``sum_j w[i, j] x_j <= c[i]``. The genome ``s`` is any bit
    string; the objective decodes it into a feasible ``x = repair(s)`` and returns
    ``(-f_1(x), ..., -f_I(x))`` (minimized). The genome itself is never rewritten.

    Repair: items are removed from ``s`` in increasing order of ``q_j = max_i p[i, j] / w[i, j]``
    (ties by item index) until every capacity constraint holds.

    Example:
        >>> k = MultiKnapsack([[10, 20, 30]], [[5, 5, 5]], [10])
        >>> k(np.ones(3, dtype=np.uint8))  # item 0 has the lowest q and is removed
        (-50.0,)
    """

    def __init__(self, profits: npt.ArrayLike, weights: npt.ArrayLike, capacities: npt.ArrayLike) -> None:
        self.p = np.asarray(profits, dtype=np.int64)
        self.w = np.asarray(weights, dtype=np.int64)
        self.c = np.asarray(capacities, dtype=np.int64)
        if self.p.ndim != 2 or self.p.shape != self.w.shape or self.c.shape != (len(self.p),):
            raise ValueError("profits and weights must be (I, J) and capacities (I,)")
        if np.any(self.w <= 0):
            raise ValueError("weights must be positive")
        self.n_knapsacks, self.n_items = self.p.shape
        q = (self.p / self.w).max(0)
        self.removal_order = np.argsort(q, kind="stable")  # lowest max profit/weight ratio first

    @classmethod
    def random(cls, n_items: int, n_knapsacks: int, seed: int = 0) -> MultiKnapsack:
        """Seeded instance generated by the rule of eq. (9); **not** the Zitzler-Thiele test data.

        ``p`` and ``w`` are uniform integers in ``[10, 100]`` and ``c_i = floor(sum_j w[i, j] / 2)``.
        """
        rng = np.random.default_rng(seed)
        w = rng.integers(10, 101, (n_knapsacks, n_items))
        p = rng.integers(10, 101, (n_knapsacks, n_items))
        return cls(p, w, w.sum(1) // 2)

    @classmethod
    def from_zt_file(cls, path: str | os.PathLike[str]) -> MultiKnapsack:
        """Read a ``knapsack.<J>.<I>`` file in the Zitzler-Thiele (1999) text format."""
        text = Path(path).read_text()
        head = re.match(r"knapsack problem specification \((\d+) knapsacks, (\d+) items\)", text)
        if head is None:
            raise ValueError(f"{path}: not a Zitzler-Thiele knapsack file")
        n_knapsacks, n_items = int(head[1]), int(head[2])
        caps = [int(v) for v in re.findall(r"capacity:\s*\+?(\d+)", text)]
        w = [int(v) for v in re.findall(r"weight:\s*\+?(\d+)", text)]
        p = [int(v) for v in re.findall(r"profit:\s*\+?(\d+)", text)]
        if len(caps) != n_knapsacks or len(w) != len(p) or len(w) != n_knapsacks * n_items:
            raise ValueError(f"{path}: expected {n_knapsacks} knapsacks x {n_items} items")
        shape = (n_knapsacks, n_items)
        return cls(np.reshape(p, shape), np.reshape(w, shape), caps)

    def is_feasible(self, x: npt.ArrayLike) -> bool:
        return bool(np.all(self.w @ np.asarray(x, dtype=np.int64) <= self.c))

    def repair(self, s: Genome) -> npt.NDArray[np.uint8]:
        """The feasible ``x`` that ``s`` decodes to: the shortest removal prefix in greedy order."""
        x = np.array(s, dtype=np.uint8)
        excess = self.w @ x.astype(np.int64) - self.c
        if np.all(excess <= 0):
            return x
        chosen = self.removal_order[x[self.removal_order] == 1]
        removed = np.cumsum(self.w[:, chosen], axis=1)  # weight removed after the first k+1 removals
        k = max(int(np.searchsorted(removed[i], excess[i])) for i in range(self.n_knapsacks) if excess[i] > 0)
        x[chosen[: k + 1]] = 0
        return x

    def profits(self, x: npt.ArrayLike) -> npt.NDArray[np.int64]:
        out: npt.NDArray[np.int64] = self.p @ np.asarray(x, dtype=np.int64)
        return out

    def __call__(self, g: Genome) -> tuple[float, ...]:
        return tuple(-float(v) for v in self.profits(self.repair(g)))

    def exact_front(self) -> npt.NDArray[np.float64]:
        """Pareto front (minimized, i.e. negated profits) of all feasible ``x``; only for ``J <= ~16``."""
        X = np.array(list(itertools.product((0, 1), repeat=self.n_items)), dtype=np.int64)
        load = X @ self.w.T
        X = X[(load <= self.c).all(1)]
        Y: npt.NDArray[np.float64] = np.unique(-(X @ self.p.T).astype(float), axis=0)
        return Y[nondominated_mask(Y)]


# --- IMOEA paper, Section VI-B: ZDT1-ZDT6 (eq. 10) ------------------------------------------------------------------

ZDT_PROBLEMS = ("zdt1", "zdt2", "zdt3", "zdt4", "zdt5", "zdt6")
"""Names of the six test problems of Zitzler, Deb & Thiele (2000)."""


class ZDT:
    """ZDT1-ZDT6 of Zitzler, Deb & Thiele (2000), "Comparison of multiobjective evolutionary
    algorithms: empirical results", Evol. Comput. 8(2):173-195; eq. (10) of the IMOEA paper.

    ``T(X) = (f1(x1), f2(X))`` with ``f2 = g(x2..xm) * h(f1, g)``, both minimized:

    * ZDT1: ``f1 = x1``, ``g = 1 + 9 sum(x2..xm) / (m-1)``, ``h = 1 - sqrt(f1/g)``; ``x in [0, 1]``.
    * ZDT2: as ZDT1 with ``h = 1 - (f1/g)^2``.
    * ZDT3: as ZDT1 with ``h = 1 - sqrt(f1/g) - (f1/g) sin(10 pi f1)``.
    * ZDT4: ``x1 in [0, 1]``, ``x2..xm in [-5, 5]``, ``g = 1 + 10(m-1) + sum(xi^2 - 10 cos(4 pi xi))``,
      ``h = 1 - sqrt(f1/g)``.
    * ZDT5: ``x1`` is 30 bits and ``x2..xm`` 5 bits each; ``u`` counts ones, ``f1 = 1 + u(x1)``,
      ``g = sum v(u(xi))`` with ``v(u) = 2 + u`` for ``u < 5`` and ``v(5) = 1``, ``h = 1/f1``.
    * ZDT6: ``f1 = 1 - exp(-4 x1) sin^6(6 pi x1)``, ``g = 1 + 9 (sum(x2..xm) / (m-1))^0.25``,
      ``h = 1 - (f1/g)^2``; ``x in [0, 1]``.

    ZDT1-4 and 6 take real parameters, so the objective on genomes is ``objective(gray)``,
    which decodes ``bits`` bits per parameter (:class:`RealEncoder`). ZDT5 is defined on the
    bits themselves, so its genome is the bit string and there is no real encoding.

    Example:
        >>> z = ZDT("zdt1", m=3)
        >>> z(np.array([0.25, 0.0, 0.0]))
        (0.25, 0.5)
    """

    def __init__(self, name: str, m: int = 30, bits: int = 30) -> None:
        if name not in ZDT_PROBLEMS:
            raise ValueError(f"unknown problem {name!r}; choose from {ZDT_PROBLEMS}")
        if m < 2:
            raise ValueError(f"m must be >= 2, got {m}")
        self.name, self.m, self.bits = name, m, bits
        rest = (-5.0, 5.0) if name == "zdt4" else (0.0, 1.0)
        self.bounds = [(0.0, 1.0)] + [rest] * (m - 1)
        self.widths = [30] + [5] * (m - 1) if name == "zdt5" else [bits] * m

    def __call__(self, x: npt.ArrayLike) -> tuple[float, float]:
        """Objectives of real parameters ``x`` (ZDT5: of the bit string)."""
        v = np.asarray(x, dtype=float)
        m = self.m
        if self.name == "zdt5":
            u = np.add.reduceat(v, np.r_[0, np.cumsum(self.widths)[:-1]])
            f1 = 1.0 + u[0]
            g = float(np.sum(np.where(u[1:] < 5, 2.0 + u[1:], 1.0)))
            return float(f1), g / f1
        x1, rest = v[0], v[1:]
        if self.name == "zdt6":
            f1 = 1.0 - np.exp(-4.0 * x1) * np.sin(6.0 * np.pi * x1) ** 6
            g = 1.0 + 9.0 * (rest.sum() / (m - 1)) ** 0.25
        elif self.name == "zdt4":
            f1 = x1
            g = 1.0 + 10.0 * (m - 1) + np.sum(rest**2 - 10.0 * np.cos(4.0 * np.pi * rest))
        else:
            f1 = x1
            g = 1.0 + 9.0 * rest.sum() / (m - 1)
        r = f1 / g
        if self.name in ("zdt2", "zdt6"):
            h = 1.0 - r**2
        elif self.name == "zdt3":
            h = 1.0 - np.sqrt(r) - r * np.sin(10.0 * np.pi * f1)
        else:
            h = 1.0 - np.sqrt(r)
        return float(f1), float(g * h)

    def encoder(self, gray: bool = False) -> RealEncoder:
        """Fixed-point encoding with ``bits`` bits per parameter (not for ZDT5)."""
        if self.name == "zdt5":
            raise ValueError("ZDT5 is defined on bits; use problem() and objective() directly")
        return RealEncoder(self.bounds, bits=self.bits, gray=gray)

    def problem(self, gray: bool = False) -> ParameterProblem:
        """Search space whose IGC cut points lie only between parameters."""
        if self.name == "zdt5":
            if gray:
                raise ValueError("ZDT5 is defined on bits; there is no Gray variant")
            return ParameterProblem(self.m, self.widths, f"zdt5-m{self.m}")
        return self.encoder(gray).problem(f"{self.name}-m{self.m}-{self.bits}b{'-gray' if gray else ''}")

    def objective(self, gray: bool = False) -> Objective:
        """Picklable objective on genomes."""
        if self.name == "zdt5":
            if gray:
                raise ValueError("ZDT5 is defined on bits; there is no Gray variant")
            return self
        return self.encoder(gray).wrap(self)

    def pareto_front(self, n: int = 1000) -> npt.NDArray[np.float64]:
        """Points of the analytic Pareto-optimal front (``g`` at its minimum), non-dominated, sorted by ``f1``."""
        if self.name == "zdt5":
            u1 = np.arange(1.0, 32.0)  # f1 = 1 + (ones in x1)
            F = np.column_stack([u1, (self.m - 1) / u1])
        elif self.name == "zdt6":
            x1 = np.linspace(0.0, 1.0, 200_001)
            lo = float((1.0 - np.exp(-4.0 * x1) * np.sin(6.0 * np.pi * x1) ** 6).min())
            f1 = np.linspace(lo, 1.0, n)
            F = np.column_stack([f1, 1.0 - f1**2])
        elif self.name == "zdt3":
            f1 = np.linspace(0.0, 1.0, 50 * n)
            F = np.column_stack([f1, 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)])
        else:
            f1 = np.linspace(0.0, 1.0, n)
            F = np.column_stack([f1, 1.0 - f1**2 if self.name == "zdt2" else 1.0 - np.sqrt(f1)])
        out: npt.NDArray[np.float64] = np.asarray(F[nondominated_mask(F)], dtype=np.float64)
        return out


class Costly:
    """Adds a fixed amount of CPU work to an objective, to profile parallel evaluation."""

    def __init__(self, inner: Objective, work: int = 200) -> None:
        self.inner, self.work = inner, work
        self.m = np.random.default_rng(0).normal(size=(64, 64)) / 8

    def __call__(self, g: Genome) -> float | Sequence[float]:
        x = self.m
        for _ in range(self.work):
            x = np.tanh(x @ self.m)
        return self.inner(g)
