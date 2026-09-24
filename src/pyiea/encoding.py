"""Binary encoding of real-valued parameters, as used for the paper's benchmarks.

Each parameter takes ``bits`` consecutive genes (most significant bit first), so a
parameter's bits stay adjacent and IGC segments tend to keep them together. With
``gray=True`` the genes are a reflected Gray code, so neighbouring parameter values
differ in one bit and bit-flip mutation can fine-tune a converged population.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from .problem import BinaryProblem, Genome, paper_n_segments

__all__ = ["EncodedObjective", "ParameterProblem", "RealEncoder"]


class RealEncoder:
    """Maps a bit string to real parameters with fixed-point resolution.

    Args:
        bounds: ``(low, high)`` per parameter.
        bits: Bits per parameter (the paper uses 10 for most benchmarks).
        gray: Read each parameter's bits as a reflected Gray code instead of plain binary.

    Example:
        >>> enc = RealEncoder([(0.0, 1.0)] * 2, bits=2)
        >>> enc.decode(np.array([1, 1, 0, 0], dtype=np.uint8)).tolist()
        [1.0, 0.0]
        >>> RealEncoder([(0.0, 3.0)], bits=2, gray=True).decode(np.array([1, 0], dtype=np.uint8)).tolist()
        [3.0]
    """

    def __init__(self, bounds: Sequence[tuple[float, float]], bits: int = 10, gray: bool = False) -> None:
        if bits < 1 or bits > 52:
            raise ValueError(f"bits must be in [1, 52], got {bits}")
        b = np.asarray(bounds, dtype=float).reshape(-1, 2)
        if len(b) == 0 or np.any(b[:, 1] <= b[:, 0]):
            raise ValueError("bounds need at least one (low, high) pair with low < high")
        self.low, self.high, self.bits, self.gray = b[:, 0], b[:, 1], bits, gray
        self.n_params = len(b)
        self.n_bits = self.n_params * bits
        self._weights = 2.0 ** np.arange(bits - 1, -1, -1)

    def decode(self, g: Genome) -> npt.NDArray[np.float64]:
        """Parameters in ``[low, high]``, evenly spaced over ``2**bits`` levels."""
        bits: npt.NDArray[np.uint8] = np.asarray(g, dtype=np.uint8).reshape(self.n_params, self.bits)
        if self.gray:
            bits = np.bitwise_xor.accumulate(bits, axis=1)  # Gray -> binary, MSB first
        ints = bits.astype(float) @ self._weights
        out: npt.NDArray[np.float64] = self.low + ints / (2.0**self.bits - 1) * (self.high - self.low)
        return out

    def problem(self, version: str | None = None) -> ParameterProblem:
        """The matching search space, whose IGC cut points lie only between parameters (Section III-A)."""
        default = f"params-{self.n_params}x{self.bits}{'-gray' if self.gray else ''}"
        return ParameterProblem(self.n_params, self.bits, version or default)

    def wrap(self, fn: Callable[[npt.NDArray[np.float64]], float | Sequence[float]]) -> EncodedObjective:
        """A picklable objective on genomes that decodes and then calls ``fn(x)``."""
        return EncodedObjective(self, fn)


class EncodedObjective:
    """``genome -> fn(encoder.decode(genome))``. Picklable when ``fn`` is."""

    def __init__(self, encoder: RealEncoder, fn: Callable[[npt.NDArray[np.float64]], float | Sequence[float]]) -> None:
        self.encoder, self.fn = encoder, fn

    def __call__(self, g: Genome) -> float | Sequence[float]:
        return self.fn(self.encoder.decode(g))


class ParameterProblem(BinaryProblem):
    """Bit strings made of consecutive parameters (``bits`` wide each, or one width per parameter).

    IGC division follows Section III-A for encoded parameters: ``M`` is the number
    of *parameters* whose bits differ between the parents, ``N = 2**floor(log2(M+1)) - 1``
    (optionally capped), and the ``N - 1`` cut points are drawn from the ``M - 1``
    candidate cut points that separate those parameters. A parameter is never split.

    Example:
        >>> ParameterProblem(3, [30, 5, 5]).n_bits
        40
    """

    def __init__(self, n_params: int, bits: int | Sequence[int], version: str = "params-v1") -> None:
        widths = np.full(n_params, bits) if isinstance(bits, int) else np.asarray(bits, dtype=int)
        if len(widths) != n_params or np.any(widths < 1):
            raise ValueError("bits must be a positive int or one positive width per parameter")
        super().__init__(int(widths.sum()), version)
        self.n_params, self.widths = n_params, widths
        self._owner = np.repeat(np.arange(n_params), widths)  # parameter index of every bit

    def divide(
        self,
        p1: Genome,
        p2: Genome,
        diff: npt.NDArray[np.intp],
        max_segments: int | None,
        rng: np.random.Generator,
    ) -> list[npt.NDArray[np.intp]]:
        param = self._owner[diff]
        starts = np.flatnonzero(np.r_[True, param[1:] != param[:-1]])  # first differing bit of each parameter
        m = len(starts)
        n = paper_n_segments(m)
        if max_segments is not None:
            n = min(n, max_segments)
        if n < 2:
            return []
        cuts = starts[np.sort(rng.choice(np.arange(1, m), n - 1, replace=False))]
        return list(np.split(diff, cuts))
