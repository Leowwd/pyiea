"""Two-level orthogonal arrays (Ho, Shu, Chen 2004, Section II-A)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["generate_oa", "oa_rows"]


def oa_rows(n_factors: int) -> int:
    """Number of combinations ``n = 2**ceil(log2(N + 1))`` for ``N`` two-level factors.

    Raises:
        ValueError: if ``n_factors < 1``.
    """
    if n_factors < 1:
        raise ValueError(f"n_factors must be >= 1, got {n_factors}")
    return 1 << n_factors.bit_length()  # == 2**ceil(log2(N+1)) for integers N >= 1


def generate_oa(n_factors: int) -> npt.NDArray[np.uint8]:
    """Algorithm ``Generate_OA(OA, N)`` of the paper, transcribed line by line.

    Returns:
        An ``(n, N)`` array of levels in ``{1, 2}``: the first ``N`` columns of
        ``L_n(2^(n-1))``. Every column is balanced, every pair of columns contains
        each level combination equally often, and row 0 is all level 1.
    """
    n = oa_rows(n_factors)
    oa = np.empty((n, n_factors), dtype=np.uint8)
    for i in range(1, n + 1):
        for j in range(1, n_factors + 1):
            level, k, mask = 0, j, n // 2
            while k > 0:
                if k % 2 and ((i - 1) & mask) != 0:
                    level = (level + 1) % 2
                k //= 2
                mask //= 2
            oa[i - 1, j - 1] = level + 1
    return oa
