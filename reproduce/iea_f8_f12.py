"""Investigate the open f8 (Ackley) and f12 (Griewank) differences from Tables V and VI of the IEA paper.

    python reproduce/iea_f8_f12.py run          # every hypothesis, D = 10 and 100, 30 runs
    python reproduce/iea_f8_f12.py report

pyiea's f8 and f12 are the formulas printed in Table IV (checked against the page image), and
``benchmarks.PaperBenchmark`` keeps them. This script tries alternatives that could explain the
paper's numbers; none of them changes the library.

  f8, Table V (D = 10): paper minimum 0.09, below the floor 0.1626 of a 10-bit grid with
      x = low + k (high - low) / (2^b - 1), whose values closest to 0 are +-0.0293.
      Hypotheses: the grid divides by 2^b (then x = 0 is on the grid), plain binary instead of Gray,
      more bits per parameter.
  f12, Table V (D = 10): paper values 0.999 [0.9994, 0.9996], far above pyiea's 0.1076.
      Hypothesis: the product is taken over cos(x_i) / sqrt(i) instead of cos(x_i / sqrt(i)). Its minimum,
      at x = 0, is 1 - 1/sqrt(D!) = 0.999475 for D = 10.

IEA settings as in Tables V and VI: N_pop 10, ps 0.2, pc 0.8, pm 0.05, N_eval 10 000, 30 runs.
Results go to reproduce/results/iea_f8_f12.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from _moo import RunCache, default_workers

import pyiea
from pyiea.benchmarks import PaperBenchmark

HERE = Path(__file__).parent
OUT = HERE / "results"
RUNS_FILE = HERE / "data" / "iea_f8_f12_runs.jsonl"

N_EVAL, RUNS, DIMS = 10_000, 30, (10, 100)
SETTINGS = {"pop_size": 10, "ps": 0.2, "pc": 0.8, "pm": 0.05}
# Tables V and VI, IEA column: (mean, min, max)
PAPER = {("f8", 10): (1.00, 0.09, 1.73), ("f8", 100): (3.69, 2.71, 4.81),
         ("f12", 10): (0.999, 0.9994, 0.9996), ("f12", 100): (32.86, 14.10, 74.85)}  # fmt: skip


class PowerOfTwoEncoder(pyiea.RealEncoder):
    """``x = low + k (high - low) / 2^b``: the grid includes the midpoint but not ``high``."""

    def decode(self, g: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
        bits = np.asarray(g, dtype=np.uint8).reshape(self.n_params, self.bits)
        if self.gray:
            bits = np.bitwise_xor.accumulate(bits, axis=1)
        ints = bits.astype(float) @ (2.0 ** np.arange(self.bits - 1, -1, -1))
        out: npt.NDArray[np.float64] = self.low + ints / 2.0**self.bits * (self.high - self.low)
        return out


def griewank_cos_over_sqrt(x: npt.NDArray[np.float64]) -> float:
    """f12 with ``prod cos(x_i) / sqrt(i)`` instead of Table IV's ``prod cos(x_i / sqrt(i))``."""
    i = np.arange(1, len(x) + 1)
    return float(np.sum(x**2) / 4000 - np.prod(np.cos(x) / np.sqrt(i)) + 1)


# name: (function, formula, encoding, grid divisor, bits)
HYPOTHESES: dict[str, tuple[str, str, str, str, int]] = {
    "f8 pyiea (Gray, 2^b-1, 10 bits)": ("f8", "table IV", "gray", "2^b-1", 10),
    "f8 binary, 2^b-1, 10 bits": ("f8", "table IV", "binary", "2^b-1", 10),
    "f8 Gray, 2^b, 10 bits": ("f8", "table IV", "gray", "2^b", 10),
    "f8 binary, 2^b, 10 bits": ("f8", "table IV", "binary", "2^b", 10),
    "f8 Gray, 2^b-1, 16 bits": ("f8", "table IV", "gray", "2^b-1", 16),
    "f8 binary, 2^b-1, 16 bits": ("f8", "table IV", "binary", "2^b-1", 16),
    "f12 pyiea (Table IV, Gray)": ("f12", "table IV", "gray", "2^b-1", 24),
    "f12 Table IV, binary": ("f12", "table IV", "binary", "2^b-1", 24),
    "f12 cos(x)/sqrt(i), Gray": ("f12", "cos(x)/sqrt(i)", "gray", "2^b-1", 24),
    "f12 cos(x)/sqrt(i), binary": ("f12", "cos(x)/sqrt(i)", "binary", "2^b-1", 24),
}


def encoder(bounds: Sequence[tuple[float, float]], bits: int, gray: bool, divisor: str) -> pyiea.RealEncoder:
    cls = PowerOfTwoEncoder if divisor == "2^b" else pyiea.RealEncoder
    return cls(bounds, bits=bits, gray=gray)


def run_key(record: dict[str, Any]) -> tuple[str, int, int]:
    return record["hypothesis"], record["dims"], record["seed"]


def run_one(hypothesis: str, dims: int, seed: int) -> dict[str, Any]:
    fname, formula, enc_name, divisor, bits = HYPOTHESES[hypothesis]
    bench = PaperBenchmark(fname, dims)
    fn = griewank_cos_over_sqrt if formula == "cos(x)/sqrt(i)" else bench
    enc = encoder([bench.bounds] * dims, bits, enc_name == "gray", divisor)
    ev = pyiea.Evaluator(enc.wrap(fn), 1, f"{hypothesis}-{dims}", max_calls=N_EVAL)
    t0 = time.perf_counter()
    res = pyiea.IEA(enc.problem(), ev, pyiea.IEAConfig(**SETTINGS), seed).optimize()
    return {
        "hypothesis": hypothesis,
        "dims": dims,
        "seed": seed,
        "value": res.best,
        "calls": res.accounting["objective_calls"],
        "seconds": time.perf_counter() - t0,
    }


def grid_floor(hypothesis: str, dims: int) -> float:
    """Best value on the grid when every parameter takes the grid value closest to the optimum x = 0."""
    fname, formula, _, divisor, bits = HYPOTHESES[hypothesis]
    bench = PaperBenchmark(fname, dims)
    low, high = bench.bounds
    k = np.arange(2**bits, dtype=float)
    grid = low + k * (high - low) / (2.0**bits if divisor == "2^b" else 2.0**bits - 1)
    x = np.full(dims, grid[np.argmin(np.abs(grid))])
    return griewank_cos_over_sqrt(x) if formula == "cos(x)/sqrt(i)" else bench(x)


def report(cache: RunCache) -> str:
    L = [
        "## IEA paper, f8 and f12: hypotheses for the open differences",
        "",
        f'IEA, N_pop 10, ps 0.2, pc 0.8, pm 0.05, N_eval 10 000, {RUNS} runs per row. "floor" is the value with '
        "every parameter at the grid point closest to the optimum x = 0 (for f8 and f12 this is also the best value "
        "on the grid). Paper: Tables V (D = 10) and VI (D = 100), IEA column.",
        "",
        "| hypothesis | D | floor | mean [min, max] | paper mean [min, max] | calls | s/run |",
        "|---|---|---|---|---|---|---|",
    ]
    data: dict[str, Any] = {}
    for h in HYPOTHESES:
        for d in DIMS:
            rows = [cache.records.get((h, d, s)) for s in range(RUNS)]
            if any(r is None for r in rows):
                continue
            v = np.array([r["value"] for r in rows])
            p = PAPER[(HYPOTHESES[h][0], d)]
            floor = round(grid_floor(h, d), 12) + 0.0  # no "-0" from rounding noise
            data[f"{h} | D={d}"] = {"floor": floor, "mean": float(v.mean()), "min": float(v.min()),
                                    "max": float(v.max()), "std": float(v.std()), "runs": RUNS,
                                    "calls_mean": float(np.mean([r["calls"] for r in rows])),
                                    "seconds_mean": float(np.mean([r["seconds"] for r in rows]))}  # fmt: skip
            L.append(
                f"| {h} | {d} | {floor:.4g} | {v.mean():.4g} [{v.min():.4g}, {v.max():.4g}] | "
                f"{p[0]:g} [{p[1]:g}, {p[2]:g}] | {data[f'{h} | D={d}']['calls_mean']:.0f} | "
                f"{data[f'{h} | D={d}']['seconds_mean']:.1f} |"
            )
    L += [
        "",
        f"f12 with cos(x)/sqrt(i) at D = 10: minimum 1 - 1/sqrt(10!) = {1 - 1 / math.sqrt(math.factorial(10)):.6f}.",
    ]
    L += ["", f"_pyiea {pyiea.__version__}_"]
    (OUT / "iea_f8_f12.json").write_text(json.dumps(data, indent=2))
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "report"])
    ap.add_argument("--workers", type=int, default=default_workers())
    a = ap.parse_args()
    cache = RunCache(RUNS_FILE, run_key)
    if a.command == "run":
        jobs = [((h, d, s), (h, d, s)) for d in sorted(DIMS, reverse=True) for h in HYPOTHESES for s in range(RUNS)]
        cache.run(jobs, run_one, a.workers)
    OUT.mkdir(exist_ok=True)
    md = report(cache)
    (OUT / "iea_f8_f12.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
