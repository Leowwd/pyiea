"""Re-run the IEA paper's single-objective experiments with pyiea and compare with the printed numbers.

    python reproduce/iea_paper.py                      # every experiment, 30 runs each
    python reproduce/iea_paper.py table5 --runs 5      # a subset

Experiments (Ho, Shu, Chen 2004):
  table5  Table V: f1..f12, D = 10, N_pop = 10, ps = 0.2, pc = 0.8, pm = 0.05, N_eval = 10 000, 30 runs
  table6  Table VI: the same settings with D = 100
  table3  Table III: f1, D = 100, 5 bits per parameter, N_pop = 30, N_eval = 12 000, 10 runs
  table3b Section V-A: f1, D = 100, 14 bits, run until 121.598 is reached (paper: 58 156 evaluations on average)
  iga3    Test data of Ho, Chen, Huang (2004, IGA), eq. (3) = f1 with D = 100, 10 bits, N = 63 segments,
          N_pop = 50, ps = 0.04, pc = 0.5, pm = 0.01, 58 100 evaluations, 30 runs. The paper's number
          (113.733) is for its GA with OAX, not for IEA, so it is a reference point, not a target.

Parameters are Gray-coded (``PaperBenchmark.encoder()``; the paper does not state its encoding) and
every IGC cuts only between parameters (``RealEncoder.problem()``). pyiea counts unique objective
calls, so cached re-evaluations are free; the paper does not say whether it counted them.
Results go to reproduce/results/<experiment>.md and .json.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import pyiea
from pyiea.benchmarks import PAPER_FUNCTIONS, PaperBenchmark

OUT = Path(__file__).parent / "results"

# (mean, min, max) printed in Table V (D = 10) and Table VI (D = 100), IEA column
TABLE5 = {
    "f1": (12.116, 12.077, 12.150),
    "f2": (15.32, 14.34, 16.76),
    "f3": (5.13, 2, 11),
    "f4": (15.42, 12.28, 19.68),
    "f5": (0.0003, 2e-4, 8e-4),
    "f6": (14.60, 12.46, 16.36),
    "f7": (0.054, 0.047, 0.067),
    "f8": (1.00, 0.09, 1.73),
    "f9": (667.4, 512.0, 1169.1),
    "f10": (116.44, 3.70, 210.79),
    "f11": (0.338, 0, 1),
    "f12": (0.999, 0.9994, 0.9996),
}
TABLE6 = {
    "f1": (120.44, 120.04, 120.77),
    "f2": (153.15, 149.23, 164.65),
    "f3": (621, 535, 824),
    "f4": (213.46, 181.82, 257.95),
    "f5": (1.60, 1.06, 3.13),
    "f6": (131.31, 122.33, 141.66),
    "f7": (0.65, 0.55, 0.82),
    "f8": (3.69, 2.71, 4.81),
    "f9": (8011, 6743, 9479),
    "f10": (2081, 879, 2803),
    "f11": (43.94, 36, 51),
    "f12": (32.86, 14.10, 74.85),
}


def run_one(
    name: str, dims: int, bits: int | None, cfg: dict, max_calls: int, seed: int, target: float | None = None
) -> dict:
    bench = PaperBenchmark(name, dims)
    enc = bench.encoder(bits)
    config = pyiea.IEAConfig(**cfg, target=None if target is None else -target if bench.maximize else target)
    t0 = time.perf_counter()
    res = pyiea.optimize(enc.wrap(bench), problem=enc.problem(), config=config, max_calls=max_calls, seed=seed)
    return {
        "function": name,
        "seed": seed,
        "value": bench.paper_value(res.best),
        "calls": res.accounting["objective_calls"],
        "cache_hits": res.accounting["cache_hits"],
        "proposals": res.accounting["proposals"],
        "stop_reason": res.stop_reason,
        "seconds": time.perf_counter() - t0,
    }


def run_many(jobs: list[tuple], workers: int) -> list[dict]:
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(run_one, *zip(*jobs, strict=True)))


def summarize(rows: list[dict]) -> dict:
    v = np.array([r["value"] for r in rows])
    return {
        "mean": float(v.mean()),
        "min": float(v.min()),
        "max": float(v.max()),
        "std": float(v.std()),
        "calls_mean": float(np.mean([r["calls"] for r in rows])),
        "proposals_mean": float(np.mean([r["proposals"] for r in rows])),
        "runs": len(rows),
        "stop_reasons": sorted({r["stop_reason"] for r in rows}),
    }


PAPER_CFG = {"pop_size": 10, "ps": 0.2, "pc": 0.8, "pm": 0.05}


def table(dims: int, printed: dict, runs: int, workers: int) -> tuple[str, dict]:
    jobs = [(f, dims, None, PAPER_CFG, 10_000, s) for f in PAPER_FUNCTIONS for s in range(runs)]
    rows = run_many(jobs, workers)
    lines = [
        "| f | optimum | pyiea mean [min, max] | paper IEA mean [min, max] | calls | proposals | stop |",
        "|---|---|---|---|---|---|---|",
    ]
    out = {}
    for f in PAPER_FUNCTIONS:
        s = summarize([r for r in rows if r["function"] == f])
        out[f] = s
        p = printed[f]
        lines.append(
            f"| {f} | {PaperBenchmark(f, dims).optimum:g} | {s['mean']:.4g} [{s['min']:.4g}, {s['max']:.4g}] | "
            f"{p[0]:g} [{p[1]:g}, {p[2]:g}] | {s['calls_mean']:.0f} | {s['proposals_mean']:.0f} | "
            f"{', '.join(s['stop_reasons'])} |"
        )
    return "\n".join(lines), out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="*", default=["table5", "table6", "table3", "table3b", "iga3"])
    ap.add_argument("--runs", type=int, default=None, help="override the paper's number of runs")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for exp in a.experiments:
        t0 = time.perf_counter()
        if exp in ("table5", "table6"):
            dims, printed = (10, TABLE5) if exp == "table5" else (100, TABLE6)
            md, data = table(dims, printed, a.runs or 30, a.workers)
            title = f"Table {'V' if dims == 10 else 'VI'}: D = {dims}, N_pop = 10, 10 000 evaluations"
        elif exp == "table3":
            rows = run_many(
                [
                    ("f1", 100, 5, {"pop_size": 30, "ps": 0.2, "pc": 0.8, "pm": 0.05}, 12_000, s)
                    for s in range(a.runs or 10)
                ],
                a.workers,
            )
            data = summarize(rows)
            title = "Table III: f1, D = 100, 5 bits per parameter (500-bit chromosome), 12 000 evaluations"
            md = (
                f"pyiea mean {data['mean']:.3f} [{data['min']:.3f}, {data['max']:.3f}] over {data['runs']} runs; "
                f"paper IEA 120.890 (10 runs). Global optimum 121.598."
            )
        elif exp == "table3b":
            rows = run_many(
                [
                    ("f1", 100, 14, {"pop_size": 30, "ps": 0.2, "pc": 0.8, "pm": 0.05}, 400_000, s, 121.5975)
                    for s in range(a.runs or 10)
                ],
                a.workers,
            )
            data = summarize(rows)
            reached = sum(r["value"] >= 121.5975 for r in rows)
            title = "Section V-A: f1, D = 100, 14 bits (1400-bit chromosome), until 121.598 is reached"
            md = (
                f"reached 121.598 in {reached}/{len(rows)} runs; mean evaluations {data['calls_mean']:.0f} "
                f"(paper: 58 156). Values: mean {data['mean']:.4f} [{data['min']:.4f}, {data['max']:.4f}]."
            )
        elif exp == "iga3":
            cfg = {"pop_size": 50, "ps": 0.04, "pc": 0.5, "pm": 0.01, "max_segments": 63}
            rows = run_many([("f1", 100, 10, cfg, 58_100, s) for s in range(a.runs or 30)], a.workers)
            data = summarize(rows)
            title = "IGA test data, eq. (3): f1, D = 100, 10 bits, N = 63, N_pop = 50, 58 100 evaluations"
            md = (
                f"pyiea IEA mean {data['mean']:.3f} [{data['min']:.3f}, {data['max']:.3f}], std {data['std']:.3f} over "
                f"{data['runs']} runs. IGA paper, Table II: GA with OAX 113.733, GA with 1-cut-point crossover 98.466. "
                f"Global optimum 121.598."
            )
        else:
            raise SystemExit(f"unknown experiment {exp}")
        text = f"## {title}\n\n{md}\n\n_{time.perf_counter() - t0:.0f} s, pyiea {pyiea.__version__}_\n"
        (OUT / f"{exp}.md").write_text(text)
        (OUT / f"{exp}.json").write_text(json.dumps(data, indent=2))
        print(text, flush=True)


if __name__ == "__main__":
    main()
