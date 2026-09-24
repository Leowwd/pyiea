"""Re-run Fig. 3 of the IEA paper (Section V-B): dist(D) of IEA and of elitist GAs with conventional crossover.

    python reproduce/iea_fig3.py run                          # f1..f12, D = 10, 20, ..., 100, 30 runs, 4 methods
    python reproduce/iea_fig3.py run --runs 3 --functions f1 f9 --dims 10 100
    python reproduce/iea_fig3.py report                       # rebuild the tables from finished runs

Settings (Section V-B, for all compared EAs): N_pop 10, ps 0.2, pc 0.8, pm 0.05, N_eval 10 000, 30 runs.
Each parameter has 10 bits (f9 and f12: 24 bits) and is Gray-coded (``PaperBenchmark.encoder()``).

Methods, with the same evaluator, budget, encoding and initial population for a given seed:
  iea    IEA (``pyiea.IEA``, IGC recombination)
  oega   baseline: ``baselines.elitist_ga(crossover="one_point")``
  tega   baseline: ``baselines.elitist_ga(crossover="two_point")``
  uega   baseline: ``baselines.elitist_ga(crossover="uniform")``
  oega_1L, tega_1L, uega_1L
         ablation (not the paper's setting): the same GAs with pm = 1 / n_bits instead of 0.05, run at
         D = 10, 50, 100 only. The paper gives one pm for all compared EAs; as a per-bit rate, 0.05 flips
         about 5% of every GA child before it is ever evaluated, while IGC evaluates its children first.
BLX-alpha GA, OGA and BOA from the paper are not run.

dist(D) = |f_opt(D) - f_best(D)| / D (eq. 6), averaged over the runs. For f2 the optimum is only
"approximately 2D" (Table IV), so its dist(D) is approximate as well.
Results go to reproduce/results/iea_fig3.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from _moo import RunCache, default_workers

import pyiea
from pyiea import baselines
from pyiea.benchmarks import PAPER_FUNCTIONS, PaperBenchmark

HERE = Path(__file__).parent
OUT = HERE / "results"
RUNS_FILE = HERE / "data" / "iea_fig3_runs.jsonl"

N_EVAL = 10_000
DIMS = tuple(range(10, 101, 10))
SETTINGS = {"pop_size": 10, "ps": 0.2, "pc": 0.8, "pm": 0.05}
# method: (crossover, pm); crossover None is IEA, pm None is the paper's 0.05, "1/L" is 1 / n_bits
METHODS: dict[str, tuple[str | None, str | None]] = {
    "iea": (None, None),
    "oega": ("one_point", None),
    "tega": ("two_point", None),
    "uega": ("uniform", None),
    "oega_1L": ("one_point", "1/L"),
    "tega_1L": ("two_point", "1/L"),
    "uega_1L": ("uniform", "1/L"),
}
ABLATION_DIMS = (10, 50, 100)
# Fig. 3, read off the plots: (IEA at D = 10, IEA at D = 100, OEGA/TEGA/UEGA at D = 10, at D = 100)
PAPER = {
    "f1": ("0.005", "0.0065", "0.003-0.005", "0.038-0.039"),
    "f2": ("0.47", "0.47", "0.35-0.48", "1.05-1.10"),
    "f3": ("~0", "~5", "~0", "550-600"),
    "f4": ("1.5", "2.1", "0.7-1.1", "7.9-8.1"),
    "f5": ("~0", "0.016", "~0", "1.45-1.55"),
    "f6": ("0.39", "0.53", "0.27-0.33", "1.08-1.10"),
    "f7": ("0.005", "0.011", "0.005", "0.31-0.34"),
    "f8": ("0.10", "0.037", "0.02-0.075", "0.17"),
    "f9": ("67", "80", "85-186", "242-247"),
    "f10": ("~10", "~20", "5-10", "900-975"),
    "f11": ("0.03", "0.44", "0-0.03", "1.55-1.70"),
    "f12": ("0.1", "0.33", "~0.1", "5.1-10.5"),
}


def run_key(record: dict[str, Any]) -> tuple[str, int, str, int, int]:
    return record["function"], record["dims"], record["method"], record["seed"], record["budget"]


def run_one(function: str, dims: int, method: str, seed: int) -> dict[str, Any]:
    bench = PaperBenchmark(function, dims)
    enc = bench.encoder()
    ev = pyiea.Evaluator(enc.wrap(bench), 1, enc.problem().version, max_calls=N_EVAL)
    t0 = time.perf_counter()
    crossover, pm = METHODS[method]
    if crossover is None:
        res = pyiea.IEA(enc.problem(), ev, pyiea.IEAConfig(**SETTINGS), seed).optimize()
    else:
        cfg = {**SETTINGS, "pm": 1.0 / enc.n_bits} if pm == "1/L" else SETTINGS
        res = baselines.elitist_ga(enc.problem(), ev, seed, crossover=crossover, **cfg)
    assert res.best is not None
    value = bench.paper_value(res.best)
    return {
        "function": function,
        "dims": dims,
        "method": method,
        "seed": seed,
        "budget": N_EVAL,
        "value": value,
        "dist": abs(bench.optimum - value) / dims,
        "calls": res.accounting["objective_calls"],
        "seconds": time.perf_counter() - t0,
        "stop_reason": res.stop_reason,
    }


def jobs(functions: list[str], dims: list[int], methods: list[str], runs: int) -> list[tuple[tuple, tuple]]:
    out = [
        ((f, d, m, s, N_EVAL), (f, d, m, s))
        for f in functions
        for d in dims
        for m in methods
        if not m.endswith("_1L") or d in ABLATION_DIMS
        for s in range(runs)
    ]
    return sorted(out, key=lambda j: -j[1][1])  # largest D first: the slowest runs start early


def summarize(cache: RunCache, functions: list[str], dims: list[int], methods: list[str], runs: int) -> dict:
    table: dict[str, Any] = {}
    for f in functions:
        for d in dims:
            for m in methods:
                rows = [cache.records.get((f, d, m, s, N_EVAL)) for s in range(runs)]
                if any(r is None for r in rows):
                    continue
                dist = np.array([r["dist"] for r in rows])
                table.setdefault(f, {}).setdefault(str(d), {})[m] = {
                    "dist_mean": float(dist.mean()),
                    "dist_std": float(dist.std()),
                    "calls_mean": float(np.mean([r["calls"] for r in rows])),
                    "seconds_mean": float(np.mean([r["seconds"] for r in rows])),
                    "stop_reasons": sorted({r["stop_reason"] for r in rows}),
                }
    return {"settings": {**SETTINGS, "n_eval": N_EVAL, "runs": runs}, "dist": table}


def _table(by_d: dict[str, dict[str, Any]], methods: list[str]) -> tuple[list[str], list[int]]:
    """Rows of mean dist(D) for ``methods`` (bold: the row minimum) and, per D, whether IEA beats every GA."""
    lines = ["| D | " + " | ".join(methods) + " |", "|---" * (len(methods) + 1) + "|"]
    wins = []
    for d, cells in by_d.items():
        if not all(m in cells for m in methods):
            continue
        means = {m: cells[m]["dist_mean"] for m in methods}
        best = min(means.values())
        lines.append(
            f"| {d} | "
            + " | ".join(
                (f"**{means[m]:.4g}**" if means[m] == best else f"{means[m]:.4g}") + f" ± {cells[m]['dist_std']:.2g}"
                for m in methods
            )
            + " |"
        )
        wins.append(int(means["iea"] < min(v for m, v in means.items() if m != "iea")))
    return lines, wins


def markdown(summary: dict) -> str:
    runs = summary["settings"]["runs"]
    main_methods = ["iea", "oega", "tega", "uega"]
    ablation = ["iea", "oega_1L", "tega_1L", "uega_1L"]
    L = [
        "## IEA paper, Fig. 3: dist(D) = |f_opt - f_best| / D, mean ± std over runs",
        "",
        f"N_pop 10, ps 0.2, pc 0.8, pm 0.05, N_eval 10 000, {runs} runs per cell; the same seeds, encoding and "
        "initial population for every method. OEGA/TEGA/UEGA are pyiea baselines (`baselines.elitist_ga`). "
        "**Bold**: the smallest mean in the row. The `_1L` columns are an ablation with pm = 1/n_bits for the GAs, "
        "not the paper's setting.",
        "",
    ]
    wins: dict[str, tuple[list[int], list[int]]] = {}
    for f, by_d in summary["dist"].items():
        main_lines, main_wins = _table(by_d, main_methods)
        abl_lines, abl_wins = _table(by_d, ablation)
        wins[f] = (main_wins, abl_wins)
        p = PAPER[f]
        L += [f"### {f}", "", *main_lines, ""]
        if abl_wins:
            L += ["Ablation, GA pm = 1/n_bits:", "", *abl_lines, ""]
        L += [
            f"Paper (read off Fig. 3): IEA {p[0]} at D = 10 and {p[1]} at D = 100; OEGA/TEGA/UEGA {p[2]} and {p[3]}.",
            "",
        ]
    L += [
        "### Summary: D values at which IEA's mean dist(D) is below all three GAs",
        "",
        "| f | GAs with pm = 0.05 (paper) | GAs with pm = 1/n_bits (ablation) |",
        "|---|---|---|",
    ]
    L += [f"| {f} | {sum(a)}/{len(a)} | {sum(b)}/{len(b)} |" for f, (a, b) in wins.items()]
    L += ["", f"_pyiea {pyiea.__version__}_"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "report"])
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--functions", nargs="*", default=list(PAPER_FUNCTIONS), choices=PAPER_FUNCTIONS)
    ap.add_argument("--dims", nargs="*", type=int, default=list(DIMS))
    ap.add_argument("--methods", nargs="*", default=list(METHODS), choices=list(METHODS))
    a = ap.parse_args()
    cache = RunCache(RUNS_FILE, run_key)
    if a.command == "run":
        cache.run(jobs(a.functions, a.dims, a.methods, a.runs), run_one, a.workers)
    summary = summarize(cache, a.functions, a.dims, a.methods, a.runs)
    OUT.mkdir(exist_ok=True)
    (OUT / "iea_fig3.json").write_text(json.dumps(summary, indent=2))
    md = markdown(summary)
    (OUT / "iea_fig3.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
