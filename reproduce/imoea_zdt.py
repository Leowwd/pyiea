"""Re-run the IMOEA paper's ZDT experiment with m = 63 parameters (Section VI-B) with pyiea.

    python reproduce/imoea_zdt.py run                        # 6 problems x 30 runs, every method and encoding
    python reproduce/imoea_zdt.py run --runs 2 --problems zdt1 zdt4
    python reproduce/imoea_zdt.py report                     # rebuild the tables from finished runs

Problems: ZDT1-ZDT6 of Zitzler, Deb & Thiele (2000) (``pyiea.benchmarks.ZDT``), extended to m = 63
parameters. Every parameter has 30 bits; ZDT5 is defined on bits (x1: 30 bits, x2..x63: 5 bits each).

Methods, all with N_eval = 25 000 objective calls:
  imoea   IMOEA, Table VIII: N_pop 30, N_Emax 30, ps 0.2, pc 0.6, pm 0.01, N = m = 63 segments.
          ``ZDT.problem()`` cuts only between parameters.
  nsga2   baseline: pyiea.baselines.nsga2 with N_pop 100 and pc 0.8 (the [26] settings quoted by the
          paper) and pm = 1 / n_bits. The paper quotes pm = 0.1, which as a per-bit rate would flip
          about 189 of 1 890 bits per child; uniform crossover.
  nsga2_pm0.1      baseline check: the same with the quoted pm = 0.1 read literally, per bit.
  nsga2_pm0.1var   baseline check: the quoted pm = 0.1 read per parameter, i.e. 0.1 / 30 per bit (about
                   0.1 flipped bits per parameter). ZDT5: 0.1 / 5 per bit.
  The two checks test whether the paper's weaker NSGA-II (Figs. 9 and 11) comes from its settings.
Encodings (ablation; the paper does not state one): ``binary`` and ``gray`` for ZDT1-4 and 6, and
``bits`` for ZDT5.

Metrics per run, on the distinct non-dominated vectors the run evaluated (its archive):
  HV   pyiea.hypervolume_2d with one fixed reference point per problem (REF), shared by every run
  IGD  mean distance from the analytic Pareto-optimal front to the run's front
  GD   mean distance from the run's front to the analytic front
  C    cover metric (eq. 8), run r of A against run r of B
Results go to reproduce/results/imoea_zdt.{md,json} (the json includes the merged fronts of Figs. 6-11);
every run's front goes to imoea_zdt_fronts.json.gz, which is not committed (the runs are seeded and
reproducible).
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from _moo import RunCache, cover_paired, default_workers, distinct_front, fmt_quartiles, merged_front, quartiles

import pyiea
from pyiea import baselines
from pyiea.benchmarks import ZDT, ZDT_PROBLEMS

HERE = Path(__file__).parent
OUT = HERE / "results"
RUNS_FILE = HERE / "data" / "imoea_zdt_runs.jsonl"

M, BITS, N_EVAL = 63, 30, 25_000
IMOEA_CFG = {"pop_size": 30, "elite_capacity": 30, "ps": 0.2, "pc": 0.6, "pm": 0.01, "max_segments": M}
NSGA2_CFG = {"pop_size": 100, "pc": 0.8}  # pm = 1 / n_bits (the nsga2 default)
# pm of the NSGA-II variants; None is the nsga2 default 1 / n_bits. ZDT5's x2..x63 have 5 bits, not BITS.
NSGA2_PM: dict[str, Any] = {
    "nsga2": None,
    "nsga2_pm0.1": lambda problem: 0.1,
    "nsga2_pm0.1var": lambda problem: 0.1 / (5 if problem == "zdt5" else BITS),
}
METHODS = ("imoea", *NSGA2_PM)
# Hypervolume reference points: about the objective values of a uniformly random genome, so that any front
# better than random sampling has a positive hypervolume (engineering choice, fixed before the runs).
REF = {
    "zdt1": (1.1, 6.0),
    "zdt2": (1.1, 6.0),
    "zdt3": (1.1, 6.0),
    "zdt4": (1.1, 1200.0),
    "zdt5": (32.0, 280.0),
    "zdt6": (1.1, 9.0),
}
# Figs. 6-11, read off the plots (merged fronts of 30 runs)
PAPER = {
    "zdt1": "IMOEA very close to the front (about 0.05 above); SPEA2/NSGA2 about 0.1-0.2 above",
    "zdt2": "IMOEA close to the front (about 0.1 above); the others 0.3-0.5 above",
    "zdt3": "IMOEA, SPEA2 and NSGA2 all well distributed over the five pieces",
    "zdt4": "only IMOEA near the front: f2 about 50 for f1 in 0-0.5; the others f2 >= 140",
    "zdt5": "IMOEA best; plotted reference curve looks like the m = 11 front (10/f1), not 62/f1",
    "zdt6": "IMOEA f2 about 2.1-2.6 for f1 in 0.28-1; NSGA2 about 4-4.3; the others >= 4",
}


def encodings(problem: str) -> tuple[str, ...]:
    return ("bits",) if problem == "zdt5" else ("binary", "gray")


@cache
def true_front(problem: str) -> np.ndarray:
    return ZDT(problem, m=M).pareto_front(200 if problem == "zdt3" else 1000)


def run_key(record: dict[str, Any]) -> tuple[str, str, str, int, int]:
    return record["problem"], record["encoding"], record["method"], record["seed"], record["budget"]


def run_one(problem: str, encoding: str, method: str, seed: int) -> dict[str, Any]:
    z, gray = ZDT(problem, m=M, bits=BITS), encoding == "gray"
    space = z.problem(gray)
    ev = pyiea.Evaluator(z.objective(gray), 2, space.version, max_calls=N_EVAL)
    t0 = time.perf_counter()
    if method == "imoea":
        res = pyiea.IMOEA(space, ev, pyiea.IMOEAConfig(**IMOEA_CFG), seed).optimize()
    elif method in NSGA2_PM:
        pm = NSGA2_PM[method]
        res = baselines.nsga2(space, ev, seed, **NSGA2_CFG, pm=None if pm is None else pm(problem))
    else:
        raise ValueError(f"unknown method {method!r}")
    seconds = time.perf_counter() - t0
    front = distinct_front(y for _, y in res.archive)
    ref_front = true_front(problem)
    return {
        "problem": problem,
        "encoding": encoding,
        "method": method,
        "seed": seed,
        "budget": N_EVAL,
        "calls": res.accounting["objective_calls"],
        "seconds": seconds,
        "generations": res.generations,
        "stop_reason": res.stop_reason,
        "hv": pyiea.hypervolume_2d(front, REF[problem]),
        "igd": pyiea.igd(front, ref_front),
        "gd": pyiea.igd(ref_front, front),
        "front": front.tolist(),
        "working_front": distinct_front(y for _, y in res.front).tolist(),
    }


def jobs(problems: list[str], methods: list[str], runs: int) -> list[tuple[tuple, tuple]]:
    out = [
        ((p, e, m, s, N_EVAL), (p, e, m, s))
        for p in problems
        for e in encodings(p)
        for m in methods
        for s in range(runs)
    ]
    return sorted(out, key=lambda j: j[1][2] != "imoea")  # IMOEA runs are the slower ones: start them first


def summarize(cache_: RunCache, problems: list[str], methods: list[str], runs: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "settings": {
            "m": M,
            "bits": BITS,
            "n_eval": N_EVAL,
            "imoea": IMOEA_CFG,
            "nsga2": {**NSGA2_CFG, "pm": "1/n_bits"},
            "nsga2_pm0.1": {**NSGA2_CFG, "pm": 0.1},
            "nsga2_pm0.1var": {**NSGA2_CFG, "pm": "0.1/bits per parameter"},
            "hv_ref": REF,
        },
        "cases": {},
    }
    fronts: dict[str, Any] = {}
    for p in problems:
        for e in encodings(p):
            case = f"{p}/{e}"
            rows = {m: [cache_.records.get((p, e, m, s, N_EVAL)) for s in range(runs)] for m in methods}
            rows = {m: rs for m, rs in rows.items() if all(r is not None for r in rs)}
            if not rows:
                continue
            info: dict[str, Any] = {"runs": runs, "paper": PAPER[p], "methods": {}, "cover": {}}
            for m, rs in rows.items():
                merged = merged_front([r["front"] for r in rs])
                info["methods"][m] = {
                    **{k: quartiles([r[k] for r in rs]) for k in ("hv", "igd", "gd")},
                    "calls_mean": float(np.mean([r["calls"] for r in rs])),
                    "seconds_mean": float(np.mean([r["seconds"] for r in rs])),
                    "stop_reasons": sorted({r["stop_reason"] for r in rs}),
                    "merged": {
                        "points": len(merged),
                        "f1": [float(merged[:, 0].min()), float(merged[:, 0].max())],
                        "f2": [float(merged[:, 1].min()), float(np.median(merged[:, 1])), float(merged[:, 1].max())],
                        "gd": pyiea.igd(true_front(p), merged),
                        "front": np.round(merged, 6).tolist(),  # plot data for Figs. 6-11
                    },
                }
                fronts.setdefault(case, {})[m] = {"runs": [r["front"] for r in rs], "merged": merged.tolist()}
            for b in (m for m in NSGA2_PM if m in rows and "imoea" in rows):
                fa, fb = fronts[case]["imoea"]["runs"], fronts[case][b]["runs"]
                info["cover"][f"C(imoea,{b})"] = quartiles(cover_paired(fa, fb))
                info["cover"][f"C({b},imoea)"] = quartiles(cover_paired(fb, fa))
            summary["cases"][case] = info
        if {f"{p}/binary", f"{p}/gray"} <= fronts.keys() and all(
            "imoea" in fronts[f"{p}/{e}"] for e in ("binary", "gray")
        ):
            fg, fb = fronts[f"{p}/gray"]["imoea"]["runs"], fronts[f"{p}/binary"]["imoea"]["runs"]
            summary["cases"][f"{p}/gray"]["cover"]["C(imoea gray,imoea binary)"] = quartiles(cover_paired(fg, fb))
            summary["cases"][f"{p}/gray"]["cover"]["C(imoea binary,imoea gray)"] = quartiles(cover_paired(fb, fg))
    return {"summary": summary, "fronts": fronts}


def markdown(summary: dict[str, Any]) -> str:
    s = summary["settings"]
    L = [
        "## IMOEA paper, Section VI-B: ZDT1-ZDT6 with m = 63 parameters",
        "",
        f"N_eval = {s['n_eval']:,} per run; {s['bits']} bits per parameter (ZDT5: x1 30 bits, x2..x63 5 bits). "
        "Every value is median [first quartile, third quartile] over the runs. HV: larger is better (reference "
        "point per problem in the json). IGD and GD: smaller is better. C(A, B) (eq. 8): run r of A against run r "
        'of B. "merged" is the non-dominated union of all runs, as in Figs. 6-11.',
        "",
    ]
    for case, info in summary["cases"].items():
        L += [f"### {case} ({info['runs']} runs)", "", f"Paper (Figs. 6-11): {info['paper']}.", ""]
        L += [
            "| method | HV | IGD | GD | merged: points, f1 range, f2 min / median / max, GD | calls | s/run |",
            "|---|---|---|---|---|---|---|",
        ]
        for m, r in info["methods"].items():
            g = r["merged"]
            L.append(
                f"| {m} | {fmt_quartiles(r['hv'], '.4g')} | {fmt_quartiles(r['igd'], '.3g')} | "
                f"{fmt_quartiles(r['gd'], '.3g')} | {g['points']}, {g['f1'][0]:.3g}-{g['f1'][1]:.3g}, "
                f"{g['f2'][0]:.3g} / {g['f2'][1]:.3g} / {g['f2'][2]:.3g}, {g['gd']:.3g} | {r['calls_mean']:.0f} | "
                f"{r['seconds_mean']:.1f} |"
            )
        if info["cover"]:
            L += ["", "| cover | value |", "|---|---|"]
            L += [f"| {k} | {fmt_quartiles(q)} |" for k, q in info["cover"].items()]
        L.append("")
    L.append(f"_pyiea {pyiea.__version__}_")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "report"])
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--problems", nargs="*", default=list(ZDT_PROBLEMS), choices=ZDT_PROBLEMS)
    ap.add_argument("--methods", nargs="*", default=list(METHODS), choices=METHODS)
    a = ap.parse_args()
    runs_cache = RunCache(RUNS_FILE, run_key)
    if a.command == "run":
        runs_cache.run(jobs(a.problems, a.methods, a.runs), run_one, a.workers)
    out = summarize(runs_cache, a.problems, a.methods, a.runs)
    OUT.mkdir(exist_ok=True)
    (OUT / "imoea_zdt.json").write_text(json.dumps(out["summary"], indent=2))
    with gzip.open(OUT / "imoea_zdt_fronts.json.gz", "wt") as fh:
        json.dump(out["fronts"], fh)
    md = markdown(out["summary"])
    (OUT / "imoea_zdt.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
