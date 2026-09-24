"""Re-run the IMOEA paper's multi-objective 0/1 knapsack experiment (Section VI-A) with pyiea.

    python reproduce/imoea_knapsack.py fetch                  # download the test data, check SHA-256
    python reproduce/imoea_knapsack.py run                    # 9 instances x 30 runs x every method
    python reproduce/imoea_knapsack.py run --runs 3 --instances 250.2 --methods imoea nsga2

Problem (eq. 9): I knapsacks, J items, maximize every knapsack's profit subject to all capacities.
Data: Zitzler & Thiele (1999) ``knapsack.<J>.<I>``. The ETH site is tried first, then a mirror in the
moead-framework/data repository. Only files whose SHA-256 matches the table below are used. The mirror
has no I = 4 files, so an instance without data runs on **seeded eq. (9) data**, reported as
"non-paper data" (``MultiKnapsack.random``). The data files are not committed; see ``fetch``.

Repair (Zitzler & Thiele 1999): items leave in increasing order of max_i p_ij / w_ij until every
constraint holds. It is applied inside the objective (decode only); the genome is never rewritten.

Methods, all with the same budget (Table VIII N_eval), the same objective and the same repair:
  imoea          IMOEA, Table VIII: N_pop 50, N_Emax 50, ps 0.2, pc 0.8, pm 0.01, N = 15 segments
  imoea_unique   the same IMOEA with ``elite_unique_objectives=True`` (engineering option: E keeps one
                 genome per objective vector)
  nsga2          baseline: pyiea.baselines.nsga2, N_pop 50, pc 0.8, pm 0.01, uniform crossover
  random         baseline: uniform random genomes + non-dominated archive
Each method's front is the non-dominated set of every solution it evaluated (the archive), reduced to
distinct objective vectors. Cover metric C(A, B) (eq. 8): fraction of B weakly dominated by A,
computed run by run (run r of A against run r of B, as in "the direct comparisons of each independent
run" of Section VI-A), so 30 values per ordered pair.

Results go to reproduce/results/imoea_knapsack.{md,json}; the json includes the merged 750.2 fronts of
Fig. 5. Every run's front goes to imoea_knapsack_fronts.json.gz, which is not committed (the runs are
seeded and reproducible).
"""

from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from _moo import RunCache, cover_paired, default_workers, distinct_front, fmt_quartiles, merged_front, quartiles

import pyiea
from pyiea import baselines
from pyiea.benchmarks import MultiKnapsack

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "results"
RUN_CACHE = DATA / "imoea_knapsack_runs.jsonl"  # finished runs, so an interrupted `run` can resume

INSTANCES = [(j, i) for i in (2, 3, 4) for j in (250, 500, 750)]
# Table VIII: N_eval per (I, J)
N_EVAL = {
    (2, 250): 75_000,
    (2, 500): 100_000,
    (2, 750): 125_000,
    (3, 250): 100_000,
    (3, 500): 125_000,
    (3, 750): 150_000,
    (4, 250): 125_000,
    (4, 500): 150_000,
    (4, 750): 175_000,
}
ETH = "https://sop.tik.ee.ethz.ch/download/supplementary/testProblemSuite/knapsack.{j}.{i}"
MIRROR = "https://raw.githubusercontent.com/moead-framework/data/master/problem/MOKP/Instances/{j}_{i}.txt"
# SHA-256 of the files served by the mirror on 2026-09-24 (the ETH originals could not be reached to compare)
SHA256 = {
    (250, 2): "1b81f82f51553db5e81e224e1f2e02f59fcd93944ac24796ae6b996d4fb688f2",
    (250, 3): "73e858a4dc07a3c121e10de6aef2a1ee8d569cc4cf9617155e7afdb4c1ac5a35",
    (500, 2): "8695b8a2bf48e65308944efcadbbaeaff98e021a058a584fdeb3fe4fa5b20041",
    (500, 3): "3568b1ce2ab6792b5a453c8f1e502095c9c600aa62b496ce0fef3dc5e29eb5a4",
    (750, 2): "f3dc26439f77fa5a353b00b1ff5e566afc7a59dbaa0f6cb1dec8a100df7734d2",
    (750, 3): "7985a3633fbeb9a487bbf08354e380cb08b31dc35514ec1bd5b05185a7345714",
}
METHODS = ("imoea", "imoea_unique", "nsga2", "random")
IMOEA_CFG = {"pop_size": 50, "elite_capacity": 50, "ps": 0.2, "pc": 0.8, "pm": 0.01, "max_segments": 15}
PAIRS = [("imoea", "nsga2"), ("imoea", "random"), ("imoea_unique", "nsga2"), ("imoea", "imoea_unique")]


def data_path(j: int, i: int) -> Path:
    return DATA / f"knapsack.{j}.{i}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch() -> None:
    DATA.mkdir(exist_ok=True)
    for j, i in INSTANCES:
        dest = data_path(j, i)
        if dest.exists() and SHA256.get((j, i)) == sha256(dest):
            print(f"knapsack.{j}.{i}: present, checksum ok")
            continue
        for url in (ETH.format(j=j, i=i), MIRROR.format(j=j, i=i)):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 - fixed https URLs
                    body = r.read()
            except OSError as e:
                print(f"knapsack.{j}.{i}: {url}: {e}")
                continue
            digest = hashlib.sha256(body).hexdigest()
            if digest != SHA256.get((j, i)):
                print(f"knapsack.{j}.{i}: {url}: unknown SHA-256 {digest}, not used")
                continue
            dest.write_bytes(body)
            print(f"knapsack.{j}.{i}: {url}: ok")
            break
        else:
            print(f"knapsack.{j}.{i}: NOT AVAILABLE; `run` will use seeded eq. (9) data (non-paper data)")


@functools.cache
def load(j: int, i: int) -> tuple[MultiKnapsack, str]:
    """The instance and its source: ``"paper"`` (checksum-verified file) or ``"generated"``."""
    path = data_path(j, i)
    if path.exists() and SHA256.get((j, i)) == sha256(path):
        k = MultiKnapsack.from_zt_file(path)
        assert (k.n_items, k.n_knapsacks) == (j, i)
        return k, "paper"
    return MultiKnapsack.random(j, i, seed=1000 * j + i), "generated"


def profits_front(items: list) -> list[list[int]]:
    """Distinct non-dominated objective vectors of ``items`` as positive profits, sorted."""
    return sorted((-distinct_front(y for _, y in items)).astype(int).tolist())


def run_key(record: dict[str, Any]) -> tuple[str, str, int, int, str]:
    return record["instance"], record["method"], record["seed"], record["budget"], record["source"]


def run_one(j: int, i: int, method: str, seed: int) -> dict:
    k, source = load(j, i)
    problem = pyiea.BinaryProblem(j, f"knapsack.{j}.{i}-{source}")
    ev = pyiea.Evaluator(k, i, problem.version, max_calls=N_EVAL[(i, j)])
    t0 = time.perf_counter()
    extra: dict = {}
    if method.startswith("imoea"):
        cfg = pyiea.IMOEAConfig(**IMOEA_CFG, elite_unique_objectives=method == "imoea_unique")
        res = pyiea.IMOEA(problem, ev, cfg, seed).optimize()
        applied = [g for g in res.igc_log if g["status"] == "applied"]
        extra = {
            "igc_applied": len(applied),
            "igc_other": {s: sum(g["status"] == s for g in res.igc_log) for s in {g["status"] for g in res.igc_log}},
            "igc_N_below_15": sum(g["N"] < 15 for g in applied),
        }
    elif method == "nsga2":
        res = baselines.nsga2(problem, ev, seed, pop_size=50, pc=0.8, pm=0.01)
    elif method == "random":
        res = baselines.random_search(problem, ev, seed)
    else:
        raise ValueError(method)
    return {
        "instance": f"{j}.{i}",
        "source": source,
        "method": method,
        "seed": seed,
        "calls": res.accounting["objective_calls"],
        "budget": N_EVAL[(i, j)],
        "seconds": time.perf_counter() - t0,
        "generations": res.generations,
        "stop_reason": res.stop_reason,
        "front": profits_front(res.archive),
        "working_front": profits_front(res.front),
        **extra,
    }


def cover_profits(fa: list, fb: list) -> list[float]:
    """Paired cover metric of fronts stored as positive profits (negated into minimized objectives)."""
    return cover_paired([[[-v for v in p] for p in f] for f in fa], [[[-v for v in p] for p in f] for f in fb])


def merged_profits(fronts: list) -> list[list[int]]:
    """Non-dominated union of fronts stored as positive profits."""
    return sorted((-merged_front([[[-v for v in p] for p in f] for f in fronts])).astype(int).tolist())


def run(runs: int, workers: int, instances: list[tuple[int, int]], methods: list[str]) -> RunCache:
    jobs = []
    for j, i in instances:
        source = load(j, i)[1]
        jobs += [((f"{j}.{i}", m, s, N_EVAL[(i, j)], source), (j, i, m, s)) for m in methods for s in range(runs)]
    jobs.sort(key=lambda t: -N_EVAL[(t[1][1], t[1][0])] * (3 if t[1][2].startswith("imoea") else 1))  # longest first
    cache = RunCache(RUN_CACHE, run_key)
    cache.run(jobs, run_one, workers)
    return cache


def report(done: dict, runs: int, instances: list[tuple[int, int]], methods: list[str]) -> None:
    summary: dict = {"settings": {"imoea": IMOEA_CFG, "nsga2": {"pop_size": 50, "pc": 0.8, "pm": 0.01}}}
    fronts: dict = {}
    per_inst: dict = {}
    for j, i in instances:
        name, source = f"{j}.{i}", load(j, i)[1]
        rows = {
            m: sorted(
                (r for k, r in done.items() if k[0] == name and k[1] == m and k[4] == source and k[2] < runs),
                key=lambda r: r["seed"],
            )
            for m in methods
        }
        rows = {m: v for m, v in rows.items() if len(v) == runs}
        info: dict = {"source": source, "budget": N_EVAL[(i, j)], "runs": runs, "methods": {}, "cover": {}}
        for m, rs in rows.items():
            info["methods"][m] = {
                "calls_mean": float(np.mean([r["calls"] for r in rs])),
                "seconds_mean": float(np.mean([r["seconds"] for r in rs])),
                "seconds_total": float(np.sum([r["seconds"] for r in rs])),
                "front_size": quartiles([len(r["front"]) for r in rs]),
                "stop_reasons": sorted({r["stop_reason"] for r in rs}),
                **(
                    {
                        "igc_N_below_15_frac": float(
                            np.sum([r["igc_N_below_15"] for r in rs]) / max(1, np.sum([r["igc_applied"] for r in rs]))
                        )
                    }
                    if m.startswith("imoea")
                    else {}
                ),
            }
            fronts.setdefault(name, {})[m] = [r["front"] for r in rs]
        for a, b in PAIRS:
            if a in rows and b in rows:
                fa, fb = fronts[name][a], fronts[name][b]
                info["cover"][f"C({a},{b})"] = quartiles(cover_profits(fa, fb))
                info["cover"][f"C({b},{a})"] = quartiles(cover_profits(fb, fa))
        per_inst[name] = info
    summary["instances"] = per_inst
    if "750.2" in fronts:
        merged = {m: merged_profits(f) for m, f in fronts["750.2"].items()}
        summary["merged_750_2_ranges"] = {
            m: {
                "f1": [min(p[0] for p in f), max(p[0] for p in f)],
                "f2": [min(p[1] for p in f), max(p[1] for p in f)],
                "points": len(f),
            }
            for m, f in merged.items()
        }
        summary["merged_750_2_fronts"] = merged  # plot data for Fig. 5 (positive profits f1, f2)
        fronts["merged_750_2"] = merged
    OUT.mkdir(exist_ok=True)
    (OUT / "imoea_knapsack.json").write_text(json.dumps(summary, indent=2))
    with gzip.open(OUT / "imoea_knapsack_fronts.json.gz", "wt") as fh:
        json.dump(fronts, fh)
    md = markdown(summary)
    (OUT / "imoea_knapsack.md").write_text(md)
    print(md)


def markdown(summary: dict) -> str:
    L = ["## IMOEA paper, Section VI-A: multi-objective 0/1 knapsack", ""]
    L.append(
        "Cover metric C(A, B) (eq. 8) = fraction of B's front weakly dominated by A's front; median "
        "[first quartile, third quartile] over the runs, run r of A against run r of B (Fig. 4). Fronts are "
        "the distinct non-dominated objective vectors each run evaluated. Every run used exactly the Table VIII "
        "budget."
    )
    L.append("")
    for name, info in summary["instances"].items():
        src = "Zitzler-Thiele data" if info["source"] == "paper" else "**non-paper data**: seeded eq. 9"
        L.append(f"### knapsack.{name} ({src}), N_eval = {info['budget']:,}, {info['runs']} runs")
        L.append("")
        L.append("| pair | C(A, B) | C(B, A) |")
        L.append("|---|---|---|")
        cov = info["cover"]
        for a, b in PAIRS:
            if f"C({a},{b})" in cov:
                L.append(
                    f"| A = {a}, B = {b} | {fmt_quartiles(cov[f'C({a},{b})'])} | {fmt_quartiles(cov[f'C({b},{a})'])} |"
                )
        L.append("")
        L.append("| method | front size median [q1, q3] | mean calls | mean wall s/run | stop |")
        L.append("|---|---|---|---|---|")
        for m, s in info["methods"].items():
            extra = f"; IGCs with N < 15: {100 * s['igc_N_below_15_frac']:.1f}%" if "igc_N_below_15_frac" in s else ""
            L.append(
                f"| {m} | {fmt_quartiles(s['front_size']).replace('.00', '')} | {s['calls_mean']:.0f} | "
                f"{s['seconds_mean']:.1f} | {', '.join(s['stop_reasons'])}{extra} |"
            )
        L.append("")
    if "merged_750_2_ranges" in summary:
        L.append("### knapsack.750.2: union of the 30 runs' fronts (cf. Fig. 5)")
        L.append("")
        L.append("| method | points | f1 range | f2 range |")
        L.append("|---|---|---|---|")
        for m, r in summary["merged_750_2_ranges"].items():
            L.append(f"| {m} | {r['points']} | {r['f1'][0]}-{r['f1'][1]} | {r['f2'][0]}-{r['f2'][1]} |")
        L.append("")
        L.append(
            "Paper, Fig. 5 (read off the plot, merged over 30 runs): IMOEA about f1 27 000-28 350, "
            "f2 26 600-28 330; NSGA2 (ETH results, [27] settings) about f1 25 950-28 480, f2 25 200-28 280."
        )
        L.append("")
    L.append(f"_pyiea {pyiea.__version__}_")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "run", "report"])
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--instances", nargs="*", default=[f"{j}.{i}" for j, i in INSTANCES])
    ap.add_argument("--methods", nargs="*", default=list(METHODS), choices=METHODS)
    a = ap.parse_args()
    instances = [(int(s.split(".")[0]), int(s.split(".")[1])) for s in a.instances]
    if a.command == "fetch":
        fetch()
    else:
        cache = run(a.runs, a.workers, instances, a.methods) if a.command == "run" else RunCache(RUN_CACHE, run_key)
        report(cache.records, a.runs, instances, a.methods)


if __name__ == "__main__":
    main()
