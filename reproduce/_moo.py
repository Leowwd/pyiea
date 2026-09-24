"""Helpers shared by the multi-objective reproduction scripts (not part of the pyiea package).

Fronts are lists of objective vectors in the **minimized** convention used by pyiea.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Hashable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

import pyiea

Front = list[list[float]]


def default_workers() -> int:
    """One worker per CPU of the machine the script runs on."""
    return os.cpu_count() or 1


def distinct_front(Y: Iterable[Sequence[float]]) -> np.ndarray:
    """Distinct non-dominated rows of ``Y``, sorted lexicographically."""
    A = np.unique(np.asarray(list(Y), dtype=float), axis=0)
    return A[pyiea.nondominated_mask(A)] if len(A) else A


def quartiles(values: Sequence[float]) -> dict[str, float]:
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return {"q1": float(q1), "median": float(med), "q3": float(q3), "mean": float(np.mean(values)), "n": len(values)}


def fmt_quartiles(q: dict[str, float], spec: str = ".2f") -> str:
    """``median [q1, q3]``."""
    return f"{q['median']:{spec}} [{q['q1']:{spec}}, {q['q3']:{spec}}]"


def cover_paired(fronts_a: Sequence[Front], fronts_b: Sequence[Front]) -> list[float]:
    """``C(A_r, B_r)`` (eq. 8) for every run ``r``: run ``r`` of A against run ``r`` of B."""
    return [
        pyiea.coverage(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
        for a, b in zip(fronts_a, fronts_b, strict=True)
    ]


def merged_front(fronts: Sequence[Front]) -> np.ndarray:
    """Distinct non-dominated points of the union of several runs' fronts."""
    return distinct_front(p for f in fronts for p in f)


class RunCache:
    """Finished runs as JSON lines, so an interrupted experiment resumes where it stopped.

    Args:
        path: The JSON-lines file.
        key: Identifies a run from its record; a job whose key is present is skipped.
    """

    def __init__(self, path: Path, key: Callable[[dict[str, Any]], Hashable]) -> None:
        self.path, self.key = path, key
        self.records: dict[Hashable, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                record = json.loads(line)
                self.records[key(record)] = record

    def __contains__(self, k: Hashable) -> bool:
        return k in self.records

    def run(
        self, jobs: Sequence[tuple[Hashable, tuple[Any, ...]]], fn: Callable[..., dict[str, Any]], workers: int
    ) -> None:
        """Run ``fn(*args)`` for every ``(key, args)`` job not yet cached, in order, on ``workers`` processes."""
        todo = [(k, args) for k, args in jobs if k not in self]
        print(f"{len(todo)} runs to do, {len(jobs) - len(todo)} cached, {workers} workers", flush=True)
        if not todo:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(workers) as pool, self.path.open("a") as log:
            futures = {pool.submit(fn, *args): k for k, args in todo}
            for n, fut in enumerate(as_completed(futures), 1):
                record = fut.result()
                assert self.key(record) == futures[fut], "record key does not match its job"
                self.records[futures[fut]] = record
                log.write(json.dumps(record) + "\n")
                log.flush()
                if n % 20 == 0 or n == len(todo):
                    print(f"{n}/{len(todo)} runs, {time.perf_counter() - t0:.0f} s", flush=True)
