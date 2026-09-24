"""Distortion vs. size trade-off for block pruning (multi-objective IMOEA).

Objectives, both minimized:
  D = output distortion (toy stand-in for the teacher->candidate KL),
  R = retained parameter ratio, counting unique parameters, with the embeddings
      and final norm always kept.
Sub-block sizes follow Llama-3.2-1B: attention 10.5M, MLP 50.3M, fixed part 262.7M.
"""

import numpy as np

import pyiea

N_LAYERS = 16
N_UNITS = 2 * N_LAYERS
ATTN, MLP, FIXED = 10_487_808, 50_333_696, 262_670_336
sizes = np.tile([ATTN, MLP], N_LAYERS).astype(float)
rng = np.random.default_rng(0)
depth = np.repeat(np.linspace(0, 1, N_LAYERS), 2)
importance = rng.uniform(0.2, 1.0, N_UNITS) * (1.5 - np.sin(np.pi * depth))
pair_cost = 0.3 * rng.uniform(0, 1, N_UNITS - 1)


def objectives(keep: pyiea.Genome) -> tuple[float, float]:
    drop = 1.0 - keep.astype(float)
    d = float(importance @ drop + pair_cost @ (drop[:-1] * drop[1:]))
    r = float((FIXED + sizes @ keep) / (FIXED + sizes.sum()))
    return d, r


if __name__ == "__main__":
    res = pyiea.optimize(objectives, n_bits=N_UNITS, mode="multi_objective", n_objectives=2, max_calls=3000, seed=0)
    front = sorted(res.archive, key=lambda item: item[1][1])
    print(f"{len(front)} non-dominated pruning plans ({res.accounting['objective_calls']:.0f} calls)")
    for keep, (d, r) in front[:: max(1, len(front) // 8)]:
        print(f"  R = {r:.3f}  D = {d:6.3f}  dropped {int((keep == 0).sum()):2d} sub-blocks")
