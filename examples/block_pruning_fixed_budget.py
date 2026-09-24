"""Block pruning with a fixed budget (single-objective IEA).

Which 8 of the 32 attention/MLP sub-blocks of a 16-layer decoder should be dropped
so that the output distortion D is smallest? Genome bit 2i keeps the attention of
layer i, bit 2i+1 its MLP (1 = keep, 0 = drop).

D here is a toy stand-in for the teacher->candidate KL of the real model: each
sub-block has an importance, and dropping neighbouring sub-blocks together costs
extra. A real objective, such as the KL of the pruned model, plugs into pyiea
the same way.
"""

import numpy as np

import pyiea

N_LAYERS = 16
N_UNITS = 2 * N_LAYERS  # attention, MLP, attention, MLP, ...
rng = np.random.default_rng(0)
depth = np.repeat(np.linspace(0, 1, N_LAYERS), 2)
importance = rng.uniform(0.2, 1.0, N_UNITS) * (1.5 - np.sin(np.pi * depth))  # middle layers matter less
pair_cost = 0.3 * rng.uniform(0, 1, N_UNITS - 1)  # dropping unit j and j+1 together


def distortion(keep: pyiea.Genome) -> float:
    drop = 1.0 - keep.astype(float)
    return float(importance @ drop + pair_cost @ (drop[:-1] * drop[1:]))


if __name__ == "__main__":
    problem = pyiea.FixedCardinalityProblem(N_UNITS, k=N_UNITS - 8)  # keep exactly 24 sub-blocks
    res = pyiea.optimize(distortion, problem=problem, max_calls=3000, seed=0)
    assert res.best_genome is not None
    dropped = np.flatnonzero(res.best_genome == 0)
    names = [f"L{i // 2}.{'attn' if i % 2 == 0 else 'mlp'}" for i in dropped]
    print(f"D = {res.best:.4f} with {res.accounting['objective_calls']:.0f} objective calls ({res.stop_reason})")
    print("dropped:", ", ".join(names))
