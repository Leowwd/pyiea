"""A custom search space: sub-blocks that must never be dropped.

The first and last layers are protected here. Putting the constraint in the
Problem (not in the fitness) means no candidate is ever infeasible, so no
objective call is wasted.
"""

import numpy as np

import pyiea

N_UNITS = 32
PROTECTED = [0, 1, 30, 31]  # layer 0 and layer 15 (attention and MLP)


class ProtectedBlocksProblem(pyiea.BinaryProblem):
    def __init__(self, n_bits: int, protected: list[int]) -> None:
        super().__init__(n_bits, version=f"protected-{sorted(protected)}")
        self.protected = np.array(sorted(protected))

    def random_genome(self, rng: np.random.Generator) -> pyiea.Genome:
        g = rng.integers(0, 2, self.n_bits, dtype=np.uint8)
        g[self.protected] = 1
        return pyiea.freeze(g)

    def is_valid(self, g: pyiea.Genome) -> bool:
        return super().is_valid(g) and bool(np.all(g[self.protected] == 1))

    def mutate(self, g: pyiea.Genome, pm: float, rng: np.random.Generator) -> pyiea.Genome:
        flip = rng.random(self.n_bits) < pm
        flip[self.protected] = False
        return pyiea.freeze(g ^ flip) if flip.any() else g

    # divide() is inherited: protected bits are 1 in both parents, so they never
    # appear among the differing positions and every OA combination stays feasible.


rng = np.random.default_rng(1)
importance = rng.uniform(0.2, 1.0, N_UNITS)
sizes = np.tile([10_487_808, 50_333_696], N_UNITS // 2).astype(float)


def objectives(keep: pyiea.Genome) -> tuple[float, float]:
    return float(importance @ (1 - keep.astype(float))), float(sizes @ keep / sizes.sum())


if __name__ == "__main__":
    problem = ProtectedBlocksProblem(N_UNITS, PROTECTED)
    res = pyiea.optimize(objectives, problem=problem, n_objectives=2, max_calls=2000, seed=0)
    kept = all(bool(np.all(g[problem.protected] == 1)) for g, _ in res.archive)
    print(f"{len(res.archive)} pruning plans; protected sub-blocks always kept: {kept}")
    print(f"infeasible candidates evaluated: {res.accounting['invalid_candidates']:.0f}")
