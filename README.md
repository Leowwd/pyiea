# pyiea

Intelligent evolutionary algorithms for binary search spaces, following

> S.-Y. Ho, L.-S. Shu, J.-H. Chen, "Intelligent Evolutionary Algorithms for Large Parameter
> Optimization Problems," *IEEE Transactions on Evolutionary Computation* 8(6):522–541, 2004.
> [doi:10.1109/TEVC.2004.835176](https://doi.org/10.1109/TEVC.2004.835176)

- **IEA** (`mode="single_objective"`): a population loop with truncation selection, IGC recombination and elitist mutation.
- **IMOEA** (`mode="multi_objective"`): GPSIFF fitness, a bounded elite set, and a separate archive of every non-dominated solution found.
- **IGC**, the intelligent gene collector. It splits two parents into gene segments, evaluates a two-level orthogonal array of segment combinations, and builds children from the main effects.

Every objective call is counted: OA rows, child confirmations, cache hits and failures. Runs are reproducible from a seed and can be checkpointed and resumed exactly.

## Install

```bash
pip install pyiea
```

For the development version, clone the repository and install it in editable mode:

```bash
git clone https://github.com/Leowwd/pyiea.git
cd pyiea
pip install -e ".[dev]"   # or: pip install -e .   (runtime only)
```

The only runtime dependency is NumPy. Python ≥ 3.10 is required.

## Quick start

A fitness function takes a genome, which is a read-only `numpy.uint8` array of 0/1 values. It returns a float, or a sequence of floats when there are several objectives. **Everything is minimized**, so negate anything you want to maximize.

The running example is block pruning of a 16-layer decoder. Bit `2i` keeps the attention of layer `i` and bit `2i+1` keeps its MLP (1 = keep, 0 = drop). Which sub-blocks can go?

```python
import numpy as np
import pyiea

rng = np.random.default_rng(0)
importance = rng.uniform(0.2, 1.0, 32)  # toy stand-in for the KL increase of dropping each sub-block


def distortion(keep):
    return float(importance @ (1 - keep.astype(float)))


# drop exactly 8 of the 32 sub-blocks
res = pyiea.optimize(distortion, problem=pyiea.FixedCardinalityProblem(32, k=24), max_calls=1000)
print(res.best, np.flatnonzero(res.best_genome == 0), res.stop_reason)
print(res.accounting)  # objective_calls, cache_hits, failed_evaluations, wall_seconds, ...
```

To trade distortion against size, return both objectives and use IMOEA. The result is a Pareto front of pruning plans:

```python
sizes = np.tile([10_487_808, 50_333_696], 16)  # attention / MLP parameters (Llama-3.2-1B)


def objectives(keep):
    return distortion(keep), float(sizes @ keep / sizes.sum())


res = pyiea.optimize(objectives, n_bits=32, mode="multi_objective", n_objectives=2, max_calls=3000)
for keep, (d, r) in sorted(res.archive, key=lambda item: item[1][1]):
    ...
```

`examples/` holds runnable versions of these, plus a custom search space in which some sub-blocks are protected.

Returning `nan` or `inf`, or raising an exception, marks that candidate as a **failed evaluation**. A failed candidate is never treated as a good fitness and never enters the OA main-effect arithmetic. Because it still costs an objective call, use a custom problem for constraints (see below).

### With a real model

A real objective plugs in the same way. Write a picklable class whose `__call__(keep)` applies the mask to a frozen model (bypassing the dropped sub-blocks, never changing the weights) and returns, for example, `(KL to the unpruned model on fixed calibration tokens, retained parameter ratio)`. Load the model and data once in `__init__`, and give the `Evaluator` a `context` that names the model revision and the calibration data, so that cached values and checkpoints are never mixed across setups. A single model on one GPU means serial evaluation (`workers=1`).

## Constraints: make infeasible genomes impossible

The paper recommends keeping every candidate feasible by construction instead of penalizing infeasible ones. In `pyiea`, a `Problem` owns initialization, mutation and the IGC division into segments. For example, "drop exactly `n - k` sub-blocks" is built in as `FixedCardinalityProblem(n_bits, k)`.

This follows the IEA paper's polygonal-approximation example (Section III-A):
- Segments are cut so that both parents have the same number of ones in each one, which makes every OA combination feasible.
- Mutation swaps a 1 with a 0.

For another constraint, subclass `BinaryProblem` and override:
- `random_genome`
- `is_valid`
- `mutate`: return the same object when nothing changes.
- `divide`: return the segments, or `[]` when fewer than two are possible.

Every OA combination built from `divide` must be legal. `examples/protected_blocks.py` shows a problem in which the first and last layers can never be dropped.

## Real-valued parameters

The paper encodes each continuous parameter as a fixed number of bits. Use Gray code
(`gray=True`) so that neighbouring values are one bit apart, and `enc.problem()` so that
IGC never cuts through a parameter:

```python
def f1(x):  # Table IV, f1: maximize sum(sin(x) + sin(2x/3)) on [3, 13]
    return -float((np.sin(x) + np.sin(2 * x / 3)).sum())


enc = pyiea.RealEncoder([(3.0, 13.0)] * 10, bits=10, gray=True)  # 10 parameters x 10 bits
res = pyiea.optimize(enc.wrap(f1), problem=enc.problem(), max_calls=10_000)
x_best = enc.decode(res.best_genome)
```

## Lower-level API

`optimize` is a thin wrapper around the classes. Use them directly when you need control over:
- **budgets:** `max_calls`, `max_seconds`
- **parallel evaluation:** `workers`
- **checkpoints**
- **the paper's options:** `step10`, and `max_segments` for the bounded-segments variant
- **engineering options:** `IMOEAConfig(elite_unique_objectives=True)` keeps one genome per objective vector in the elite set. It helps when many genomes decode to the same solution.

```python
problem = pyiea.BinaryProblem(32)
with pyiea.Evaluator(
    distortion, n_objectives=1, context="my-fitness-v1", is_valid=problem.is_valid, max_calls=5000, workers=4
) as ev:
    cfg = pyiea.IEAConfig(pop_size=30, ps=0.2, pc=0.8, pm=0.05, checkpoint_every=10)
    res = pyiea.IEA(problem, ev, cfg, seed=0).optimize(checkpoint_path="run.ckpt")
```

- **Context.** The `context` string names everything the fitness depends on, such as the data and model versions. A checkpoint refuses to load into an evaluator with a different context.
- **Resuming.** To resume, construct the same objects and call `optimize(checkpoint_path=..., resume=True)`. You may change the stopping conditions (`max_generations`, `target`, `max_stall_generations`, `checkpoint_every`) to extend a run.
- **Parallel workers.** With `workers > 1`, the fitness must be picklable, which means a module-level function or a class instance.
- **Security.** Checkpoints are pickles, so load only files you trust.

## What is faithful to the paper

[`docs/paper_map.md`](https://github.com/Leowwd/pyiea/blob/main/docs/paper_map.md) maps every step of Sections II–IV to code and tests. It also marks each place the paper leaves open and names the rule chosen there, such as rounding, ties, and how the population is refilled. The `max_segments` cap (`bounded_segments`) is a variant and should be reported as one, unless the paper itself fixes N for the experiment (Table VIII).

`pyiea.benchmarks` holds the paper's test problems: `PaperBenchmark` (f1–f12 of Table IV), `MultiKnapsack` (eq. 9) and `ZDT` (ZDT1–ZDT6). The repository's [`reproduce/`](https://github.com/Leowwd/pyiea/tree/main/reproduce) directory reruns Tables III, V and VI, Section V-A, Fig. 3 and Section VI with them. The results, including where they differ from the paper, are in `reproduce/results/`.

`pyiea.baselines` (random search, vanilla GA, the elitist GAs with one-point, two-point and uniform crossover, NSGA-II) share the same evaluator and budget, so comparisons are fair. They are baselines, not IEA or IMOEA.

## Development

```bash
pip install -e ".[dev]"
pytest --cov                 # tests and doctests
ruff check . && ruff format --check .
mypy                         # strict
python -m build && twine check dist/*
```

## License

Apache-2.0. See [LICENSE](https://github.com/Leowwd/pyiea/blob/main/LICENSE).
