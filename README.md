# pyiea

Intelligent evolutionary algorithms for bit strings and real-valued parameters, with one objective or several, following

> S.-Y. Ho, L.-S. Shu, J.-H. Chen, "Intelligent Evolutionary Algorithms for Large Parameter
> Optimization Problems," *IEEE Transactions on Evolutionary Computation* 8(6):522–541, 2004.
> [doi:10.1109/TEVC.2004.835176](https://doi.org/10.1109/TEVC.2004.835176)

- **IEA**, for one objective: a population loop with truncation selection, IGC recombination and elitist mutation.
- **IMOEA**, for two or more objectives: GPSIFF fitness, a bounded elite set, and a separate archive of every non-dominated solution found. It returns a Pareto front.
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

**Everything is minimized**, so negate anything you want to maximize.

### Real-valued parameters

Give the bounds of each parameter. The fitness receives a float array:

```python
import numpy as np
import pyiea


def sphere(x):
    return float(((x - 1.5) ** 2).sum())


res = pyiea.optimize(sphere, bounds=[(-5.0, 5.0)] * 3, max_calls=3000)
print(res)  # IEAResult(best=..., best_x=[1.5 1.5 1.5], stop_reason=..., objective_calls=...)
print(res.best_x, res.best)
```

Each parameter is encoded with `bits=10` bits by default (use more for finer resolution) and Gray-coded, as the paper's benchmarks need (see [Encoding real parameters](#encoding-real-parameters)).

### Bit strings

With `n_bits` (or a `problem`), the fitness receives a read-only `numpy.uint8` array of 0/1 values. The running example is block pruning of a 16-layer decoder: bit `2i` keeps the attention of layer `i` and bit `2i+1` keeps its MLP (1 = keep, 0 = drop). Which sub-blocks can go?

```python
rng = np.random.default_rng(0)
importance = rng.uniform(0.2, 1.0, 32)  # toy stand-in for the KL increase of dropping each sub-block


def distortion(keep):
    return float(importance @ (1 - keep.astype(float)))


# drop exactly 8 of the 32 sub-blocks
res = pyiea.optimize(distortion, problem=pyiea.FixedCardinalityProblem(32, k=24), max_calls=1000)
print(res.best, np.flatnonzero(res.best_genome == 0), res.stop_reason)
print(res.accounting)  # objective_calls, cache_hits, failed_evaluations, wall_seconds, ...
```

### Several objectives

Return one value per objective and pass `n_objectives`; pyiea then runs IMOEA and returns a Pareto front. To trade distortion against size:

```python
sizes = np.tile([10_487_808, 50_333_696], 16)  # attention / MLP parameters (Llama-3.2-1B)


def objectives(keep):
    return distortion(keep), float(sizes @ keep / sizes.sum())


res = pyiea.optimize(objectives, n_bits=32, n_objectives=2, max_calls=3000)
print(res)  # IMOEAResult(front=... points, archive=... points, ...)
F = res.archive_objectives  # one row (distortion, size) per non-dominated plan found
for keep, (d, r) in sorted(res.archive, key=lambda item: item[1][1]):
    ...
```

`res.front` is the bounded elite set of the last generation; `res.archive` keeps every non-dominated solution found. With `bounds`, `res.front_x` holds the real parameters of the front.

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

## Encoding real parameters

As in the paper, each real parameter is a fixed number of bits: `x = low + k (high - low) / (2^bits - 1)` for the integer `k` the bits encode. `optimize(bounds=...)` does this for you. Gray code (the default, `gray=True`) makes neighbouring values one bit apart, so mutation can fine-tune a converged population. With plain binary, pyiea does not reach the paper's single-objective results (`docs/paper_map.md`). IGC never cuts through a parameter.

To maximize, minimize the negation. For example, the paper's Table IV asks to maximize f1(x) = −Σ [sin(xᵢ) + sin(2xᵢ/3)] on [3, 13], whose maximum is 1.21598 per parameter. That means minimizing Σ [sin(xᵢ) + sin(2xᵢ/3)]:

```python
def minus_f1(x):
    return float((np.sin(x) + np.sin(2 * x / 3)).sum())


res = pyiea.optimize(minus_f1, bounds=[(3.0, 13.0)] * 10, bits=10, max_calls=10_000)
print(-res.best)  # f1 ≈ 12.16 = 1.21598 × 10
```

`pyiea.RealEncoder` is the same encoding for the lower-level API: `enc.wrap(f)` is the objective on genomes, `enc.problem()` the search space and `enc.decode(genome)` the parameters.

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `the objective returned 2 values for a genome, but n_objectives=1` | The fitness returns several objectives. Pass `n_objectives=2` to run IMOEA. |
| `... returned 1 value ..., but n_objectives=2` | Return one value per objective, e.g. `return distortion, size`. |
| `EvaluationError: the objective failed on all 30 genomes of the initial population; the first failure was ...` | The fitness raised (or returned NaN/inf) for every genome. The message quotes the first error, so fix that and rerun. pyiea stops right away instead of spending the budget. |
| `... assignment destination is read-only` | Genomes are read-only so that parents cannot change. Work on `g.copy()`. |
| `workers > 1 needs a picklable objective` | Worker processes cannot receive lambdas or closures. Use a module-level function or a class instance. |
| `stop_reason='stalled'` | No new genome was evaluated for `max_stall_generations` generations, which usually means the search space is exhausted. |

Occasional failures are normal: they are counted in `res.accounting["failed_evaluations"]` and never count as good fitness.

To follow a long run, turn on the log: `logging.basicConfig(); logging.getLogger("pyiea").setLevel(logging.DEBUG)` prints one line per generation.

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
