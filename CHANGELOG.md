# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0.0, minor
versions may change the public API.

## [Unreleased]

## [0.2.0] - 2026-09-24

### Added
- `RealEncoder(gray=True)` reads each parameter as a reflected Gray code.
- `reproduce/iea_paper.py`, which reruns the IEA paper's Tables III, V and VI and Section V-A.
- `benchmarks.MultiKnapsack`: the multi-objective 0/1 knapsack of the IMOEA experiments (eq. 9), with the greedy repair of Zitzler & Thiele (1999) applied when decoding. It reads their `knapsack.<J>.<I>` files, and it can generate seeded instances by the eq. 9 rule.
- `IMOEAConfig(elite_unique_objectives=True)`, an opt-in engineering option: E keeps one genome per objective vector before the random truncation. `ParetoSet.drop_duplicate_objectives()` implements it.
- `reproduce/imoea_knapsack.py`, which reruns Section VI-A (9 instances, 30 runs) against NSGA-II and random-search baselines and reports the cover metric.
- `reproduce/imoea_zdt.py`, which reruns Section VI-B (ZDT1–ZDT6, m = 63, 30 runs) against an NSGA-II baseline in binary and Gray encodings, with hypervolume, IGD/GD and the cover metric. Resumable run caching and the front helpers live in `reproduce/_moo.py`, which other multi-objective scripts can share.
- `baselines.elitist_ga(crossover=...)` with `baselines.CROSSOVERS` (`"one_point"`, `"two_point"`, `"uniform"`): the elitist GAs OEGA, TEGA and UEGA that the IEA paper compares with IGC. They follow IEA's Steps 1–6 with IGC replaced by the crossover, and they are baselines, not IEA.
- `reproduce/iea_fig3.py`, which reruns Fig. 3 (dist(D) for f1–f12, D = 10…100, 30 runs) for IEA against OEGA, TEGA and UEGA, plus an ablation with pm = 1/n_bits for the GAs.
- `reproduce/iea_f8_f12.py`, which tests explanations for the open f8/f12 differences from Tables V and VI. The f12 values at D = 10 match a `cos(x_i)/sqrt(i)` Griewank. f8 at D = 100 stays unresolved.
- `benchmarks.ZDT`: ZDT1–ZDT6 of Zitzler, Deb & Thiele (2000) for the IMOEA experiments (eq. 10). It covers any number of parameters `m`, real parameters through `RealEncoder` (binary or Gray), ZDT5 on its bit string, and the analytic Pareto-optimal fronts.

### Changed
- `PaperBenchmark.encoder()` now uses Gray code. With it, Table III and Section V-A match the paper (see `docs/paper_map.md`).

### Performance
- `ParetoSet.update` compares only the new items with the members, instead of rebuilding the full dominance matrix. Results are unchanged, and large archives (many objectives) update in linear time.
- `nondominated_mask` handles two objectives with an `O(n log n)` sort instead of the `O(n²)` pairwise matrix. The result is unchanged, and equal rows are still all kept.

### Packaging
- pyiea has its own repository, https://github.com/Leowwd/pyiea. Clone it and run `pip install -e .`.
- `.github/workflows/ci.yml` runs lint, strict typing, the tests on Python 3.10–3.13, and the build. `.github/workflows/release.yml` publishes to PyPI with trusted publishing when a `vX.Y.Z` tag is pushed. It checks the tag against the version and CHANGELOG, and it tests the built wheel first. See "Releasing" in `CONTRIBUTING.md`.
- The build requires `hatchling>=1.27`. Older hatchling rejects the `license-files` array (≤ 1.25) or writes no PEP 639 license metadata (1.26).
- Project URLs for the documentation, changelog and issues. The README links to the license with an absolute URL, so the link also works on PyPI.

## [0.1.0] - 2026-09-24

### Added
- `IEA` and `IMOEA` from Ho, Shu, Chen (2004): the IGC operator (Steps 1–10), `Generate_OA`, GPSIFF, elite sets, and a separate non-dominated archive.
- `Evaluator`, a single point for objective calls. It handles caching per context, call and time budgets, a per-call ledger, accounting counters, and an optional process pool.
- `BinaryProblem` and `FixedCardinalityProblem`, whose operators keep genomes feasible. The latter uses the division criterion of the paper's Section III-A example and a swap mutation.
- `RealEncoder`, which maps bit strings to real-valued parameters.
- `optimize()`, a one-call entry point for both modes.
- Deterministic checkpoint and resume. Stopping conditions may change when a run is resumed.
- `pyiea.baselines` (random search, vanilla GA, NSGA-II) and `pyiea.benchmarks`, test problems whose optima can be checked.
