# Paper → code map

Source: Ho, Shu, Chen (2004), *Intelligent Evolutionary Algorithms for Large Parameter Optimization Problems*, IEEE TEVC 8(6):522–541. The relevant parts are Sections II–IV, printed pages 523–528. This package implements IEA and IMOEA only.

Formulas were checked against the page images, not the text extraction. Code paths are relative to `src/pyiea/`.

Legend for the **Status** column:
- **faithful**: the code does what the paper says.
- **eng**: an engineering choice for something the paper leaves unspecified.
- **variant**: a deliberate deviation. It must be reported under the "IEA-based variant" name.

All objectives are **minimized**. GPSIFF is **maximized**. A maximization objective has to be passed in as its negation (see `test_iea_maximization_by_documented_negation`). Nothing in the code flips signs anywhere else.

## Section II-A: orthogonal array

| Paper | Meaning | Code | Test | Status |
|---|---|---|---|---|
| `n = 2^⌈log2(N+1)⌉`, `L_n(2^(n−1))`, first N columns | two-level OA for N factors | `oa.oa_rows`, `oa.generate_oa` | `test_oa_balance[N=1,3,7,10,15]` | faithful |
| Algorithm `Generate_OA(OA, N)` | level = parity of `(i−1)` bits (MSB first) masked by the bits of `j` | `oa.generate_oa` (transcribed line by line) | `test_oa_matches_paper_L4` (the paper's printed L4: 111/122/212/221) | faithful. It equals the common closed form `parity((i−1) & j)` with the rows bit-reversed, so the balance properties are the same. |
| Level 1 / 2 of factor j | segment j comes from parent 1 / parent 2 | `igc.decode` | `test_decode_by_hand` | faithful. The OA level is the parent source, not the keep/drop bit. |
| Eq. (2) `S_jk = Σ_t y_t F_t` | main effect as a **sum** (not a mean; the OA is balanced, so the two are equivalent) | `igc.main_effects` | `test_paper_table2_main_effects` (the paper's S values 119/319, 229/209, 220/218) | faithful |
| minimize: level 1 is better iff `S_j1 < S_j2` | better level per factor | `igc.igc` Step 7 | `test_additive_child1_is_best_segment_combination` | faithful. **eng:** ties `S_j1 = S_j2` pick level 1 (parent 1). See `test_main_effect_tie_prefers_parent1_reproducibly`. |
| `MED = |S_j1 − S_j2|` | main-effect difference | `igc.igc` | the same tests | faithful. **eng:** a tie on the smallest MED picks the first factor index. |

## Section III: IGC

| Paper | Meaning | Code | Test | Status |
|---|---|---|---|---|
| III-A / Step 1 | ignore positions where the parents agree | `igc.igc` (`diff`) | `test_segmentation_covers_diff_exactly_once` | faithful |
| III-A eq. (3), Step 2 | N non-empty, disjoint segments that cover the differing positions exactly | `problem.BinaryProblem.divide` | the same test | faithful |
| III-A: low-epistasis `N = 2^⌊log2(M+1)⌋ − 1`, with N−1 cut points drawn at random from the M−1 candidates | division size (`paper_reference` mode) | `problem.paper_n_segments`, `BinaryProblem.divide(max_segments=None)` | the same test | faithful. **eng:** segments are contiguous runs of differing positions in genome order. |
| III-A: "if the maximal N equals one, IGC is not necessarily applied" | M ≤ 2 gives N = 1 | `igc` returns the parents with `status="not_divisible"` and 0 calls | `test_identical_and_one_bit_parents_cost_nothing` | faithful. Identical parents give `status="identical"` and 0 calls. |
| III-A / V: "M is the number of parameters participated in the division phase"; the N−1 cut points separate individual parameters | division of encoded real parameters | `encoding.ParameterProblem.divide` (from `RealEncoder.problem()`) | `test_api.py` encoder tests, `reproduce/iea_paper.py` | faithful. M counts differing parameters, not bits, and a parameter is never split. |
| cap on N | `bounded_segments` mode (`max_segments`) | `divide(max_segments=k)` | `test_bounded_segments_cap` | **variant.** Always report it as bounded. |
| III-A PAP: maximal N such that each pair of segments has the same number of 1s | feasibility-preserving division for a fixed cardinality | `FixedCardinalityProblem.divide`: cut wherever parent 1's running surplus of 1s over the differing positions returns to 0 | `test_fixed_cardinality_division_keeps_all_rows_legal` | faithful to the PAP criterion. **eng:** when capped, a random subset of the balanced cut points is kept (`bounded_segments`). |
| Steps 3–5 | build the OA, decode the rows, evaluate the rows | `igc.igc` | `test_accounting_matches_mock_calls_and_ledger` | faithful. Row 1 is always parent 1, so it is a cache hit. |
| Steps 6–9 | main effects → C1; C2 = C1 with the smallest-MED factor flipped | `igc.igc` | `test_additive_child1_is_best_segment_combination` | faithful |
| C1/C2 fitness | "compute" the children | `evaluator.evaluate_batch([c1, c2], phase="child")` | `test_epistatic_children_really_evaluated_and_step10_keeps_best_row` | faithful. Children are always really evaluated. |
| Step 10 (optional, IEA elitism) | "select the best two individuals from the **n generated combinations, C1, and C2**"; "P1 is combination 1" | `igc(step10=True)`. The candidate set is the n OA rows (including row 1 = P1) plus C1 and C2. **P2 is not in the set** (P2 is not an OA row). | the same test (`children[0] ≤ best row`) | faithful. **eng:** candidates are deduplicated by genome, and ties keep row order (rows first, then C1, C2). Read literally, "the parents" in this candidate set means P1 only. |
| — | an OA row or child fails or is invalid | `status="failed_row"`: the parents come back unchanged and no main effects are computed | `test_failed_row_never_enters_main_effects` | **eng** (the paper assumes every combination is feasible) |
| — | not enough budget for the whole OA plus 2 children | `status="budget"`, nothing is evaluated, and the run stops | `test_budget_never_exceeded_and_no_partial_oa` | **eng:** stop rather than shrink N, so a partial OA is never used |

## Section IV-B: GPSIFF

| Paper | Code | Test | Status |
|---|---|---|---|
| eq. (4) `GPSIFF(X) = p − q + c`, with c = the number of participants | `pareto.gpsiff` | `test_gpsiff_hand_example` (a hand-computed S → 5, 3, 4, 4); `test_gpsiff_paper_fig1_formula` (3 − 2 + 12 = 13) | faithful |
| dominance | `pareto.dominance_matrix` | `test_equal_vectors_do_not_dominate` | faithful. Equal vectors do not dominate each other. |
| participant set: population fitness | the current population, with failed individuals excluded (they get a score of 0, below the minimum possible of 1) | `IMOEA._fitness` | — | **eng.** Duplicate genomes count as separate participants. |
| participant set: multi-objective IGC | GPSIFF over **this IGC's n OA combinations**, recomputed for every IGC and never reused across contexts. The main effect is the sum of GPSIFF per level, larger is better. | `igc(multi_objective=True)` | `test_byproducts_reach_temporary_and_final_sets` | **eng.** The paper says the IGC uses GPSIFF but does not name the comparison set. |
| objective cache separate from GPSIFF | objectives are cached per genome, while scores are computed fresh from the comparison set | `test_new_point_changes_gpsiff_not_cache` | faithful |

## Section IV-C: IEA

| Step | Code (`iea.IEA.optimize`) | Status |
|---|---|---|
| 1 random initial population of N_pop | `problem.random_genome` | faithful |
| 2 evaluate | `_evaluate_population`: only individuals that are new or changed since their last evaluation cost calls | faithful |
| 3 truncation: the best `(1−ps)·N_pop` form the new population | **the worst `ps·N_pop` are replaced by copies of the best `ps·N_pop`**, so N_pop stays constant; `test_truncation_refill_copies_the_best` | **eng.** The paper does not say how N_pop is restored. Rounding: `int(ps·N_pop)`. The earlier research prototype refilled with random copies of survivors instead. |
| 3 `I_best` | index 0 after sorting | faithful |
| 4 randomly select `pc·N_pop` parents, including `I_best` | `int(pc·N_pop)` rounded down to an even number. I_best is parent 1 of the first pair, and the other parents are drawn without replacement and paired in draw order. Children replace their parents. | faithful. **eng:** rounding and the pairing order. |
| 5 bit-inverse mutation with p_m, not applied to the best individual | `problem.mutate`, which skips the current population best | `test_iea_best_so_far_never_regresses` (checks that the population best never gets worse) | faithful |
| 6 termination | call budget, wall time, target, `max_generations`, and `max_stall_generations` (no new objective call for K generations) | **eng.** The stall stop keeps a fully cached search space from looping; see `test_exhausted_search_space_stops_as_stalled`. |
| — | a best-so-far over **every** evaluated candidate (OA rows included), kept separately from the population | **eng** |

## Section IV-D: IMOEA

| Step | Code (`imoea.IMOEA.optimize`) | Status |
|---|---|---|
| 1 random population; empty E and E′ | `ParetoSet()` × 2 | faithful |
| 2 evaluate + GPSIFF | the GPSIFF scores are computed in Step 4, where they are used; this is equivalent | faithful |
| 3 add the non-dominated members of the population and of E′ to E, empty E′, remove dominated members, randomly discard any excess over N_Emax | `_update_elite`, `ParetoSet.truncate(rng)` | faithful. **eng:** E is deduplicated by genome, and equal vectors from different genomes are kept. See `test_elite_set_dedup_dominance_and_seeded_truncation`. **eng option** `IMOEAConfig(elite_unique_objectives=True)` (off by default): before the random discard, E keeps only the first genome of each objective vector (`ParetoSet.drop_duplicate_objectives`, `test_elite_unique_objectives_keeps_distinct_vectors`). The paper says nothing about duplicates. The option matters when many genomes decode to one solution; see Section VI-A below. |
| 4 `N_pop − N_ps` by binary tournament and `N_ps` drawn at random from E, where `N_ps = N_pop·ps`, reduced to N_E if larger | as written | faithful. **eng:** `int()` rounding; tournament ties are broken at random; the two contestants are drawn with replacement; the elite draw is without replacement. |
| 5 IGC on `pc·N_pop` selected parents; the non-dominated **by-products and two children** go to E′ | `igc(multi_objective=True)`. Non-dominated within that IGC's rows plus children → `temp.update` | faithful. **eng:** the parents are a random subset of the new population. Step 10 is not used (the paper describes it for IEA). |
| 6 mutation with p_m on the population | no best-individual exemption | faithful |
| 7 termination | as for IEA. At the end, the completed E′ is merged into the output. | faithful + **eng** |
| "an additional external set" | `archive`: every evaluated non-dominated point, unbounded, and **never used by selection** | `test_working_elite_is_bounded_and_separate_from_archive`, `test_archive_hypervolume_is_monotone` | faithful to the optional set. Its memory cost is reported as `archive_bytes`. |

## Section V: benchmark encoding and reproduction

| Paper | Meaning | Code | Test | Status |
|---|---|---|---|---|
| Table IV, Section V | f1…f12 on real parameters, fixed bits per parameter | `benchmarks.PaperBenchmark`, `RealEncoder` | `test_paper_f1_reaches_near_optimum` | faithful functions and domains |
| not stated | how bits map to a parameter value | `PaperBenchmark.encoder()` uses `RealEncoder(gray=True)`: reflected Gray code, `x = low + int / (2^bits − 1) · (high − low)` | `test_gray_decoding_makes_neighbours_one_bit_apart` | **eng.** With plain binary, Hamming cliffs (e.g. 5-bit `00111` → `01000`) block bit-flip mutation once the population converges: Table III gave 120.42 (paper 120.890) and Section V-A never reached 121.598 in 400 000 calls. With Gray code: 120.80 [120.53, 120.99] and 10/10 runs reach 121.598 in 59 196 calls on average (paper 58 156). `RealEncoder` itself defaults to plain binary. |

`reproduce/iea_paper.py` reruns Tables III, V, VI and Section V-A; results are in `reproduce/results/`.

**f8 and f12, investigated.** `reproduce/iea_f8_f12.py` tests the hypotheses below with 30 IEA runs at D = 10 and 100 (results in `reproduce/results/iea_f8_f12.md`). The variants live in that script only: `PaperBenchmark` keeps the Table IV formulas, which were checked against the page image and match the standard Ackley and Griewank.

| | Paper (Tables V / VI) | Finding | Status |
|---|---|---|---|
| f8, D = 10 | 1.00 [0.09, 1.73] | On pyiea's grid `x = low + k·(high − low)/(2^b − 1)` with 10 bits, the values closest to 0 are ±0.0293, which gives a floor of 0.1626. Every pyiea run ends exactly there, so the paper's minimum 0.09 is off this grid. With a grid that divides by 2^b instead, x = 0 is on the grid: IEA (Gray) gives 0.0031 [0, 0.092]. With 16 bits it gives 0.037 [0.009, 0.12]. Either change explains a minimum below 0.163. The paper's mean (1.00) is worse than every variant tried (0.003–0.36). | **explained in part**: the paper's grid is probably not pyiea's; which grid it used is not identifiable |
| f8, D = 100 | 3.69 [2.71, 4.81] | Every variant gives 8.8–9.0 (2^b vs 2^b − 1, binary vs Gray, 10 vs 16 bits). | **未解 (unresolved)** |
| f12, D = 10 | 0.999 [0.9994, 0.9996] | With the product over `cos(x_i)/sqrt(i)` instead of Table IV's `cos(x_i/sqrt(i))`, the minimum (x = 0) is 1 − 1/√(10!) = 0.999475, the paper's printed minimum to four digits. IEA then gives 0.9995 [0.9995, 0.9995] in both encodings. Table IV's own formula gives 0.108 [0.037, 0.351]. | **explained**: the paper's numbers fit a `cos(x_i)/sqrt(i)` implementation. Table IV is unchanged here, and the variant is documented only. |
| f12, D = 100 | 32.86 [14.10, 74.85] | The product term is negligible at D = 100, so both formulas give the same values: 6.38 [3.28, 12.4] with Gray and 3.34 [1.72, 6.08] with binary. pyiea's IEA gets closer to the optimum than the paper's. | no formula difference; the gap is optimizer performance |

The earlier committed Tables V and VI had stale f8 rows (D = 10: 0.1957 [0.1626, 1.157]; D = 100: 8.853). The committed code, before and after this work, gives 0.1626 [0.1626, 0.1626] and 8.824, and the tables were regenerated.

**IBCGA-style loop, checked and not adopted.** A generation loop in the style of the lab's IBCGA (binary tournament on a copy of the population, IGC with probability pc, per-bit mutation, then keep the best N_pop of parents plus children) was compared with Steps 1–6 on f1, D = 100 at the paper's settings. It was worse in every case: 5 bits and 12 000 calls gave 117–119 (Steps 1–6: 120.4–121.0), and 14 bits and 150 000 calls reached 121.598 in 0/32 runs (Steps 1–6 with Gray code: 16/16).

### Section V-B, Fig. 3: IGC vs conventional crossover

Checked against the page images of pp. 532–533 (Fig. 3, eq. 6, the settings paragraph).

| Paper | Meaning | Code | Test | Status |
|---|---|---|---|---|
| "For all compared EAs, N_pop = 10, ps = 0.2, pc = 0.8, pm = 0.05", N_eval 10 000; D = 10, 20, …, 100; 30 runs | settings | `reproduce/iea_fig3.py` | — | faithful |
| eq. (6) `dist(D) = |f_opt(D) − f_best(D)| / D` | mean distance per parameter | `run_one` (`dist`), averaged over runs | — | faithful. f2's optimum is only "approximately 2D" (Table IV), so its dist(D) is approximate. |
| "EAs with elitist strategy and the associated crossover": (OEGA, one-point), (TEGA, two-point), (UEGA, uniform) | the same EA with IGC swapped for a conventional crossover | `baselines.elitist_ga(crossover=...)`, `baselines.CROSSOVERS` | `test_crossover_children_are_complementary`, `test_elitist_ga_keeps_the_best_and_the_budget`, `test_same_initial_population_as_iea` | **baseline**, never IEA. **eng:** the loop mirrors IEA Steps 1–6 (the same truncation refill, pc·N_pop parents paired in draw order, the same per-bit mutation). The best individual is the elite: it is never recombined or mutated. Children are evaluated after mutation, as in a conventional GA. For a given seed the initial population equals IEA's. |
| pm for the compared GAs | not defined separately | pm = 0.05 per bit, as for IEA | — | **eng** (reading). At pm = 0.05 per bit, every GA child loses about 5% of its bits before its first evaluation, while IGC evaluates its children before Step 5. Because the paper's GA curves lie between pyiea's GAs at pm = 0.005 and at 1/n_bits, the script also runs an **ablation** with pm = 1/n_bits at D = 10, 50, 100. It is reported separately and is not the paper's setting. |
| BLX-α GA, OGA, BOA | — | not run | — | out of scope (the task says they are optional) |

Results (`reproduce/results/iea_fig3.md`; 30 runs per cell, 10 000 calls each):

- **Paper settings (pm = 0.05 for every EA).** IEA has the smallest mean dist(D) of the four methods at all 120 (function, D) cells, D = 10 included. The paper's other finding, that IEA does not lead at small D (Table V: IEA ranked 6, OEGA 2), is **not reproduced** under this reading of pm. pyiea's GAs at D = 10 are also weaker than the paper's on some functions (f3: 3.3 vs about 0; f4: 1.5 vs 0.7–1.1).
- **Ablation, GA pm = 1/n_bits.** Here the GAs tie IEA at D = 10 on f1, f3, f5 and f8, and beat it on f10 (0.068 vs 0.29). IEA's lead grows with D, as the paper describes. The exceptions are f8 (the GAs win at D = 50) and f10 (the GAs win at D = 10 and 50).
- **IEA's own curve against Fig. 3.** At D = 100 the pyiea IEA is as good as or better than the paper's IEA: f1 0.0020 (paper 0.0065), f3 3.6 (≈ 5), f4 1.5 (2.1), f9 70 (80), f12 0.064 (0.33). f8 is the exception, at 0.088 vs 0.037; see the f8 entry below.

### Section V-C, Table VII: IEA vs OGA/Q (deferred)

Not reproduced, and deliberately deferred. Two reasons:

- **It needs a feature outside pyiea's focus on 0/1 genomes.** Table VII starts IEA from an OA-based initial population in the style of OGA/Q (Leung & Wang 2001, [8]): quantized 14-bit real parameters, Q² points per subspace, and one main-effect point per subspace, for S(Q² + 1) evaluations. That only matters for continuous numerical optimization. pyiea has no such option; random initialization (Step 1) is the only one.
- **Part of it cannot be reproduced exactly.** In [8], g8's χ_ij and φ_ij are random integers in [−100, 100] and ω_j is a random number in [−π, π]. Their values are not printed, so g8 can be matched only in distribution. The IEA column of Table VII, however, reports exact values for it (402 064 evaluations, 1.6999×10⁻⁷).

Everything else Table VII would exercise (IGC, IEA Steps 1–6) is already checked by Tables III, V and VI, Section V-A and Fig. 3. Revisit this if a complete reproduction of the paper is required.

## Section VI-A: multi-objective 0/1 knapsack

Checked against the page images of pp. 535–537 (Figs. 4–5, eqs. 8–9, Table VIII).

| Paper | Meaning | Code | Test | Status |
|---|---|---|---|---|
| eq. (9) | I knapsacks, J items, maximize every `f_i(x) = Σ_j p_ij x_j` s.t. `Σ_j w_ij x_j ≤ c_i` | `benchmarks.MultiKnapsack` returns `(−f_1, …, −f_I)` (minimized) | `test_objective_is_negated_profit_of_repaired_solution` | faithful |
| "the test data sets are available from the authors [27]" | Zitzler & Thiele `knapsack.<J>.<I>` | `MultiKnapsack.from_zt_file`; `reproduce/imoea_knapsack.py fetch` (SHA-256-checked) | `test_reads_zitzler_thiele_format` | The ETH site was unreachable, so the I = 2, 3 files come from the moead-framework/data mirror and their checksums pin that copy. No I = 4 copy was found. Those instances use `MultiKnapsack.random` (eq. 9 rule, seeded) and are reported as **non-paper data**. |
| repair of [27]: remove items step by step, ordered by the maximum profit/weight ratio, until every constraint holds | `MultiKnapsack.repair`: remove selected items in increasing `q_j = max_i p_ij / w_ij` and stop at the first feasible point | `test_repair_removes_the_shortest_greedy_prefix` | faithful to the paper's text. **eng:** ties in `q_j` go by item index. The direction (lowest ratio first) follows [27] as quoted in the task, because the IEA paper does not state it. |
| repair: where the result goes | the repair happens inside the objective. The genome is **not** rewritten, and IGC, mutation and E all act on the unrepaired bits | `MultiKnapsack.__call__` | the same test (`s` stays read-only) | **eng.** Neither the IEA paper nor the task text says the repaired string is written back, and [27] could not be re-read here. This deliberately departs from the "feasible by construction" rule for `Problem`: the genotype space is all of `{0,1}^J`, and feasibility belongs to the decoder. |
| Table VIII: N = 15 gene segments ("divided into N = 15 gene segments", p. 536) | fixed division size **specified by the paper** for this experiment | `IMOEAConfig(max_segments=15)` | — | faithful to Table VIII. This is not the `bounded_segments` variant: the paper fixes N itself. **eng:** when fewer than 16 bits differ, `N = 2^⌊log2(M+1)⌋ − 1` (< 15) is used. The report gives the share of such IGCs. |
| Table VIII: N_pop 50, N_Emax 50, ps 0.2, pc 0.8, pm 0.01; N_eval 75k–175k; 30 runs | settings | `reproduce/imoea_knapsack.py` | — | faithful |
| eq. (8) `C(A, B)` = share of B weakly dominated by A | cover metric | `pareto.coverage` (`a ≤ b` in every objective) | `test_cover_metric_eq8_uses_weak_dominance` | faithful. **eng (report):** fronts are reduced to distinct objective vectors before C is computed. |
| Fig. 4: "direct comparisons of each independent run" | run r of A against run r of B, giving 30 values per ordered pair | `cover_paired` in the script | — | faithful as far as the text allows. **eng:** runs are paired by seed index. |
| compared EAs: SPEA, SPEA2, NSGA2 results "gleaned from the authors' website" | — | not reproduced: the ETH site is unreachable, and SPEA2 is not reimplemented | — | The comparisons use pyiea's own `baselines.nsga2` (N_pop 50, pc 0.8, pm 0.01, uniform crossover, the same repair and budget) and `random_search`. Neither is the paper's NSGA2 setup ([27] settings, one-point crossover). |

Results (`reproduce/results/imoea_knapsack.md`; 30 runs per instance, every run used exactly its Table VIII budget):

- **IMOEA vs the pyiea NSGA-II baseline.** On 8 of the 9 instances IMOEA covers much of NSGA-II's front and is almost never covered: median C(IMOEA, NSGA-II) / C(NSGA-II, IMOEA) is 0.29 / 0.03 (500.2), 0.38 / 0.00 (750.2), 0.34–0.77 / 0.00 (I = 3), and 0.48–0.88 / 0.00 (I = 4, non-paper data). The margin grows with J, as the paper reports. The exception is 250.2, at 0.04 / 0.76: NSGA-II wins there.
- **E deduplication (`imoea_unique`, eng option).** It helps most where the repair maps many genomes to one solution (2 knapsacks). On 250.2 it reverses the result, to 0.35 / 0.11 against NSGA-II. With I = 3, 4 and larger J the two IMOEA variants are close.
- **Fig. 5 (750.2, union of 30 runs).** IMOEA spans f1 27 377–28 314 and f2 27 317–28 248; the paper's IMOEA spans about 27 000–28 350 and 26 600–28 330. The pyiea NSGA-II front is wider (f1 26 228–28 869) but dominated in the middle.
- **Not reproduced.** The SPEA, SPEA2 and NSGA2 fronts from the ETH site (unreachable), and the three I = 4 instances (no copy found).

## Section VI-B: ZDT1–ZDT6 with m = 63

Checked against the page images of pp. 537–538 (eq. 10, Table VIII, Figs. 6–11).

| Paper | Meaning | Code | Test | Status |
|---|---|---|---|---|
| eq. (10), "six test problems ZDT1…ZDT6 can be retrieved from [41]" | `T(X) = (f1(x1), f2(X))`, `f2 = g·h` | `benchmarks.ZDT` after Zitzler, Deb & Thiele (2000) | `test_real_zdt_by_hand`, `test_zdt5_unitation_by_hand` | faithful to ZDT (2000). The formulas are not printed in the IEA paper. |
| "extended test problems with a large number of parameters (m = 63)" | the same `g` formulas with `m − 1 = 62` | `ZDT(name, m=63)` | `test_paper_sizes_m63` | faithful. ZDT4 keeps `x2..xm ∈ [−5, 5]`. |
| "each parameter … 30 bits, except x2…xm … 5 bits for ZDT5" | 1 890-bit genomes; ZDT5 has 30 + 62·5 = 340 bits | `ZDT.problem()` (`ParameterProblem` widths) | the same test | faithful. x1 of ZDT5 has 30 bits, as in ZDT (2000). |
| encoding of the 30-bit parameters | not stated | `ZDT.encoder(gray)`: plain binary or reflected Gray | `test_encoded_objective_decodes_then_evaluates` | **eng.** Both encodings are run and reported (an ablation). ZDT5 has no real encoding. |
| Table VIII: N = m = 63 | one gene segment per parameter | `IMOEAConfig(max_segments=63)` with `ParameterProblem` | — | faithful. When fewer than 63 parameters differ, Section III-A's `N = 2^⌊log2(M+1)⌋ − 1` applies (**eng**, the same rule as elsewhere). |
| Table VIII: N_pop 30, N_Emax 30, ps 0.2, pc 0.6, pm 0.01, N_eval 25 000, 30 runs | settings | `reproduce/imoea_zdt.py` | — | faithful |
| the curve in Figs. 6–11 (except ZDT3) is the Pareto-optimal front | `g` at its minimum | `ZDT.pareto_front()` | `test_fronts_are_analytic_optima` | faithful. ZDT5's front is `f2 = (m − 1)/f1`. The curve drawn in Fig. 10 looks like the m = 11 front `10/f1`, not `62/f1` (a reading, not verified). |
| compared EAs: VEGA … SPEA2 with the [26] settings (pc 0.8, pm 0.1, N_pop 100) | — | not reproduced | — | The comparison uses pyiea's `baselines.nsga2` with N_pop 100, pc 0.8 and pm = 1/n_bits. As a per-bit rate, the quoted pm = 0.1 would flip about 189 of 1 890 bits per child. Uniform crossover. Two baseline checks read the quoted pm = 0.1 per bit (`nsga2_pm0.1`) and per parameter, i.e. 0.1/30 per bit (`nsga2_pm0.1var`, **eng** reading). They are baselines, not the paper's NSGA2. |
| comparison | cover metric (eq. 8) and the merged fronts of Figs. 6–11 | `reproduce/imoea_zdt.py` | — | as in the paper. In addition: hypervolume with one fixed reference point per problem (**eng**, near the values of a random genome), and IGD/GD against the analytic front. |

Results (`reproduce/results/imoea_zdt.md`; 30 runs per case, 25 000 calls each):

- **Encoding (eng ablation).** For IMOEA, plain binary beats Gray on ZDT1, 2, 3 and 6 (C(binary, Gray) ≈ 0.5–0.8, C(Gray, binary) ≈ 0). Gray wins on ZDT4 (merged f2 24.5–33 vs 61). This is the opposite of the single-objective benchmarks, where only Gray matches the paper.
- **IMOEA against the paper's figures.** ZDT4, Gray: f2 24.5–33 for f1 in 0–0.5 (paper ≈ 50). ZDT6, binary: f2 2.45–2.91 (paper 2.1–2.6). ZDT1 and 2, binary: merged GD 0.011 and 0.016 from the front (paper "very close"). ZDT3: all five pieces are covered. ZDT4 and ZDT6 fronts are sparse (1–9 merged points), as in Figs. 9 and 11.
- **IMOEA vs the pyiea NSGA-II baseline.** In Gray encoding, IMOEA dominates on ZDT1, 2, 3, 5 and 6 (median C(IMOEA, NSGA-II) 0.75–1.0, reverse ≈ 0). In binary encoding, NSGA-II has the better hypervolume and IGD on ZDT1, 2 and 6, and covers IMOEA on ZDT6 (1.0). **The paper's claim that only IMOEA gets near the ZDT4 and ZDT6 fronts is not reproduced against this baseline.** pyiea's NSGA-II (N_pop 100, pm = 1/n_bits) reaches f2 ≈ 28–34 on ZDT4 and 2.2–2.7 on ZDT6, far better than the NSGA2 curves in the paper (≥ 140 and ≈ 4). The paper's compared EAs used the [26] settings with pm = 0.1.
- **Is the gap the baseline's settings? (ZDT4 and ZDT6, same budget and seeds.)** Only the baseline's pm changes; IMOEA keeps Table VIII. With the quoted pm = 0.1 read per bit, NSGA-II falls to merged f2 571–829 on ZDT4 and 6.6–7.6 on ZDT6, in both encodings, and IMOEA covers it (median C(IMOEA, NSGA-II) 0.37–1.0, reverse 0). That meets the paper's "others f2 ≥ 140" on ZDT4 but overshoots the paper's NSGA2 on ZDT6 (≈ 4–4.3). Read per parameter (0.1/30 per bit), NSGA-II stays close to the 1/n_bits baseline (ZDT4 f2 29–98, ZDT6 2.4–3.5). So the baseline's mutation rate alone moves NSGA-II from better than the paper's curves to worse than them: the ZDT4/ZDT6 difference is a baseline-setting question, not an IMOEA one. Neither reading reproduces the paper's NSGA2 exactly; its operators and encoding are not stated.

## Paper parameters vs prototype defaults

| | Paper | Prototype |
|---|---|---|
| IEA (Section V) | ps 0.2, pc 0.8, pm 0.05; N_pop 30 (Table III) / 10 (Fig. 3) | `IEAConfig` defaults are ps 0.2, pc 0.8, pm 0.05, N_pop 30 |
| IMOEA ZDT (Table VIII) | N_pop 30, N_Emax 30, ps 0.2, pc 0.6, pm 0.01 | `IMOEAConfig` defaults are the same |
| IMOEA knapsack (Table VIII) | N_pop 50, N_Emax 50, N = 15, ps 0.2, pc 0.8, pm 0.01 | set explicitly in `reproduce/imoea_knapsack.py` |
| budgets | 12k–25k evaluations | the benchmarks use 512 and 5000 calls, both prototype settings |

## Problem-specific operators

- `FixedCardinalityProblem.divide`: the PAP criterion of Section III-A (see the table above).
- `FixedCardinalityProblem.mutate`: a swap mutation that keeps the cardinality. **eng:** each 1-bit, with probability p_m, swaps with a random 0-bit, so p_m is a per-bit rate as in the bit-inverse mutation of Step 5.

## Engineering rules the papers do not cover

- **Stall stop:** `max_stall_generations` (default 50) stops a run after that many generations without a new objective call. This keeps a fully cached search space from looping. The baselines use the same rule.
- **Resume:** the kind, seed, algorithm parameters and evaluator context must match the checkpoint; the stopping conditions may differ, so that a run can be extended (`checkpoint.STOPPING_FIELDS`).
- **Fail fast:** if the objective fails (exception, NaN or inf, or an invalid genome) on every genome of the initial population, IEA and IMOEA raise `EvaluationError` with the first failure, instead of spending the budget on failures. Occasional failures still just count as failed evaluations (`test_fitness_failing_everywhere_fails_fast_with_the_first_error`).
- **Objective count:** an objective that returns a different number of values than `n_objectives` is a usage error and raises `ValueError` at once (`test_wrong_number_of_objectives_is_a_clear_error`).
- **Budget:** an IGC that does not fit the remaining call budget returns `status="budget"` without evaluating anything, and the run stops (`BudgetExhaustedError` is the low-level signal).
- `*.crossover`: uniform crossover. Only the vanilla GA and NSGA-II **baselines** use it; IEA and IMOEA never do.
