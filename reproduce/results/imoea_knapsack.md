## IMOEA paper, Section VI-A: multi-objective 0/1 knapsack

Cover metric C(A, B) (eq. 8) = fraction of B's front weakly dominated by A's front; median [first quartile, third quartile] over the runs, run r of A against run r of B (Fig. 4). Fronts are the distinct non-dominated objective vectors each run evaluated. Every run used exactly the Table VIII budget.

### knapsack.250.2 (Zitzler-Thiele data), N_eval = 75,000, 30 runs

| pair | C(A, B) | C(B, A) |
|---|---|---|
| A = imoea, B = nsga2 | 0.04 [0.00, 0.15] | 0.76 [0.45, 0.95] |
| A = imoea, B = random | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| A = imoea_unique, B = nsga2 | 0.35 [0.26, 0.41] | 0.11 [0.08, 0.16] |
| A = imoea, B = imoea_unique | 0.00 [0.00, 0.02] | 1.00 [0.96, 1.00] |

| method | front size median [q1, q3] | mean calls | mean wall s/run | stop |
|---|---|---|---|---|
| imoea | 26.50 [24, 29] | 74993 | 28.4 | budget_exhausted; IGCs with N < 15: 16.5% |
| imoea_unique | 44 [39, 49] | 74992 | 7.6 | budget_exhausted; IGCs with N < 15: 5.2% |
| nsga2 | 82 [75, 86] | 75000 | 11.6 | budget_exhausted |
| random | 7 [6, 9] | 75000 | 4.8 | budget_exhausted |

### knapsack.500.2 (Zitzler-Thiele data), N_eval = 100,000, 30 runs

| pair | C(A, B) | C(B, A) |
|---|---|---|
| A = imoea, B = nsga2 | 0.29 [0.26, 0.33] | 0.03 [0.00, 0.07] |
| A = imoea, B = random | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| A = imoea_unique, B = nsga2 | 0.43 [0.39, 0.47] | 0.00 [0.00, 0.02] |
| A = imoea, B = imoea_unique | 0.05 [0.00, 0.22] | 0.82 [0.51, 1.00] |

| method | front size median [q1, q3] | mean calls | mean wall s/run | stop |
|---|---|---|---|---|
| imoea | 35 [30, 37.75] | 99993 | 12.3 | budget_exhausted; IGCs with N < 15: 2.8% |
| imoea_unique | 43.50 [41.25, 48.50] | 99995 | 8.5 | budget_exhausted; IGCs with N < 15: 1.6% |
| nsga2 | 75.50 [71, 85] | 100000 | 16.8 | budget_exhausted |
| random | 6 [4.25, 7] | 100000 | 7.2 | budget_exhausted |

### knapsack.750.2 (Zitzler-Thiele data), N_eval = 125,000, 30 runs

| pair | C(A, B) | C(B, A) |
|---|---|---|
| A = imoea, B = nsga2 | 0.38 [0.33, 0.41] | 0.00 [0.00, 0.00] |
| A = imoea, B = random | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| A = imoea_unique, B = nsga2 | 0.47 [0.43, 0.54] | 0.00 [0.00, 0.00] |
| A = imoea, B = imoea_unique | 0.05 [0.00, 0.32] | 0.81 [0.42, 0.97] |

| method | front size median [q1, q3] | mean calls | mean wall s/run | stop |
|---|---|---|---|---|
| imoea | 40 [34.75, 44.75] | 124994 | 12.9 | budget_exhausted; IGCs with N < 15: 1.4% |
| imoea_unique | 46.50 [43.25, 52.75] | 124991 | 11.4 | budget_exhausted; IGCs with N < 15: 0.8% |
| nsga2 | 76.50 [74.25, 80] | 125000 | 26.6 | budget_exhausted |
| random | 5.50 [4, 6.75] | 125000 | 11.7 | budget_exhausted |

### knapsack.250.3 (Zitzler-Thiele data), N_eval = 100,000, 30 runs

| pair | C(A, B) | C(B, A) |
|---|---|---|
| A = imoea, B = nsga2 | 0.34 [0.31, 0.39] | 0.00 [0.00, 0.00] |
| A = imoea, B = random | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| A = imoea_unique, B = nsga2 | 0.42 [0.40, 0.46] | 0.00 [0.00, 0.00] |
| A = imoea, B = imoea_unique | 0.05 [0.01, 0.10] | 0.78 [0.59, 0.94] |

| method | front size median [q1, q3] | mean calls | mean wall s/run | stop |
|---|---|---|---|---|
| imoea | 168 [151, 188.75] | 99994 | 29.9 | budget_exhausted; IGCs with N < 15: 27.0% |
| imoea_unique | 281 [254.25, 293.75] | 99993 | 13.3 | budget_exhausted; IGCs with N < 15: 8.5% |
| nsga2 | 603.50 [561.50, 655] | 100000 | 19.6 | budget_exhausted |
| random | 26 [20.50, 28] | 100000 | 7.1 | budget_exhausted |

### knapsack.500.3 (Zitzler-Thiele data), N_eval = 125,000, 30 runs

| pair | C(A, B) | C(B, A) |
|---|---|---|
| A = imoea, B = nsga2 | 0.55 [0.51, 0.59] | 0.00 [0.00, 0.00] |
| A = imoea, B = random | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| A = imoea_unique, B = nsga2 | 0.57 [0.53, 0.59] | 0.00 [0.00, 0.00] |
| A = imoea, B = imoea_unique | 0.20 [0.15, 0.40] | 0.36 [0.23, 0.61] |

| method | front size median [q1, q3] | mean calls | mean wall s/run | stop |
|---|---|---|---|---|
| imoea | 249 [219.25, 275.75] | 124992 | 13.8 | budget_exhausted; IGCs with N < 15: 1.7% |
| imoea_unique | 266 [220.75, 276.50] | 124993 | 12.8 | budget_exhausted; IGCs with N < 15: 1.4% |
| nsga2 | 681 [648.50, 714.50] | 125000 | 26.3 | budget_exhausted |
| random | 29 [21.50, 34] | 125000 | 10.1 | budget_exhausted |

### knapsack.750.3 (Zitzler-Thiele data), N_eval = 150,000, 30 runs

| pair | C(A, B) | C(B, A) |
|---|---|---|
| A = imoea, B = nsga2 | 0.77 [0.73, 0.80] | 0.00 [0.00, 0.00] |
| A = imoea, B = random | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| A = imoea_unique, B = nsga2 | 0.78 [0.75, 0.82] | 0.00 [0.00, 0.00] |
| A = imoea, B = imoea_unique | 0.23 [0.14, 0.45] | 0.28 [0.21, 0.55] |

| method | front size median [q1, q3] | mean calls | mean wall s/run | stop |
|---|---|---|---|---|
| imoea | 190 [176.25, 210] | 149994 | 17.9 | budget_exhausted; IGCs with N < 15: 0.8% |
| imoea_unique | 194 [182.25, 206.50] | 149991 | 15.9 | budget_exhausted; IGCs with N < 15: 0.7% |
| nsga2 | 594 [562.50, 635.50] | 150000 | 41.6 | budget_exhausted |
| random | 21 [17, 25.75] | 150000 | 15.3 | budget_exhausted |

### knapsack.250.4 (**non-paper data**: seeded eq. 9), N_eval = 125,000, 30 runs

| pair | C(A, B) | C(B, A) |
|---|---|---|
| A = imoea, B = nsga2 | 0.48 [0.45, 0.51] | 0.00 [0.00, 0.00] |
| A = imoea, B = random | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| A = imoea_unique, B = nsga2 | 0.52 [0.50, 0.56] | 0.00 [0.00, 0.00] |
| A = imoea, B = imoea_unique | 0.08 [0.04, 0.17] | 0.33 [0.26, 0.60] |

| method | front size median [q1, q3] | mean calls | mean wall s/run | stop |
|---|---|---|---|---|
| imoea | 785.50 [670.25, 870] | 124995 | 58.8 | budget_exhausted; IGCs with N < 15: 17.9% |
| imoea_unique | 948.50 [835.75, 1049.50] | 124993 | 27.8 | budget_exhausted; IGCs with N < 15: 9.0% |
| nsga2 | 1901 [1826.25, 2074.25] | 125000 | 40.7 | budget_exhausted |
| random | 62.50 [52.75, 73.75] | 125000 | 10.3 | budget_exhausted |

### knapsack.500.4 (**non-paper data**: seeded eq. 9), N_eval = 150,000, 30 runs

| pair | C(A, B) | C(B, A) |
|---|---|---|
| A = imoea, B = nsga2 | 0.75 [0.73, 0.76] | 0.00 [0.00, 0.00] |
| A = imoea, B = random | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| A = imoea_unique, B = nsga2 | 0.75 [0.73, 0.77] | 0.00 [0.00, 0.00] |
| A = imoea, B = imoea_unique | 0.29 [0.15, 0.37] | 0.21 [0.13, 0.33] |

| method | front size median [q1, q3] | mean calls | mean wall s/run | stop |
|---|---|---|---|---|
| imoea | 817 [785.25, 869.25] | 149991 | 20.3 | budget_exhausted; IGCs with N < 15: 1.7% |
| imoea_unique | 814 [721.50, 899.75] | 149995 | 20.3 | budget_exhausted; IGCs with N < 15: 1.6% |
| nsga2 | 2339 [2254, 2384] | 150000 | 64.4 | budget_exhausted |
| random | 70.50 [60.50, 79.50] | 150000 | 18.8 | budget_exhausted |

### knapsack.750.4 (**non-paper data**: seeded eq. 9), N_eval = 175,000, 30 runs

| pair | C(A, B) | C(B, A) |
|---|---|---|
| A = imoea, B = nsga2 | 0.88 [0.86, 0.89] | 0.00 [0.00, 0.00] |
| A = imoea, B = random | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| A = imoea_unique, B = nsga2 | 0.89 [0.86, 0.91] | 0.00 [0.00, 0.00] |
| A = imoea, B = imoea_unique | 0.23 [0.06, 0.34] | 0.29 [0.12, 0.52] |

| method | front size median [q1, q3] | mean calls | mean wall s/run | stop |
|---|---|---|---|---|
| imoea | 604 [528, 644.75] | 174993 | 25.3 | budget_exhausted; IGCs with N < 15: 0.8% |
| imoea_unique | 625.50 [574, 709.50] | 174993 | 23.1 | budget_exhausted; IGCs with N < 15: 0.7% |
| nsga2 | 2011 [1952, 2108.25] | 175000 | 37.4 | budget_exhausted |
| random | 63 [54, 71.50] | 175000 | 11.1 | budget_exhausted |

### knapsack.750.2: union of the 30 runs' fronts (cf. Fig. 5)

| method | points | f1 range | f2 range |
|---|---|---|---|
| imoea | 80 | 27377-28314 | 27317-28248 |
| imoea_unique | 99 | 27275-28394 | 27230-28332 |
| nsga2 | 120 | 26228-28869 | 25966-28715 |
| random | 9 | 21440-23273 | 21671-22947 |

Paper, Fig. 5 (read off the plot, merged over 30 runs): IMOEA about f1 27 000-28 350, f2 26 600-28 330; NSGA2 (ETH results, [27] settings) about f1 25 950-28 480, f2 25 200-28 280.

_pyiea 0.1.0_
