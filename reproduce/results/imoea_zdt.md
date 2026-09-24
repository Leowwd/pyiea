## IMOEA paper, Section VI-B: ZDT1-ZDT6 with m = 63 parameters

N_eval = 25,000 per run; 30 bits per parameter (ZDT5: x1 30 bits, x2..x63 5 bits). Every value is median [first quartile, third quartile] over the runs. HV: larger is better (reference point per problem in the json). IGD and GD: smaller is better. C(A, B) (eq. 8): run r of A against run r of B. "merged" is the non-dominated union of all runs, as in Figs. 6-11.

### zdt1/binary (30 runs)

Paper (Figs. 6-11): IMOEA very close to the front (about 0.05 above); SPEA2/NSGA2 about 0.1-0.2 above.

| method | HV | IGD | GD | merged: points, f1 range, f2 min / median / max, GD | calls | s/run |
|---|---|---|---|---|---|---|
| imoea | 6.212 [6.21, 6.219] | 0.0296 [0.0272, 0.0327] | 0.017 [0.0151, 0.0186] | 320, 3.17e-08-0.992, 0.0146 / 0.46 / 1.03, 0.0114 | 24986 | 3.0 |
| nsga2 | 6.238 [6.235, 6.241] | 0.0177 [0.0153, 0.0194] | 0.0175 [0.0151, 0.0191] | 160, 0-0.992, 0.0129 / 0.378 / 1.02, 0.00886 | 25000 | 4.3 |

| cover | value |
|---|---|
| C(imoea,nsga2) | 0.07 [0.05, 0.16] |
| C(nsga2,imoea) | 0.28 [0.20, 0.41] |

### zdt1/gray (30 runs)

Paper (Figs. 6-11): IMOEA very close to the front (about 0.05 above); SPEA2/NSGA2 about 0.1-0.2 above.

| method | HV | IGD | GD | merged: points, f1 range, f2 min / median / max, GD | calls | s/run |
|---|---|---|---|---|---|---|
| imoea | 6.161 [6.143, 6.167] | 0.0551 [0.0515, 0.0625] | 0.0509 [0.0465, 0.0572] | 212, 3.53e-06-0.995, 0.0385 / 0.47 / 1.1, 0.0367 | 24979 | 2.8 |
| nsga2 | 6.064 [6.044, 6.085] | 0.134 [0.118, 0.147] | 0.131 [0.119, 0.147] | 360, 1.86e-09-1, 0.0926 / 0.495 / 1.3, 0.0892 | 25000 | 4.7 |

| cover | value |
|---|---|
| C(imoea,nsga2) | 0.97 [0.95, 0.98] |
| C(nsga2,imoea) | 0.00 [0.00, 0.02] |
| C(imoea gray,imoea binary) | 0.00 [0.00, 0.00] |
| C(imoea binary,imoea gray) | 0.79 [0.73, 0.85] |

### zdt2/binary (30 runs)

Paper (Figs. 6-11): IMOEA close to the front (about 0.1 above); the others 0.3-0.5 above.

| method | HV | IGD | GD | merged: points, f1 range, f2 min / median / max, GD | calls | s/run |
|---|---|---|---|---|---|---|
| imoea | 5.84 [5.821, 5.853] | 0.0561 [0.0453, 0.0656] | 0.0219 [0.0196, 0.0255] | 116, 0-0.993, 0.0656 / 0.614 / 1.01, 0.0162 | 24985 | 2.6 |
| nsga2 | 5.889 [5.884, 5.895] | 0.0241 [0.0208, 0.0276] | 0.0235 [0.0202, 0.027] | 127, 0-0.992, 0.046 / 0.763 / 1.02, 0.0147 | 25000 | 4.3 |

| cover | value |
|---|---|
| C(imoea,nsga2) | 0.07 [0.01, 0.17] |
| C(nsga2,imoea) | 0.31 [0.07, 0.67] |

### zdt2/gray (30 runs)

Paper (Figs. 6-11): IMOEA close to the front (about 0.1 above); the others 0.3-0.5 above.

| method | HV | IGD | GD | merged: points, f1 range, f2 min / median / max, GD | calls | s/run |
|---|---|---|---|---|---|---|
| imoea | 5.748 [5.734, 5.762] | 0.0983 [0.0932, 0.11] | 0.0908 [0.0805, 0.105] | 89, 0-1, 0.114 / 0.658 / 1.07, 0.0652 | 24985 | 2.7 |
| nsga2 | 5.599 [5.57, 5.657] | 0.219 [0.177, 0.24] | 0.207 [0.168, 0.226] | 132, 0-1, 0.217 / 0.803 / 1.13, 0.105 | 25000 | 4.4 |

| cover | value |
|---|---|
| C(imoea,nsga2) | 0.97 [0.90, 0.99] |
| C(nsga2,imoea) | 0.00 [0.00, 0.03] |
| C(imoea gray,imoea binary) | 0.00 [0.00, 0.00] |
| C(imoea binary,imoea gray) | 0.81 [0.71, 0.95] |

### zdt3/binary (30 runs)

Paper (Figs. 6-11): IMOEA, SPEA2 and NSGA2 all well distributed over the five pieces.

| method | HV | IGD | GD | merged: points, f1 range, f2 min / median / max, GD | calls | s/run |
|---|---|---|---|---|---|---|
| imoea | 6.599 [6.578, 6.612] | 0.0562 [0.049, 0.0636] | 0.00851 [0.00666, 0.0123] | 337, 9.31e-10-0.848, -0.756 / 0.261 / 1.05, 0.00369 | 24984 | 2.7 |
| nsga2 | 6.647 [6.635, 6.656] | 0.0277 [0.024, 0.0328] | 0.0148 [0.013, 0.0172] | 237, 0-0.855, -0.734 / 0.301 / 1.06, 0.00941 | 25000 | 4.1 |

| cover | value |
|---|---|
| C(imoea,nsga2) | 0.47 [0.39, 0.51] |
| C(nsga2,imoea) | 0.03 [0.02, 0.07] |

### zdt3/gray (30 runs)

Paper (Figs. 6-11): IMOEA, SPEA2 and NSGA2 all well distributed over the five pieces.

| method | HV | IGD | GD | merged: points, f1 range, f2 min / median / max, GD | calls | s/run |
|---|---|---|---|---|---|---|
| imoea | 6.598 [6.569, 6.615] | 0.0554 [0.052, 0.0604] | 0.0232 [0.0201, 0.0273] | 236, 1.3e-06-0.852, -0.729 / 0.192 / 1.1, 0.0121 | 24984 | 3.1 |
| nsga2 | 6.421 [6.399, 6.453] | 0.134 [0.121, 0.143] | 0.118 [0.101, 0.133] | 401, 3.73e-09-0.852, -0.597 / 0.39 / 1.39, 0.083 | 25000 | 4.4 |

| cover | value |
|---|---|
| C(imoea,nsga2) | 0.88 [0.82, 0.94] |
| C(nsga2,imoea) | 0.00 [0.00, 0.00] |
| C(imoea gray,imoea binary) | 0.03 [0.00, 0.04] |
| C(imoea binary,imoea gray) | 0.53 [0.45, 0.62] |

### zdt4/binary (30 runs)

Paper (Figs. 6-11): only IMOEA near the front: f2 about 50 for f1 in 0-0.5; the others f2 >= 140.

| method | HV | IGD | GD | merged: points, f1 range, f2 min / median / max, GD | calls | s/run |
|---|---|---|---|---|---|---|
| imoea | 1236 [1224, 1244] | 75.6 [67.9, 86.5] | 74.9 [68.6, 86.1] | 1, 0-0, 61.1 / 61.1 / 61.1, 60.1 | 24987 | 2.9 |
| nsga2 | 1192 [1183, 1202] | 110 [104, 121] | 117 [107, 124] | 35, 0-0.969, 75.6 / 78.3 / 84.7, 77.9 | 25000 | 4.4 |

| cover | value |
|---|---|
| C(imoea,nsga2) | 1.00 [1.00, 1.00] |
| C(nsga2,imoea) | 0.00 [0.00, 0.00] |

### zdt4/gray (30 runs)

Paper (Figs. 6-11): only IMOEA near the front: f2 about 50 for f1 in 0-0.5; the others f2 >= 140.

| method | HV | IGD | GD | merged: points, f1 range, f2 min / median / max, GD | calls | s/run |
|---|---|---|---|---|---|---|
| imoea | 1277 [1273, 1281] | 37.7 [34.7, 40.6] | 39.1 [36.1, 42.8] | 6, 0-0.5, 24.5 / 28.3 / 33, 27.7 | 24982 | 3.1 |
| nsga2 | 1277 [1274, 1279] | 36.4 [34.6, 39.2] | 38.9 [37, 41.6] | 63, 0-0.984, 28 / 30.4 / 33.9, 29.6 | 25000 | 4.7 |

| cover | value |
|---|---|
| C(imoea,nsga2) | 0.83 [0.00, 0.98] |
| C(nsga2,imoea) | 0.00 [0.00, 1.00] |
| C(imoea gray,imoea binary) | 0.58 [0.00, 1.00] |
| C(imoea binary,imoea gray) | 0.00 [0.00, 0.00] |

### zdt5/bits (30 runs)

Paper (Figs. 6-11): IMOEA best; plotted reference curve looks like the m = 11 front (10/f1), not 62/f1.

| method | HV | IGD | GD | merged: points, f1 range, f2 min / median / max, GD | calls | s/run |
|---|---|---|---|---|---|---|
| imoea | 7601 [7098, 7839] | 4.6 [4.12, 4.92] | 2.63 [2.54, 2.81] | 24, 3-26, 3.96 / 6.77 / 32.7, 2.25 | 24997 | 3.6 |
| nsga2 | 8216 [8208, 8225] | 3.01 [2.92, 3.07] | 4.89 [4.63, 5.03] | 30, 1-30, 3.63 / 6.97 / 108, 4.32 | 25000 | 5.2 |

| cover | value |
|---|---|
| C(imoea,nsga2) | 0.75 [0.68, 0.82] |
| C(nsga2,imoea) | 0.00 [0.00, 0.00] |

### zdt6/binary (30 runs)

Paper (Figs. 6-11): IMOEA f2 about 2.1-2.6 for f1 in 0.28-1; NSGA2 about 4-4.3; the others >= 4.

| method | HV | IGD | GD | merged: points, f1 range, f2 min / median / max, GD | calls | s/run |
|---|---|---|---|---|---|---|
| imoea | 5.05 [4.996, 5.111] | 2.12 [2.08, 2.21] | 1.89 [1.84, 1.95] | 7, 0.281-0.956, 2.45 / 2.72 / 2.91, 1.8 | 24985 | 2.8 |
| nsga2 | 5.238 [5.15, 5.28] | 1.93 [1.89, 2.05] | 1.74 [1.7, 1.85] | 39, 0.281-0.997, 2.22 / 2.43 / 2.71, 1.57 | 25000 | 4.2 |

| cover | value |
|---|---|
| C(imoea,nsga2) | 0.00 [0.00, 0.20] |
| C(nsga2,imoea) | 1.00 [0.50, 1.00] |

### zdt6/gray (30 runs)

Paper (Figs. 6-11): IMOEA f2 about 2.1-2.6 for f1 in 0.28-1; NSGA2 about 4-4.3; the others >= 4.

| method | HV | IGD | GD | merged: points, f1 range, f2 min / median / max, GD | calls | s/run |
|---|---|---|---|---|---|---|
| imoea | 4.365 [4.239, 4.44] | 2.98 [2.92, 3.1] | 2.76 [2.67, 2.85] | 9, 0.281-0.998, 3.27 / 3.45 / 3.56, 2.55 | 24986 | 2.9 |
| nsga2 | 3.832 [3.692, 3.963] | 3.68 [3.51, 3.85] | 3.43 [3.27, 3.6] | 40, 0.281-0.998, 3.82 / 3.9 / 4.18, 3.05 | 25000 | 4.4 |

| cover | value |
|---|---|
| C(imoea,nsga2) | 1.00 [0.94, 1.00] |
| C(nsga2,imoea) | 0.00 [0.00, 0.00] |
| C(imoea gray,imoea binary) | 0.00 [0.00, 0.00] |
| C(imoea binary,imoea gray) | 0.50 [0.50, 0.67] |

_pyiea 0.1.0_
