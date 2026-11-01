## E3 summary

### BB1 bump hunt (BB1-trained runs; window 3.6-4.0 TeV; local Z)

| encoder | seed | Z no cut | Z 10 % cut | Z 2 % cut | true signal kept (10 % / 2 %) of 512 |
|---|--:|--:|--:|--:|---|
| edgeconv | 1 | +0.24 | +2.41 | +3.01 | 391 / 193 |
| edgeconv | 2 | +0.24 | +2.40 | +4.22 | 381 / 187 |
| edgeconv | 3 | +0.24 | +2.25 | +2.95 | 390 / 188 |
| ewmp | 1 | +0.24 | +1.78 | +3.42 | 360 / 155 |
| ewmp | 2 | +0.24 | +3.00 | +4.18 | 390 / 187 |
| ewmp | 3 | +0.24 | +2.85 | +3.73 | 385 / 169 |
| efn | 1 | +0.24 | +2.26 | +4.35 | 376 / 179 |
| efn | 2 | +0.24 | +2.24 | +3.73 | 373 / 168 |
| efn | 3 | +0.24 | +2.47 | +1.55 | 374 / 162 |

Mean +- standard deviation of Z over seeds:

| encoder | seeds | no cut | 10 % cut | 2 % cut |
|---|--:|--:|--:|--:|
| edgeconv | 3 | +0.24 +- 0.00 | +2.36 +- 0.09 | +3.39 +- 0.71 |
| ewmp | 3 | +0.24 +- 0.00 | +2.54 +- 0.66 | +3.78 +- 0.38 |
| efn | 3 | +0.24 +- 0.00 | +2.33 +- 0.13 | +3.21 +- 1.47 |

### BB1 metrics on all events and on the validation part alone (event `prod` AUC, mean +- sd over seeds)

| encoder | seeds | window, all | window, validation part | full, all | full, validation part |
|---|--:|--:|--:|--:|--:|
| edgeconv | 3 | 0.9080 +- 0.0020 | 0.8944 +- 0.0022 | 0.8488 +- 0.0014 | 0.8429 +- 0.0022 |
| ewmp | 3 | 0.9058 +- 0.0061 | 0.8987 +- 0.0035 | 0.8429 +- 0.0060 | 0.8391 +- 0.0042 |
| efn | 3 | 0.9039 +- 0.0008 | 0.8970 +- 0.0006 | 0.8413 +- 0.0013 | 0.8356 +- 0.0010 |

### BB2 null test of the R&D-trained 100-constituent runs (BB2 subsample: 200 000 events; local Z)

| encoder | seed | KS (`prod`) | Z 3300-3700 GeV: no cut / 10 % / 2 % | Z 3600-4000 GeV: no cut / 10 % / 2 % | Z 4000-4400 GeV: no cut / 10 % / 2 % |
|---|--:|--:|---|---|---|
| edgeconv | 1 | 0.044 | -1.67 / -1.20 / -0.08 | +0.37 / +0.72 / +0.45 | -0.04 / +0.90 / +0.32 |
| edgeconv | 2 | 0.041 | -1.67 / -1.83 / +0.53 | +0.37 / +0.35 / -0.45 | -0.04 / +0.15 / +1.15 |
| edgeconv | 3 | 0.043 | -1.67 / -1.96 / -0.15 | +0.37 / +0.28 / +0.69 | -0.04 / +0.25 / -0.13 |
| ewmp | 1 | 0.032 | -1.67 / -1.68 / -0.91 | +0.37 / +0.91 / -0.08 | -0.04 / +0.01 / +1.25 |
| ewmp | 2 | 0.032 | -1.67 / -2.57 / -0.70 | +0.37 / -0.58 / +1.23 | -0.04 / +1.91 / +0.07 |
| ewmp | 3 | 0.033 | -1.67 / -2.48 / -0.21 | +0.37 / +0.58 / -0.71 | -0.04 / +1.54 / +1.76 |
| transformer | 1 | 0.030 | -1.67 / -2.49 / -0.06 | +0.37 / +0.10 / +1.04 | -0.04 / +1.32 / +0.95 |
| transformer | 2 | 0.031 | -1.67 / -2.36 / -1.08 | +0.37 / +0.25 / +0.38 | -0.04 / +0.40 / +1.53 |

Largest |Z| over all runs, windows and cuts: 2.57

### Training wall time and flags

| group | encoder | seeds | training minutes (sum) | best_val_loss per seed | flagged |
|---|---|--:|--:|---|---|
| bb1data | edgeconv | 3 | 26 | 0.2692, 0.2480, 0.2624 | none |
| bb1data | ewmp | 3 | 74 | 0.2366, 0.2260, 0.2658 | none |
| bb1data | efn | 3 | 6 | 0.2452, 0.2442, 0.2530 | none |
| n100 | edgeconv | 3 | 55 | 0.2370, 0.2278, 0.2280 | none |
| n100 | ewmp | 3 | 275 | 0.2127, 0.2067, 0.2133 | none |
| n100 | transformer | 3 | 67 | 0.1710, 0.1680, 0.1620 | none |
