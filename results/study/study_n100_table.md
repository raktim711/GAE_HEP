# Study table: `study_n100`

Each cell is the mean ± the sample standard deviation over the seeds, with the value of the rank ensemble in brackets (`prod` event score). *runs* counts the seed runs found; *flagged* counts runs whose best validation loss is above 1.5 × the median of their configuration (they stay in every mean). A dash means that no file with that number was found. The R&D 2-prong signal is tuned and never evidence.

## AUC in the signal window

| configuration | runs | flagged | R&D 2-prong (tuned), SR | R&D 3-prong, SR | BB1, window | BB3, window |
|---|--:|--:|---|---|---|---|
| edgeconv | 3 | 0 | 0.855 ± 0.004 | 0.801 ± 0.002 | 0.912 ± 0.002 | 0.474 ± 0.001 |
| ewmp | 3 | 0 | 0.863 ± 0.005 | 0.820 ± 0.008 | 0.910 ± 0.002 | 0.474 ± 0.000 |
| transformer | 3 | 0 | 0.849 ± 0.001 | 0.817 ± 0.002 | 0.902 ± 0.002 | 0.472 ± 0.001 |

## max SIC in the signal window

| configuration | runs | flagged | R&D 2-prong (tuned), SR | R&D 3-prong, SR | BB1, window | BB3, window |
|---|--:|--:|---|---|---|---|
| edgeconv | 3 | 0 | 1.75 ± 0.04 | 1.44 ± 0.01 | 3.09 ± 0.01 | 1.00 ± 0.00 |
| ewmp | 3 | 0 | 1.80 ± 0.05 | 1.52 ± 0.02 | 2.87 ± 0.20 | 1.00 ± 0.00 |
| transformer | 3 | 0 | 1.68 ± 0.00 | 1.48 ± 0.01 | 2.67 ± 0.13 | 1.00 ± 0.00 |

## 1/ε_B at ε_S = 0.3 in the signal window

| configuration | runs | flagged | R&D 2-prong (tuned), SR | R&D 3-prong, SR | BB1, window | BB3, window |
|---|--:|--:|---|---|---|---|
| edgeconv | 3 | 0 | 22.0 ± 0.7 | 10.5 ± 0.2 | 99.2 ± 4.1 | 3.2 ± 0.1 |
| ewmp | 3 | 0 | 22.0 ± 3.3 | 11.3 ± 1.0 | 80.4 ± 15.3 | 3.2 ± 0.1 |
| transformer | 3 | 0 | 21.1 ± 0.9 | 12.8 ± 0.3 | 70.8 ± 0.7 | 3.3 ± 0.1 |

## AUC over the full m_jj range

| configuration | runs | flagged | R&D 2-prong (tuned), full | R&D 3-prong, full | BB1, full | BB3, full |
|---|--:|--:|---|---|---|---|
| edgeconv | 3 | 0 | 0.849 ± 0.004 | 0.805 ± 0.001 | 0.852 ± 0.001 | 0.575 ± 0.001 |
| ewmp | 3 | 0 | 0.855 ± 0.005 | 0.822 ± 0.007 | 0.848 ± 0.002 | 0.573 ± 0.001 |
| transformer | 3 | 0 | 0.841 ± 0.001 | 0.819 ± 0.002 | 0.841 ± 0.001 | 0.579 ± 0.001 |

## max SIC over the full m_jj range

| configuration | runs | flagged | R&D 2-prong (tuned), full | R&D 3-prong, full | BB1, full | BB3, full |
|---|--:|--:|---|---|---|---|
| edgeconv | 3 | 0 | 1.75 ± 0.04 | 1.45 ± 0.01 | 2.76 ± 0.10 | 1.00 ± 0.00 |
| ewmp | 3 | 0 | 1.78 ± 0.05 | 1.52 ± 0.02 | 2.51 ± 0.26 | 1.00 ± 0.00 |
| transformer | 3 | 0 | 1.66 ± 0.01 | 1.48 ± 0.01 | 2.23 ± 0.06 | 1.00 ± 0.00 |

## 1/ε_B at ε_S = 0.3 over the full m_jj range

| configuration | runs | flagged | R&D 2-prong (tuned), full | R&D 3-prong, full | BB1, full | BB3, full |
|---|--:|--:|---|---|---|---|
| edgeconv | 3 | 0 | 23.5 ± 0.7 | 11.4 ± 0.1 | 72.7 ± 1.1 | 4.6 ± 0.0 |
| ewmp | 3 | 0 | 22.9 ± 3.2 | 12.1 ± 1.1 | 64.0 ± 7.8 | 4.6 ± 0.1 |
| transformer | 3 | 0 | 21.7 ± 0.7 | 13.7 ± 0.3 | 53.0 ± 4.8 | 4.7 ± 0.1 |

## Sculpting and training loss

| configuration | runs | flagged | JSD at ε_B = 1 % (R&D bkg_test) | background validation loss |
|---|--:|--:|---|---|
| edgeconv | 3 | 0 | 0.0126 ± 0.0004 | 0.231 ± 0.005257 |
| ewmp | 3 | 0 | 0.0097 ± 0.0006 | 0.2109 ± 0.00362 |
| transformer | 3 | 0 | 0.0084 ± 0.0018 | 0.167 ± 0.004579 |

## Latent probes (mlp, R², 5-fold)

| configuration | runs | flagged | m_j | pt_j | tau21 | tau32 | n_const | efp_wedge | efp_triangle | efp_square |
|---|--:|--:|---|---|---|---|---|---|---|---|
| edgeconv | 3 | 0 | 0.815 ± 0.018 | 0.025 ± 0.034 | 0.547 ± 0.036 | 0.088 ± 0.024 | 0.689 ± 0.031 | 0.859 ± 0.012 | 0.669 ± 0.023 | 0.760 ± 0.018 |
| ewmp | 3 | 0 | 0.851 ± 0.024 | -0.006 ± 0.003 | 0.606 ± 0.032 | 0.100 ± 0.007 | 0.663 ± 0.027 | 0.921 ± 0.041 | 0.764 ± 0.087 | 0.862 ± 0.059 |
| transformer | 3 | 0 | 0.817 ± 0.021 | -0.007 ± 0.004 | 0.620 ± 0.051 | 0.113 ± 0.013 | 0.679 ± 0.038 | 0.886 ± 0.023 | 0.726 ± 0.032 | 0.806 ± 0.025 |

## Latent probes (ridge, R², 5-fold)

| configuration | runs | flagged | m_j | pt_j | tau21 | tau32 | n_const | efp_wedge | efp_triangle | efp_square |
|---|--:|--:|---|---|---|---|---|---|---|---|
| edgeconv | 3 | 0 | 0.764 ± 0.024 | 0.020 ± 0.021 | 0.377 ± 0.018 | 0.058 ± 0.001 | 0.537 ± 0.056 | 0.796 ± 0.024 | 0.567 ± 0.036 | 0.568 ± 0.034 |
| ewmp | 3 | 0 | 0.783 ± 0.050 | 0.006 ± 0.001 | 0.365 ± 0.003 | 0.065 ± 0.008 | 0.492 ± 0.020 | 0.862 ± 0.053 | 0.642 ± 0.051 | 0.671 ± 0.032 |
| transformer | 3 | 0 | 0.755 ± 0.022 | 0.008 ± 0.004 | 0.368 ± 0.007 | 0.066 ± 0.002 | 0.567 ± 0.057 | 0.764 ± 0.012 | 0.545 ± 0.012 | 0.530 ± 0.017 |

## IRC probe

| configuration | runs | flagged | ρ: split top-1 | Δ AUC: split top-1 | ρ: split top-5 | Δ AUC: split top-5 | ρ: split all | Δ AUC: split all | ρ: 10 soft (0.2% pT each) | Δ AUC: 10 soft (0.2% pT each) | ρ: 10 soft (1% pT each) | Δ AUC: 10 soft (1% pT each) | ρ: random re-order | Δ AUC: random re-order | k-NN tie fraction |
|---|--:|--:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edgeconv | 3 | 0 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ewmp | 3 | 0 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| transformer | 3 | 0 | – | – | – | – | – | – | – | – | – | – | – | – | – |

## Flagged runs

None.
