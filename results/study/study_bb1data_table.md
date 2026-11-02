# Study table: `study_bb1data`

Each cell is the mean ± the sample standard deviation over the seeds, with the value of the rank ensemble in brackets (`prod` event score). *runs* counts the seed runs found; *flagged* counts runs whose best validation loss is above 1.5 × the median of their configuration (they stay in every mean). A dash means that no file with that number was found. The R&D 2-prong signal is tuned and never evidence.

## AUC in the signal window

| configuration | runs | flagged | R&D 2-prong (tuned), SR | R&D 3-prong, SR | BB1, window | BB3, window |
|---|--:|--:|---|---|---|---|
| edgeconv | 3 | 0 | 0.843 ± 0.004 | 0.799 ± 0.004 | 0.908 ± 0.002 | – |
| efn | 3 | 0 | 0.857 ± 0.003 | 0.819 ± 0.001 | 0.904 ± 0.001 | – |
| ewmp | 3 | 0 | 0.853 ± 0.004 | 0.820 ± 0.005 | 0.906 ± 0.006 | – |

## max SIC in the signal window

| configuration | runs | flagged | R&D 2-prong (tuned), SR | R&D 3-prong, SR | BB1, window | BB3, window |
|---|--:|--:|---|---|---|---|
| edgeconv | 3 | 0 | 1.66 ± 0.02 | 1.44 ± 0.01 | 2.66 ± 0.12 | – |
| efn | 3 | 0 | 1.74 ± 0.02 | 1.52 ± 0.00 | 2.55 ± 0.08 | – |
| ewmp | 3 | 0 | 1.72 ± 0.04 | 1.52 ± 0.01 | 2.66 ± 0.16 | – |

## 1/ε_B at ε_S = 0.3 in the signal window

| configuration | runs | flagged | R&D 2-prong (tuned), SR | R&D 3-prong, SR | BB1, window | BB3, window |
|---|--:|--:|---|---|---|---|
| edgeconv | 3 | 0 | 17.6 ± 1.2 | 9.8 ± 0.4 | 68.0 ± 1.5 | – |
| efn | 3 | 0 | 19.3 ± 0.4 | 11.0 ± 0.2 | 55.6 ± 1.3 | – |
| ewmp | 3 | 0 | 19.0 ± 0.6 | 11.2 ± 0.7 | 57.9 ± 10.3 | – |

## AUC over the full m_jj range

| configuration | runs | flagged | R&D 2-prong (tuned), full | R&D 3-prong, full | BB1, full | BB3, full |
|---|--:|--:|---|---|---|---|
| edgeconv | 3 | 0 | 0.837 ± 0.003 | 0.803 ± 0.004 | 0.849 ± 0.001 | – |
| efn | 3 | 0 | 0.849 ± 0.003 | 0.821 ± 0.001 | 0.841 ± 0.001 | – |
| ewmp | 3 | 0 | 0.845 ± 0.004 | 0.822 ± 0.004 | 0.843 ± 0.006 | – |

## max SIC over the full m_jj range

| configuration | runs | flagged | R&D 2-prong (tuned), full | R&D 3-prong, full | BB1, full | BB3, full |
|---|--:|--:|---|---|---|---|
| edgeconv | 3 | 0 | 1.65 ± 0.02 | 1.44 ± 0.01 | 2.28 ± 0.08 | – |
| efn | 3 | 0 | 1.72 ± 0.02 | 1.52 ± 0.01 | 2.14 ± 0.07 | – |
| ewmp | 3 | 0 | 1.69 ± 0.04 | 1.52 ± 0.01 | 2.20 ± 0.11 | – |

## 1/ε_B at ε_S = 0.3 over the full m_jj range

| configuration | runs | flagged | R&D 2-prong (tuned), full | R&D 3-prong, full | BB1, full | BB3, full |
|---|--:|--:|---|---|---|---|
| edgeconv | 3 | 0 | 18.5 ± 1.4 | 10.6 ± 0.5 | 54.3 ± 2.6 | – |
| efn | 3 | 0 | 19.7 ± 0.5 | 11.6 ± 0.1 | 44.5 ± 1.8 | – |
| ewmp | 3 | 0 | 19.5 ± 0.7 | 11.8 ± 0.8 | 47.2 ± 5.8 | – |

## Sculpting and training loss

| configuration | runs | flagged | JSD at ε_B = 1 % (R&D bkg_test) | background validation loss |
|---|--:|--:|---|---|
| edgeconv | 3 | 0 | 0.0120 ± 0.0019 | 0.2599 ± 0.01085 |
| efn | 3 | 0 | 0.0101 ± 0.0008 | 0.2475 ± 0.004819 |
| ewmp | 3 | 0 | 0.0099 ± 0.0008 | 0.2428 ± 0.02062 |

## Latent probes (mlp, R², 5-fold)

| configuration | runs | flagged | m_j | pt_j | tau21 | tau32 | n_const | efp_wedge | efp_triangle | efp_square |
|---|--:|--:|---|---|---|---|---|---|---|---|
| edgeconv | 3 | 0 | – | – | – | – | – | – | – | – |
| efn | 3 | 0 | – | – | – | – | – | – | – | – |
| ewmp | 3 | 0 | – | – | – | – | – | – | – | – |

## Latent probes (ridge, R², 5-fold)

| configuration | runs | flagged | m_j | pt_j | tau21 | tau32 | n_const | efp_wedge | efp_triangle | efp_square |
|---|--:|--:|---|---|---|---|---|---|---|---|
| edgeconv | 3 | 0 | – | – | – | – | – | – | – | – |
| efn | 3 | 0 | – | – | – | – | – | – | – | – |
| ewmp | 3 | 0 | – | – | – | – | – | – | – | – |

## IRC probe

| configuration | runs | flagged | ρ: split top-1 | Δ AUC: split top-1 | ρ: split top-5 | Δ AUC: split top-5 | ρ: split all | Δ AUC: split all | ρ: 10 soft (0.2% pT each) | Δ AUC: 10 soft (0.2% pT each) | ρ: 10 soft (1% pT each) | Δ AUC: 10 soft (1% pT each) | ρ: random re-order | Δ AUC: random re-order | k-NN tie fraction |
|---|--:|--:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edgeconv | 3 | 0 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| efn | 3 | 0 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ewmp | 3 | 0 | – | – | – | – | – | – | – | – | – | – | – | – | – |

## Flagged runs

None.
