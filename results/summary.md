# Results summary (test splits)

2p = 2-prong W'->XY(qq) signal, 3p = 3-prong signal (never used for training or tuning), SR = 3.3 < m_jj < 3.7 TeV. Event score from the two jets' scores: max / sum / min, or prod = log F_B(s_1) + log F_B(s_2) with F_B the background (validation) CDF of jet scores. maxSIC requires >= 50 background test events after the cut.

## AUC in the SR

| run | 2p max | 2p sum | 2p min | 2p prod | 3p max | 3p sum | 3p min | 3p prod |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_-tau21 | 0.857 | 0.893 | 0.852 | – | 0.593 | 0.658 | 0.682 | – |
| baseline_-tau32 | 0.673 | 0.688 | 0.652 | – | 0.838 | 0.854 | 0.772 | – |
| baseline_m_j | 0.811 | 0.763 | 0.549 | – | 0.817 | 0.767 | 0.550 | – |
| baseline_n_const | 0.390 | 0.379 | 0.424 | – | 0.585 | 0.596 | 0.595 | – |
| ef_ae | 0.701 | 0.731 | 0.764 | 0.780 | 0.850 | 0.861 | 0.782 | 0.828 |
| ef_ae+qn | 0.678 | 0.653 | 0.585 | 0.618 | 0.787 | 0.822 | 0.795 | 0.816 |
| set_ae | 0.488 | 0.450 | 0.387 | – | 0.532 | 0.471 | 0.353 | – |
| set_ae+flow | 0.569 | 0.421 | 0.339 | – | 0.583 | 0.415 | 0.320 | – |
| set_ae+qn | 0.380 | 0.346 | 0.363 | – | 0.417 | 0.364 | 0.353 | – |

## max SIC in the SR

| run | 2p max | 2p sum | 2p min | 2p prod | 3p max | 3p sum | 3p min | 3p prod |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_-tau21 | 1.78 | 2.25 | 1.85 | – | 1.08 | 1.12 | 1.13 | – |
| baseline_-tau32 | 1.04 | 1.06 | 1.01 | – | 1.94 | 2.11 | 1.42 | – |
| baseline_m_j | 1.91 | 1.67 | 1.22 | – | 1.94 | 1.68 | 1.24 | – |
| baseline_n_const | 1.01 | 1.01 | 1.01 | – | 1.08 | 1.12 | 1.13 | – |
| ef_ae | 1.19 | 1.25 | 1.33 | 1.39 | 1.66 | 1.71 | 1.34 | 1.52 |
| ef_ae+qn | 1.08 | 1.07 | 1.00 | 1.02 | 1.35 | 1.56 | 1.46 | 1.54 |
| set_ae | 1.00 | 1.00 | 1.00 | – | 1.00 | 1.00 | 1.00 | – |
| set_ae+flow | 1.00 | 1.00 | 1.00 | – | 1.00 | 1.00 | 1.00 | – |
| set_ae+qn | 1.00 | 1.00 | 1.00 | – | 1.00 | 1.00 | 1.00 | – |

## Background rejection 1/εB at εS = 0.3 in the SR

| run | 2p max | 2p sum | 2p min | 2p prod | 3p max | 3p sum | 3p min | 3p prod |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_-tau21 | 31.7 | 49.1 | 33.7 | – | 3.7 | 5.1 | 5.9 | – |
| baseline_-tau32 | 8.4 | 10.0 | 8.2 | – | 40.6 | 48.7 | 21.4 | – |
| baseline_m_j | 7.1 | 5.2 | 2.4 | – | 6.9 | 5.0 | 2.3 | – |
| baseline_n_const | 2.0 | 1.9 | 2.1 | – | 3.5 | 3.4 | 3.3 | – |
| ef_ae | 6.0 | 6.8 | 8.2 | 8.5 | 25.3 | 24.6 | 10.1 | 13.0 |
| ef_ae+qn | 7.3 | 5.8 | 5.2 | 5.6 | 15.8 | 25.5 | 22.7 | 25.2 |
| set_ae | 3.1 | 2.6 | 2.1 | – | 3.6 | 2.8 | 1.9 | – |
| set_ae+flow | 4.6 | 2.6 | 2.0 | – | 4.6 | 2.4 | 1.8 | – |
| set_ae+qn | 2.2 | 1.9 | 1.8 | – | 2.5 | 2.0 | 1.8 | – |

## Sculpting and correlations (background test events)

| run | JSD(m_jj) @εB=10% max | @1% max | @10% prod | @1% prod | dCor(jet score, m_j) | dCor(jet score, n) |
|---|---:|---:|---:|---:|---:|---:|
| baseline_-tau21 | 0.005 | 0.015 | – | – | 0.03 | 0.03 |
| baseline_-tau32 | 0.003 | 0.012 | – | – | 0.03 | 0.03 |
| baseline_m_j | 0.043 | 0.251 | – | – | 1.00 | 0.72 |
| baseline_n_const | 0.007 | 0.036 | – | – | 0.72 | 1.00 |
| ef_ae | 0.002 | 0.010 | 0.001 | 0.008 | 0.72 | 0.64 |
| ef_ae+qn | 0.016 | 0.026 | 0.020 | 0.043 | 0.02 | 0.18 |
| set_ae | 0.001 | 0.004 | – | – | 0.25 | 0.15 |
| set_ae+flow | 0.002 | 0.011 | – | – | 0.48 | 0.22 |
| set_ae+qn | 0.002 | 0.006 | – | – | 0.02 | 0.34 |

## Headline (max aggregation)

| run | AUC 2p SR | maxSIC 2p SR | 1/εB@εS=0.3 2p SR | AUC 3p SR | maxSIC 3p SR | AUC 2p full | JSD mjj @εB=1% | dCor(score, m_j) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_-tau21 | 0.857 | 1.78 | 31.7 | 0.593 | 1.08 | 0.855 | 0.015 | 0.03 |
| baseline_-tau32 | 0.673 | 1.04 | 8.4 | 0.838 | 1.94 | 0.689 | 0.012 | 0.03 |
| baseline_m_j | 0.811 | 1.91 | 7.1 | 0.817 | 1.94 | 0.833 | 0.251 | 1.00 |
| baseline_n_const | 0.390 | 1.01 | 2.0 | 0.585 | 1.08 | 0.399 | 0.036 | 0.72 |
| ef_ae | 0.701 | 1.19 | 6.0 | 0.850 | 1.66 | 0.698 | 0.010 | 0.72 |
| ef_ae+qn | 0.678 | 1.08 | 7.3 | 0.787 | 1.35 | 0.634 | 0.026 | 0.02 |
| set_ae | 0.488 | 1.00 | 3.1 | 0.532 | 1.00 | 0.499 | 0.004 | 0.25 |
| set_ae+flow | 0.569 | 1.00 | 4.6 | 0.583 | 1.00 | 0.579 | 0.011 | 0.48 |
| set_ae+qn | 0.380 | 1.00 | 2.2 | 0.417 | 1.00 | 0.382 | 0.006 | 0.02 |
