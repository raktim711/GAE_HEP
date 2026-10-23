# Results summary (event score = max over the two leading jets)

2p = 2-prong W'->XY(qq) signal, 3p = 3-prong signal (never used for training or tuning), SR = 3.3 < m_jj < 3.7 TeV. maxSIC requires >= 50 background test events after the cut.

| run | AUC 2p SR | maxSIC 2p SR | 1/εB@εS=0.3 2p SR | AUC 3p SR | maxSIC 3p SR | AUC 2p full | JSD mjj @εB=1% | dCor(score, m_j) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_-tau21 | 0.857 | 1.78 | 31.7 | 0.593 | 1.08 | 0.855 | 0.015 | 0.03 |
| baseline_-tau32 | 0.673 | 1.04 | 8.4 | 0.838 | 1.94 | 0.689 | 0.012 | 0.03 |
| baseline_m_j | 0.811 | 1.91 | 7.1 | 0.817 | 1.94 | 0.833 | 0.251 | 1.00 |
| baseline_n_const | 0.390 | 1.01 | 2.0 | 0.585 | 1.08 | 0.399 | 0.036 | 0.72 |
| set_ae | 0.488 | 1.00 | 3.1 | 0.532 | 1.00 | 0.499 | 0.004 | 0.25 |
| set_ae+flow | 0.569 | 1.00 | 4.6 | 0.583 | 1.00 | 0.579 | 0.011 | 0.48 |
| set_ae+qn | 0.380 | 1.00 | 2.2 | 0.417 | 1.00 | 0.382 | 0.006 | 0.02 |
