# Development runs (validation splits: bkg_val vs 2-prong sig_val; never the test splits)

AUC / max SIC in the signal region (3.3-3.7 TeV) for event scores max/sum/min over the two jets, jet-level AUC, heavier/lighter jet AUC, and Spearman correlations of the per-jet score with m_j, multiplicity and τ21 on background; last column = training minutes (2 CPU cores).

| dev run | SR max AUC/maxSIC | SR sum AUC/maxSIC | SR min AUC | SR jet AUC | heavy | light | ρ(m) | ρ(n) | ρ(τ21) | min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_-tau21 | 0.856 / 1.82 | 0.893 / 2.27 | 0.854 | 0.800 | 0.845 | 0.790 | +0.44 | +0.14 | -1.00 | 0 |
| baseline_-tau32 | 0.675 / 1.05 | 0.690 / 1.07 | 0.655 | 0.637 | 0.602 | 0.673 | +0.24 | +0.06 | -0.08 | 0 |
| baseline_m_j | 0.812 / 1.91 | 0.760 / 1.67 | 0.544 | 0.617 | 0.812 | 0.544 | +1.00 | +0.78 | -0.44 | 0 |
| baseline_n_const | 0.391 / 1.01 | 0.381 / 1.01 | 0.423 | 0.424 | 0.424 | 0.409 | +0.78 | +1.00 | -0.14 | 0 |
| deep_svdd_d1 | 0.629 / 1.06 | 0.635 / 1.06 | 0.605 | 0.597 | 0.652 | 0.546 | +0.43 | +0.31 | -0.31 | 13 |
| ef_ae_d1 | 0.740 / 1.26 | 0.774 / 1.37 | 0.786 | 0.713 | 0.745 | 0.747 | +0.76 | +0.64 | -0.52 | 17 |
| ef_ae_d2_s05 | 0.698 / 1.24 | 0.720 / 1.29 | 0.762 | 0.685 | 0.703 | 0.738 | +0.84 | +0.74 | -0.48 | 16 |
| node_gae_d1 | 0.468 / 1.00 | 0.423 / 1.00 | 0.354 | 0.437 | 0.581 | 0.293 | +0.17 | -0.16 | -0.09 | 17 |
| set_ae@full | 0.488 / 1.00 | 0.451 / 1.00 | 0.389 | 0.453 | 0.579 | 0.324 | +0.27 | -0.14 | -0.14 | 97 |
