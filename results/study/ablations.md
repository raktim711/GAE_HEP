# E2 ablations (validation splits only)

Descriptive only (protocol H5: no verdict). All runs are 2 epochs of 120 000 jets, seeds 1-3, evaluated with the
development protocol: background `bkg_val` against the 2-prong `sig_val`, `prod` event score, SR. The 2-prong
signal is the tuned one, so these numbers are never evidence for a hypothesis. mean ± std over seeds (sample
standard deviation); `flagged` = runs whose `best_val_loss` exceeds 1.5 × the median of their configuration
(kept in every mean). `default` rows are E1's `std_<enc>_s<seed>` runs (merged by E1; absent if E1 is not merged).

| encoder | ablation | setting | seeds | best val loss | AUC (SR) | max SIC (SR) | 1/ε_B @ ε_S=0.3 | AUC (full m_jj) | flagged |
|---|---|---|---|---|---|---|---|---|---|
| deepsets | latent width | d = 2 | 1 | 0.4050 | 0.812 | 1.58 | 9.1 | 0.810 | 0 |
| deepsets | latent width | d = 8 | 1 | 0.2866 | 0.850 | 1.72 | 17.0 | 0.847 | 0 |
| deepsets | latent width | d = 16 | 1 | 0.2904 | 0.856 | 1.75 | 19.0 | 0.851 | 0 |
| deepsets | latent width | d = 32 | 1 | 0.5181 | 0.790 | 1.52 | 7.3 | 0.789 | 0 |
| deepsets | pooling | pma | 1 | 0.5154 | 0.779 | 1.51 | 6.7 | 0.779 | 0 |
| deepsets | pooling | weighted | 1 | 0.2523 | 0.868 | 1.87 | 21.9 | 0.863 | 0 |
| deepsets | constituents | n = 30 | 1 | 0.3361 | 0.840 | 1.65 | 14.1 | 0.835 | 0 |
| edgeconv | latent width | d = 2 | 1 | 0.3977 | 0.813 | 1.57 | 9.2 | 0.811 | 0 |
| edgeconv | latent width | d = 8 | 1 | 0.2315 | 0.847 | 1.69 | 18.6 | 0.844 | 0 |
| edgeconv | latent width | d = 16 | 1 | 0.2405 | 0.847 | 1.68 | 19.1 | 0.844 | 0 |
| edgeconv | latent width | d = 32 | 1 | 0.2756 | 0.849 | 1.72 | 17.3 | 0.846 | 0 |
| edgeconv | constituents | n = 30 | 1 | 0.2511 | 0.839 | 1.62 | 16.4 | 0.835 | 0 |
| ewmp | latent width | d = 2 | 1 | 0.3584 | 0.819 | 1.59 | 10.4 | 0.816 | 0 |
| ewmp | latent width | d = 8 | 1 | 0.2315 | 0.858 | 1.76 | 20.0 | 0.853 | 0 |
| ewmp | latent width | d = 16 | 1 | 0.2134 | 0.853 | 1.70 | 19.3 | 0.848 | 0 |
| ewmp | latent width | d = 32 | 1 | 0.3188 | 0.824 | 1.54 | 12.5 | 0.822 | 0 |
