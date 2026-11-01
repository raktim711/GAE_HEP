# E2 ablations (validation splits only)

Descriptive only (protocol H5: no verdict). All runs are 2 epochs of 120 000 jets, seeds 1-3, evaluated with the
development protocol: background `bkg_val` against the 2-prong `sig_val`, `prod` event score, SR. The 2-prong
signal is the tuned one, so these numbers are never evidence for a hypothesis. mean ± std over seeds (sample
standard deviation); `flagged` = runs whose `best_val_loss` exceeds 1.5 × the median of their configuration
(kept in every mean). `default` rows are E1's `std_<enc>_s<seed>` runs, seeds 1-3 (E1 ran seeds 1-5).

| encoder | ablation | setting | seeds | best val loss | AUC (SR) | max SIC (SR) | 1/ε_B @ ε_S=0.3 | AUC (full m_jj) | flagged |
|---|---|---|---|---|---|---|---|---|---|
| deepsets | default | latent 4, mean+max (EWMP: weighted), n=50 | 3 | 0.3321 ± 0.0742 | 0.840 ± 0.023 | 1.68 ± 0.09 | 14.7 ± 4.7 | 0.837 ± 0.023 | 0 |
| deepsets | latent width | d = 2 | 3 | 0.4131 ± 0.0085 | 0.813 ± 0.001 | 1.58 ± 0.01 | 9.2 ± 0.1 | 0.811 ± 0.001 | 0 |
| deepsets | latent width | d = 8 | 3 | 0.3512 ± 0.0704 | 0.835 ± 0.020 | 1.66 ± 0.07 | 13.4 ± 4.0 | 0.832 ± 0.020 | 0 |
| deepsets | latent width | d = 16 | 3 | 0.4495 ± 0.1378 | 0.803 ± 0.046 | 1.58 ± 0.15 | 10.8 ± 7.1 | 0.801 ± 0.043 | 0 |
| deepsets | latent width | d = 32 | 3 | 0.5226 ± 0.0060 | 0.784 ± 0.006 | 1.50 ± 0.01 | 7.0 ± 0.3 | 0.783 ± 0.006 | 0 |
| deepsets | pooling | pma | 3 | 0.5142 ± 0.0010 | 0.778 ± 0.000 | 1.51 ± 0.00 | 6.7 ± 0.0 | 0.778 ± 0.000 | 0 |
| deepsets | pooling | weighted | 3 | 0.3794 ± 0.1306 | 0.817 ± 0.047 | 1.62 ± 0.21 | 12.6 ± 8.1 | 0.815 ± 0.044 | 0 |
| deepsets | constituents | n = 30 | 3 | 0.4472 ± 0.1151 | 0.803 ± 0.036 | 1.56 ± 0.08 | 9.4 ± 4.2 | 0.800 ± 0.034 | 0 |
| edgeconv | default | latent 4, mean+max (EWMP: weighted), n=50 | 3 | 0.2418 ± 0.0065 | 0.849 ± 0.003 | 1.70 ± 0.02 | 18.5 ± 0.5 | 0.846 ± 0.004 | 0 |
| edgeconv | latent width | d = 2 | 3 | 0.3972 ± 0.0109 | 0.812 ± 0.002 | 1.56 ± 0.01 | 9.2 ± 0.2 | 0.810 ± 0.001 | 0 |
| edgeconv | latent width | d = 8 | 3 | 0.2381 ± 0.0105 | 0.851 ± 0.004 | 1.72 ± 0.04 | 19.2 ± 0.6 | 0.848 ± 0.004 | 0 |
| edgeconv | latent width | d = 16 | 3 | 0.2461 ± 0.0120 | 0.847 ± 0.001 | 1.69 ± 0.01 | 18.6 ± 0.6 | 0.844 ± 0.001 | 0 |
| edgeconv | latent width | d = 32 | 3 | 0.3170 ± 0.0846 | 0.834 ± 0.023 | 1.65 ± 0.08 | 14.7 ± 5.0 | 0.832 ± 0.022 | stw_edgeconv_l32_s3 |
| edgeconv | pooling | pma | 3 | 0.2902 ± 0.0815 | 0.842 ± 0.021 | 1.69 ± 0.09 | 16.1 ± 5.2 | 0.840 ± 0.021 | 0 |
| edgeconv | pooling | weighted | 3 | 0.2058 ± 0.0034 | 0.854 ± 0.003 | 1.74 ± 0.03 | 21.6 ± 1.4 | 0.850 ± 0.003 | 0 |
| edgeconv | constituents | n = 30 | 3 | 0.2501 ± 0.0075 | 0.839 ± 0.001 | 1.63 ± 0.01 | 16.6 ± 0.2 | 0.835 ± 0.001 | 0 |
| ewmp | default | latent 4, mean+max (EWMP: weighted), n=50 | 3 | 0.2266 ± 0.0132 | 0.857 ± 0.002 | 1.74 ± 0.04 | 18.8 ± 0.8 | 0.851 ± 0.002 | 0 |
| ewmp | latent width | d = 2 | 3 | 0.3599 ± 0.0016 | 0.818 ± 0.002 | 1.57 ± 0.01 | 10.4 ± 0.4 | 0.816 ± 0.003 | 0 |
| ewmp | latent width | d = 8 | 3 | 0.2219 ± 0.0127 | 0.858 ± 0.004 | 1.75 ± 0.03 | 20.2 ± 1.0 | 0.852 ± 0.003 | 0 |
| ewmp | latent width | d = 16 | 3 | 0.2204 ± 0.0121 | 0.855 ± 0.004 | 1.72 ± 0.03 | 18.8 ± 1.2 | 0.850 ± 0.003 | 0 |
| ewmp | latent width | d = 32 | 3 | 0.3204 ± 0.0686 | 0.831 ± 0.028 | 1.61 ± 0.14 | 14.3 ± 5.9 | 0.828 ± 0.026 | 0 |
| ewmp | constituents | n = 30 | 3 | 0.2818 ± 0.0989 | 0.842 ± 0.027 | 1.69 ± 0.11 | 15.4 ± 4.9 | 0.837 ± 0.024 | stn_ewmp_n30_s2 |
