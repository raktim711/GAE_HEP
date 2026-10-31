# E2 ablations (validation splits only)

Descriptive only (protocol H5: no verdict). All runs are 2 epochs of 120 000 jets, seeds 1-3, evaluated with the
development protocol: background `bkg_val` against the 2-prong `sig_val`, `prod` event score, SR. The 2-prong
signal is the tuned one, so these numbers are never evidence for a hypothesis. mean ± std over seeds (sample
standard deviation); `flagged` = runs whose `best_val_loss` exceeds 1.5 × the median of their configuration
(kept in every mean). `default` rows are E1's `std_<enc>_s<seed>` runs (merged by E1; absent if E1 is not merged).

| encoder | ablation | setting | seeds | best val loss | AUC (SR) | max SIC (SR) | 1/ε_B @ ε_S=0.3 | AUC (full m_jj) | flagged |
|---|---|---|---|---|---|---|---|---|---|
