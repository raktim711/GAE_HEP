# Reference results on the LHC Olympics 2020 R&D dataset

Compiled on 2026-10-22 by an automated literature scan (paper text/tables only; numbers that appear only in
figures are marked *F-only*).  **Re-check every number against the paper before quoting it.**  Items marked
UNVERIFIED could not be confirmed in the source text.

Codes: U-bkg / U-data = unsupervised, trained on background only / on (signal-contaminated) data;
WS = weakly supervised (needs a signal region + sidebands); IAD = idealised anomaly detector (perfect
background template); SUP = fully supervised.  HL4 = {m_J1, Δm_J, τ21_J1, τ21_J2}.
SR = 3.3 < m_jj < 3.7 TeV.  inj1000 = 1000 injected signal events (≈772 in the SR on ≈121k background,
S/B ≈ 0.6%).

| Paper | Family / regime | Inputs | Result |
|---|---|---|---|
| Tsan et al., *Particle Graph Autoencoders and Differentiable, Learned EMD*, arXiv:2111.12849 | EdgeConv graph AE, U-bkg | constituents (px, py, pz) | ROC only (Fig. 1); per-jet score |
| Araz et al., *Graph theory inspired anomaly detection at the LHC*, arXiv:2506.19920 | EdgeConv graph AE, U-data (S/B = 3%) | 30 kT subjets | max SIC 2.94, AUC 0.925 (best configuration) |
| D'Avanzo, *Unsupervised ML for Anomaly Detection in LHC Collider Searches*, arXiv:2509.24723 | EGAT + Deep SVDD; transformer AE | constituent graphs | AUC ≈ 0.82 (event level), 0.75 (jet level) |
| Khan, *When Graph Autoencoders Look the Wrong Way…*, Zenodo 10.5281/zenodo.23432779 (2026 preprint, not peer reviewed) | node/event-level GAE, U-bkg | constituents | event-level AUC 0.40–0.44 (inverted); node-level 0.78; label-free HT/N 0.81, −τ21 0.80 (UNVERIFIED) |
| Vaslin et al., *GAN-AE*, arXiv:2305.15179 | GAN-AE, U-bkg | 42 high-level variables | AUC 0.82 (2-prong), 0.74 (3-prong); supervised BDT 0.99 |
| Bortolato et al., *Bump Hunting in Latent Space*, arXiv:2103.06595 | VAE, U-data | m_j, τ21, τ32 | significance improvement 2.75 at ε_S ≈ 0.5, S/B = 0.1% |
| Dillon et al., *Self-supervised Anomaly Detection for New Physics*, arXiv:2205.10380 | contrastive transformer (heads use labels) | constituents | AUC 0.918, max SIC 3.8 (latent + FCN) |
| Nachman & Shih, *ANODE*, arXiv:2001.04990 | WS | HL4 | AUC 0.82; SIC up to ≈7; inj1000 |
| Hallin et al., *CATHODE*, arXiv:2109.00546 | WS / IAD / SUP | HL4 | max SIC 14 (CATHODE), 11 (CWoLa hunting), 6.5 (ANODE); IAD ≈ CATHODE |
| Finke et al., *Back To The Roots*, arXiv:2309.13111 | BDT IAD / SUP | HL4, extended N-subjettiness | IAD ≈ 15, SUP ≈ 17; extended features ≈ 40 |
| Buhmann et al., *Full Phase Space Resonant AD*, arXiv:2310.06897 | CATHODE-like WS | constituents | max SIC ≈ 40–50 |

Conventions used by the literature (and adopted here where relevant):
* SR = 3.3–3.7 TeV, sidebands elsewhere (ANODE, CATHODE, BTR).
* Jet ordering: ANODE/CATHODE order by mass (m_J1 < m_J2); the official feature files order by pT (verified
  here: 99.9995% of events have pT_j1 >= pT_j2).  Our tables keep pT order and match the official jets 1:1.
* SIC = ε_S / sqrt(ε_B); minimum-background rules differ (BTR: <= 20% relative error on ε_B; Buhmann et al.:
  >= 10 events).  We require >= 50 background test events after the cut.
* AUC definitions differ (per jet vs per event; full sample vs SR).  We report both, per jet and per event
  (max / min / sum over the two jets), full range and SR.

Take-away for this project: fully unsupervised results on this dataset sit around AUC 0.75–0.93 with max SIC
≈ 2–4; weakly supervised HL4 methods reach max SIC ≈ 6.5–14 at S/B ≈ 0.6%, idealised/supervised ≈ 15–17.
Label-free single observables already give AUC ≈ 0.8, so they are mandatory baselines.
