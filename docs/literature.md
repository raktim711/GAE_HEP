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

---

## Method literature sweep (25 Oct 2026): what could improve an unsupervised jet score

Read through the fetch tool (abstract + body text; summaries below are of what the text says).  The fetch
tool became unavailable (rate limit) part-way through the sweep, so the second group was **not read** and is
listed from search hits only (those that the fetch tool could read later are in the second table).

**Read**

| Paper | What it says that matters here |
|---|---|
| Finke et al., *Autoencoders for unsupervised anomaly detection in high energy physics*, arXiv:2104.09051 (JHEP 06 (2021) 161) | CNN AE on 40×40 top-tagging images. Direct tagger (train QCD, find top) AUC 0.912 (MSE) / 0.928 (kernel MSE); the **inverse** tagger (train top, find QCD) 0.365 / 0.486 — **complexity bias**: the AE reconstructs sparse jets better whatever they are. Mitigations: intensity remapping of the pixels and a *kernel* (smeared) MSE; none of the inverse taggers beats a pixel-count tagger. (This paper does not use EMD or Chamfer losses.) |
| Fraser et al., *Challenges for unsupervised anomaly detection in particle physics*, arXiv:2110.06948 | VAE on jet images: **the lowest validation loss is not the best detector** (the network with one downsampling block had the lowest loss and was the worst detector); Wasserstein-trained networks were no better than MSE-trained ones; a training-free **k-medoids score with a Wasserstein distance** to a few QCD medoids matches the VAE (top 0.87 vs 0.84, W 0.71 vs 0.65 AUC). |
| Mikuni & Nachman, *High-dimensional and permutation invariant anomaly detection*, arXiv:2306.03933 | Point-cloud **diffusion model** (DeepSets + transformer layers), likelihood from the probability-flow ODE; score = −log p(jet) − (1/N) log p(particles | jet), i.e. the particle part is **normalised by the multiplicity**. Benchmarks: top-tagging data with a hidden-valley Z′ signal (QCD vs Z′: AUC 0.924, max SIC 3.8), not LHCO. |
| *A Normalized Autoencoder for LHC Triggers*, arXiv:2206.14225 | Energy-based AE (energy = reconstruction MSE; Langevin sampling of negatives, two-stage training, many stabilisation tricks) on jet images: top vs QCD AUC 0.91 (NAE) vs 0.875 (AE); the **inverse** direction 0.89 vs 0.58. Very sensitive to the pixel remapping exponent and the sampler temperature. |
| Ostdiek, *Deep Set Auto Encoders for Anomaly Detection in Particle Physics*, arXiv:2109.01695 (SciPost Phys. 12, 045) | Permutation-invariant (DeepSets, Chamfer-type loss) VAE on Dark Machines events: the best detector had **no reconstruction term at all** (β = 1, score = KL of the latent); fixed-target scores beat reconstruction scores, reason unclear. |
| Araz et al., arXiv:2506.19920 | (details of the table above) subjets from exclusive k_T clustering as nodes, pT as the only node feature, **edge features θ_ij, k_T,ij, z_ij reconstructed from pairs of latents**, 2-d bottleneck, trained on signal-contaminated data (S/B = 3 %); performance peaks at 25–30 subjets; relative geometry beats absolute positions. Their node feature is the absolute pT in GeV and no SR-restricted metric is given, so AUC 0.925 is not comparable to our signal-region numbers (jet pT alone separates signal from the bulk of a pT > 1.2 TeV QCD sample). |
| Tsan et al., arXiv:2111.12849; Dillon et al., arXiv:2205.10380 | abstracts only (learned EMD for particle graph AEs; contrastive transformer + latent density/bump hunt). |

**Read after the fetch tool came back** (26 Oct):

| Paper | What it says that matters here |
|---|---|
| Atkinson et al., *IRC-safe Graph Autoencoder for unsupervised anomaly detection*, arXiv:2204.12231 | Nodes carry only directions (Δη, Δφ, ΔR); **radius graph** (R₀ = 0.3) instead of k-NN (neighbourhoods of a k-NN graph change under soft/collinear splittings); messages **weighted by the neighbours' pT fractions**; edge-convolution encoder 128-64-2 (2-d latent), loss Σ z_i·MSE(node features), shown to be collinear- and soft-safe. Top vs QCD (R = 0.8, 550–650 GeV): AUC 0.902; no IRC-unsafe graph baseline and no mass study. The 2-d latent is almost perfectly (anti-)correlated across its two dimensions and with the loss; it correlates with two-point energy correlators. |
| Atkinson et al., *Anomaly detection with Convolutional Graph Neural Networks*, arXiv:2105.07988 | Micro-jets (anti-k_T, R = 0.1) as nodes of a complete graph with C/A, k_T and log m_ij edge features, 6-d latent per node; node *and* edge reconstruction ("crucial" — a node-only graph AE did not respond to the n-prong structure); loss correlates with jet mass; latent-space scores (mean readout of the latent) were better than the loss for some signals (best AUC 0.84 for a 700 GeV scalar → 4 jets, 0.74 top, 0.78 W). |
| Dillon et al., *Anomalies, representations, and self-supervision*, arXiv:2301.04660 | Contrastive (AnomalyCLR) permutation-invariant transformer representation of CMS ADC2021 events, scored with an autoencoder trained on the background representation; improves on the raw-data baseline for all signals (no numbers in the abstract). |

**Not read** (title/search hit only; do not quote numbers): *Anomalous Jet Identification via Sequence
Modeling*, arXiv:2105.09274; Dillon et al., *Better Latent Spaces for Better Autoencoders*,
arXiv:2104.08291; *Symmetries, safety, and self-supervision* (SciPost Phys. 12, 188); HEP-JEPA,
arXiv:2502.03933.  From memory (not re-checked):
likelihood regret (Xiao et al., NeurIPS 2020) and the input-complexity correction of likelihood scores
(Serrà et al., ICLR 2020) against complexity bias; random-network distillation (Burda et al. 2018) and Deep
SVDD (Ruff et al. 2018) as fixed-target scores.

**Ideas this suggests for our energy-flow graph VAE** (status filled in RESULTS.md §11):
1. complexity bias → a likelihood-type (cross-entropy) loss and an energy-exponent remapping instead of the
   bounded kernel distance; 2. "best fit ≠ lowest loss" is our inverted U (§6), so a *different* kind of
   information is needed, not more fitting: fixed-target scores (random-network distillation), training-free
   optimal-transport/kernel-embedding k-NN scores; 3. symmetry-aware inputs (principal-axis alignment, so
   that the latent does not spend a dimension on the jet's orientation); 4. the signal-region-local
   training that resonant searches use (train on a window around the SR).
