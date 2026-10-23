# Results — permutation-invariant graph autoencoders on the LHC Olympics 2020 R&D dataset

> Numbers marked *dev* come from the development protocol (`gaehep dev`: short CPU training, background
> validation events vs the 2-prong **signal validation** split); numbers marked *test* come from the held-out
> test splits.  **What was tuned on what:** models are *trained* on background only, but every design choice
> (latent size, kernel scales, training budget, event aggregation) was made with the 2-prong signal
> validation split, so 2-prong numbers are signal-tuned.  No choice was made with the 3-prong signal, but the
> test results (2- and 3-prong) of earlier models — the Chamfer set AE and the long-budget energy-flow AE —
> were seen during development; the training-budget choice was taken after the long run's test numbers were
> available (on its dev result, and against its better 3-prong number, §5).

## 1. Setup in one paragraph

LHCO 2020 R&D: 1M QCD dijet events and 100k W′(3.5 TeV) → X(500 GeV) Y(100 GeV), X, Y → qq̄ events, plus
100k events of the 3-prong variant (X, Y → qqq).  Events are clustered into anti-kT R = 1.0 jets with
`fastjet` and the two leading jets are kept with their top-100 constituents; our jets agree with the
official feature files for all events within 10⁻³ (relative) and for 99.9 % of them within 1.1 × 10⁻⁵
(1.2 × 10⁻⁵ for the 3-prong sample).  Each jet is a graph whose nodes are its constituents with features
relative to the jet axis (Δη, Δφ, ΔR, log pT, log E, log pT/pT_J, log E/E_J; the energies depend on the
jet's η), standardised on background-training jets.  Splits: background 60/10/30 % (train/val/test), 2-prong
signal 40/10/50 %, 3-prong signal test only.  Models are trained on background-training jets with early
stopping on the background-validation loss.  Every anomaly score is invariant to the order of the
constituents (unit tests permute the stored constituents and interleave extra padding, for every model below
including the legacy port), and the event score is a symmetric function of the two jets: max, sum, min or
**prod** = log F_B(s₁) + log F_B(s₂), where F_B is the CDF of background-validation jet scores (large only
when *both* jets are in the background tail).  Metrics are evaluated in the signal region SR =
3.3 < m_jj < 3.7 TeV: AUC, maximum significance improvement SIC = ε_S/√ε_B over cuts keeping ≥ 50 background
test events (≥ 20 for dev), rejection at ε_S = 0.3, and sculpting of the m_jj spectrum (Jensen–Shannon
divergence of the background m_jj histogram after cuts at ε_B = 10 % / 1 %; random cuts give 0.0003 / 0.003,
the statistical floor).

## 2. Reference scores

**Single observables (test, SR).**  AUC with the best of the four event aggregations in brackets:

| observable | 2-prong AUC | 3-prong AUC | comment |
|---|---:|---:|---|
| −τ21 | 0.893 (sum) | 0.684 (prod) | built for 2-prong jets |
| m_J | 0.811 (max) | 0.817 (max) | balanced, but strong m_jj sculpting (JSD 0.25 at ε_B = 1 %) |
| −τ32 | 0.688 (sum) | 0.854 (sum) | built for 3-prong jets |
| multiplicity | 0.424 (min) | 0.616 (prod) | colour-singlet signal jets have *fewer* constituents |

The N-subjettiness ratios are each good for one signal only; the jet mass is good for both but sculpts the
m_jj spectrum.

**Oracle background density (dev).**  Score = −log p_QCD(observables) with p_QCD estimated from a histogram
of 1.2M background-training jets (per jet, coarse bins):

| density in | max | sum | min | prod |
|---|---:|---:|---:|---:|
| (m_J) | 0.704 | 0.660 | 0.515 | 0.585 |
| (m_J, τ21) | 0.697 | 0.733 | 0.750 | 0.761 |
| (m_J, τ21, τ32) | 0.734 | 0.775 | 0.777 | 0.790 |
| (m_J, pT_J, τ21) | 0.729 | 0.783 | 0.794 | 0.799 |

So "the signal is rare under the background" in these coordinates is worth an AUC of ≈ 0.80: the X jet
(m = 500 GeV) sits in the QCD mass tail, but so do very light QCD jets, and the Y jet (m = 100 GeV) sits near
the QCD mass peak.  This is a reference point, not a bound (density scores depend on the coordinates, and
no joint event-level density was tried); the energy-flow VAE of §5 exceeds it.  Observables do better on the
signal they were designed for because they encode a signal hypothesis ("heavy", "2-prong").

## 3. The AIMS graph autoencoder collapses

The AIMS GAE (ported faithfully in `models/legacy_gae.py`: three PyG convolutions on *complete* graphs of
raw (pT, η, φ) node features, node-wise MLP decoder, MSE score) cannot represent individual nodes:

* `GCNConv` on a complete graph with self loops averages every node with weight 1/N, and `GINConv` with
  ε = 0 feeds x_i + Σ_{j≠i} x_j = Σ_j x_j to its MLP, so **every node of a graph gets exactly the same
  embedding** (`tests/test_legacy_collapse.py`).
* On a complete graph every message-passing layer averages over the whole jet, so differences between the
  nodes of a jet are washed out with depth.  On real LHCO jets (512 background jets, 3 random
  initialisations, `gaehep collapse`, `results/legacy_collapse.json`), the share of the embedding variance
  that lies *within* jets after layers 1 / 2 / 3 is (input: 0.98)

  | conv | layer 1 | layer 2 | layer 3 (latent) |
  |---|---:|---:|---:|
  | GCN | 2e-14 | 3e-15 | 7e-16 |
  | GIN (ε = 0) | 1e-13 | 4e-14 | 1e-14 |
  | GAT | 1e-4 | 4e-10 | 4e-14 |
  | SAGE (sum) | 0.23 | 5e-4 | 1.5e-6 |

  GCN and GIN are identical across nodes from the first layer (machine precision); GAT and SAGE shrink the
  within-jet differences by orders of magnitude per layer (GAT's attention ranks the keys identically for
  every query, and in SAGE-sum the neighbour sum outweighs the root term by a factor ~N).  Only SAGE, with
  its separate root weight, could learn to undo this — but it does not: after training on LHCO (dev
  budget) the latent share is 3e-7 for SAGE and 4e-16 for GCN.
* Consequence: the decoder reconstructs all objects of an event with one vector, so the AIMS anomaly score
  measured how far the raw (pT, η, φ) of the objects scatter around their event mean — an event-level
  statistic, not a learned graph representation.

*dev*: trained on LHCO jets with the AIMS hyper-parameters (top-30 constituents), the legacy GCN and SAGE
autoencoders behave identically and are close to random: SR event AUC 0.49–0.50 / 0.52 / 0.61 / 0.59
(max / sum / min / prod), anti-correlated with multiplicity (ρ = −0.70) and mass (ρ = −0.43).

GAE_HEP therefore uses local k-NN graphs in (Δη, Δφ), EdgeConv blocks with dynamic graphs, standardised
jet-relative features and an explicit bottleneck.

## 4. Constituent-level graph autoencoders are inverted anomaly detectors

| model | protocol | SR AUC max / sum / min / prod | 3-prong SR AUC (max) |
|---|---|---|---:|
| set AE, Chamfer loss on (Δη, Δφ, log pT) of all constituents | test, 25 epochs | 0.488 / 0.450 / 0.387 / 0.402 | 0.532 |
| set AE + latent flow | test | 0.569 / 0.421 / 0.339 / 0.366 | 0.583 |
| set AE, mass-quantile normalised | test | 0.380 / 0.346 / 0.363 / 0.353 | 0.417 |
| node-level GAE (2-d node bottleneck) | dev | 0.468 / 0.423 / 0.354 / 0.364 | – |
| Deep SVDD (same encoder; weak, not inverted) | dev | 0.629 / 0.635 / 0.605 / 0.625 | – |

Correctly permutation-invariant graph autoencoders that reconstruct the constituents rank the signal as
*more* normal than QCD.  The reconstructions (`results/runs/set_ae/figures/reconstructions.png`) show why:
the signal jets are colour-singlet decays, two (or three) clean, collimated prongs with little soft
radiation and fewer constituents, which an 8-dimensional jet latent reconstructs almost perfectly; a QCD jet
of the same mass gets its mass from wide-angle soft radiation that is hard to reconstruct point by point.
After removing the mass dependence (quantile normalisation) the score is anti-correlated with the signal
(AUC 0.38).  This is the "complexity bias" of autoencoders: reconstruction error measures how complicated
a jet is, not how unlikely it is.

## 5. Energy-flow graph autoencoder

Same EdgeConv encoder and jet latent, but the decoder emits 32 points in (Δη, Δφ) **with pT fractions**
(softmax) and the loss/score is the energy-weighted maximum mean discrepancy between the input energy flow
{(Δη_i, Δφ_i), z_i = pT_i/ΣpT} and its reconstruction, with a sum of Gaussian kernels at angular scales
0.02–0.4.  The distance is order-free on both sides, ignores padding, is exactly invariant under collinear
splittings and changes by O(z) under a soft emission of energy fraction z (all unit-tested), so soft
constituents barely enter the loss.  (The encoder still sees all constituents, and the score remains
correlated with multiplicity, ρ ≈ +0.6, largely through the jet mass.)

*dev*, 5 epochs × 120k jets on one CPU core (17 min), single seeds:

| variant | SR AUC max / sum / min / prod | max SIC (prod) | ρ(score, m_J) on QCD |
|---|---|---:|---:|
| latent 8, kernels 0.02–0.4 | 0.740 / 0.774 / 0.786 / 0.805 | 1.50 | +0.76 |
| latent 8, kernels 0.05–0.4 | 0.698 / 0.720 / 0.762 / 0.779 | 1.45 | +0.84 |
| **latent 4**, kernels 0.02–0.4 | 0.769 / 0.805 / 0.810 / **0.829** | 1.55 | +0.71 |
| latent 8 + latent flow, −log p(z) | 0.714 / 0.770 / 0.748 / – | – | +0.83 |
| latent 8 + conditional latent flow, −log p(z \| m_J, pT_J) | 0.674 / 0.752 / 0.747 / – | – | +0.80 |
| latent 8 + event-level flow, −log p(z_light, z_heavy) | 0.765 (event score) | 1.41 | +0.58 |

The inversion is gone and the energy-flow AE reaches (latent 8) and passes (latent 4) the oracle-density
reference of §2 (0.799).  On the 2-prong dev split a tighter bottleneck helps — plausibly because the
autoencoder can then no longer reproduce the rare, symmetric multi-prong energy flows it was not trained on —
and the fine kernel scale matters (0.02 helps).  The score is strongly mass-correlated; densities on the
latent space do not add discrimination (per jet, per event, or conditioned on the jet mass), and
conditioning on the jet mass does not remove the mass correlation of the density score (the conditional
entropy of the latent grows with m_J), so mass decorrelation is done by quantile normalisation instead.

**Training budget.**  The latent-8 model trained 5× longer (15 epochs × 200k jets, 80 min; patience and
validation-set size also changed) reaches a much lower background reconstruction loss but a *lower* 2-prong
dev AUC: 0.805 → 0.776 (prod), heavier-jet AUC 0.745 → 0.698 (single seeds).  A better background model also
reproduces the signal's clean two-prong jets better.  On the 3-prong test signal the long-trained model is
*better* (prod 0.828 vs 0.794 / 0.795 for two short-budget latent-8 VAEs; AE vs VAE), so the effect is
signal dependent.  The training budget was chosen like the other hyper-parameters, on the 2-prong dev split:
the final models use the short budget.

*test*, latent 8, long training (`results/runs/ef_ae`):

| signal | SR AUC max / sum / min / prod | max SIC (best) | 1/ε_B at ε_S = 0.3 (best) |
|---|---|---:|---:|
| 2-prong | 0.701 / 0.731 / 0.764 / 0.780 | 1.39 (prod) | 8.5 (prod) |
| 3-prong | 0.850 / 0.861 / 0.782 / 0.828 | 1.71 (sum) | 25.3 (max) |

With the aggregation chosen on the 2-prong dev split (prod) this model reaches 0.828 on the 3-prong signal,
below the dedicated −τ32 (0.854); its best aggregation (sum, 0.861) was not chosen beforehand.  m_jj
sculpting: JSD 0.010 (max) / 0.008 (prod) at ε_B = 1 %, versus 0.251 for a jet-mass cut and a floor of 0.003.

Sweep (dev, short budget, single seeds): latent 2 → 0.821 (prod; within the seed spread of latent 4, see
below); adding a 0.01 kernel at latent 4 → 0.822; decoder conditioned on (log m_J, log pT_J) at latent 4 →
0.797, and the score is *still* mass-correlated (ρ = +0.70): heavier jets have intrinsically more complex
energy flows, whatever the decoder is told about the mass.  **VAE**: making the latent variational
(β = 10⁻³, latent 8) leaves the reconstruction score unchanged (0.803) and adds the per-jet KL divergence of
the posterior as a second, density-like score: 0.811 / 0.832 / 0.778 / 0.817 (max / sum / min / prod;
heavier jet 0.810).  Their rank ensemble as built by `gaehep ensemble` (average of the two background CDFs)
gives 0.835 (prod), 0.845 (sum), max SIC 1.68 / 1.72.  At latent 4 the VAE's reconstruction score is as
good as the plain autoencoder's but its KL score is weak and unstable (0.657 / 0.676 / 0.714 for three
seeds, prod): with four latent dimensions the posteriors are broad and the KL carries little information.

Seed spread (dev, prod unless noted): latent-4 VAE reconstruction 0.826 / 0.824 / 0.837; latent-8 VAE
reconstruction 0.803 / 0.792 / 0.796 and KL 0.832 / 0.842 / 0.847 (sum).  Several dev choices above rest on
single-seed differences of the same size as this spread.

**Final models** (short budget, three seeds each, trained with `gaehep dev` and evaluated on the test splits
with `gaehep promote`; `configs/ef_vae.yaml` carries the same budget): (a) latent-4 energy-flow VAE,
reconstruction score (`ef_vae_s*`); (b) latent-8 energy-flow VAE, reconstruction + KL rank ensemble
(`ef_vae8_s*+ens`).

*test*, final model (a), mean ± standard deviation over three seeds; bootstrap (statistical) uncertainty of a
single seed in brackets; the "3-seed ensemble" averages the per-jet background CDFs of the three seeds:

| signal | aggregation | SR AUC | max SIC | 1/ε_B at ε_S = 0.3 | 3-seed ensemble AUC |
|---|---|---:|---:|---:|---:|
| 2-prong | **prod** (chosen on dev) | **0.829 ± 0.006** (±BOOT2A) | 1.57 ± 0.04 (±BOOT2S) | 15.1 | 0.832 |
| 2-prong | min / sum / max | 0.811 / 0.804 / 0.764 | 1.48 / 1.42 / 1.29 | | |
| 3-prong | **prod** (same choice) | **0.805 ± 0.002** (±BOOT3A) | 1.45 ± 0.01 (±BOOT3S) | 10.7 | 0.807 |
| 3-prong | sum / max (not chosen beforehand) | 0.845 / 0.835 | 1.63 / 1.58 | | |

The chosen configuration did not regress from dev (0.824–0.837) to test (0.825–0.836), seed by seed.
Jet-level AUC over the full m_jj range 0.74 (2-prong) and 0.73 (3-prong); m_jj sculpting JSD = 0.008 (prod,
seed mean) at ε_B = 1 %; distance correlation of the jet score with m_J 0.64.  After mass-quantile
normalisation (score decorrelated from m_J, dCor 0.02) the 3-seed ensemble keeps AUC 0.725 (2-prong, prod)
and 0.794 (3-prong): the jet mass accounts for about a third of the 2-prong AUC excess over 0.5
(0.332 → 0.225) but three quarters of the max-SIC excess over 1 (0.58 → 0.15); most of the 3-prong
separation is substructure.

Comparison with the observables under the same protocol (aggregation chosen on the 2-prong dev split, then
applied unchanged to both test signals; JSD floor ≈ 0.003):

| score | 2-prong AUC | 3-prong AUC | worse of the two | m_jj sculpting (JSD @ ε_B = 1 %) |
|---|---:|---:|---:|---:|
| −τ21 (sum) | **0.893** | 0.658 | 0.658 | 0.018 |
| −τ32 (sum) | 0.688 | **0.854** | 0.688 | 0.014 |
| m_J (max) | 0.811 | 0.817 | **0.811** | 0.251 |
| multiplicity (min) | 0.424 | 0.595 | 0.424 | 0.021 |
| set AE, Chamfer (max) | 0.488 | 0.532 | 0.488 | 0.004 |
| **energy-flow VAE, latent 4 (prod)** | 0.829 | 0.805 | 0.805 | **0.008** |
| energy-flow VAE, mass-decorrelated (3-seed ensemble) | 0.725 | 0.794 | 0.725 | 0.015 |

The energy-flow GNN is the only learned score that is useful on both signals; among single observables only
the jet mass is as balanced — with far stronger m_jj sculpting and a higher max SIC (1.9 against 1.6).  It
does not beat a dedicated observable on the signal that observable was designed for.

Final model (b), latent-8 VAE with reconstruction + KL ensemble: FINALB

## 6. Supervised reference

SUPERVISED

## 7. Weak supervision on the GNN latents

Unsupervised GNN scores give AUC ≈ 0.8 but only max SIC ≈ 1.6 — too little purity to make a small
resonance visible.  The standard way to win on this benchmark is weak supervision in the signal region
(CATHODE, "Back To The Roots").  `gaehep weak` runs the *idealised anomaly detector* (IAD): a classifier
separates SR "data" (36.3k background-train events + n injected 2-prong signal events from `sig_train`) from
an independent, perfectly modelled background template (the other 36.3k background-train SR events); no
event label is used.  Classifier: 3 bagged histogram gradient-boosted tree models, evaluated on the test SR
(36.6k background, 37.7k signal), 3 injection repetitions.  Event features are symmetric in the two jets
(ordered by mass).  The "supervised" column trains the same classifier on `sig_train` vs background with
labels (one training each).  max SIC on the test events (mean of 3 repetitions):

| injected S/B in SR | 0.41 % | 0.83 % | 1.7 % | 3.3 % | 8.3 % | supervised: max SIC / AUC |
|---|---:|---:|---:|---:|---:|---:|
| HL4: m_J1, Δm_J, τ21_J1, τ21_J2 | **1.6** | **7.0** | **9.4** | **10.1** | **10.6** | 11.2 / 0.970 |
| latent-4 VAE: latents of both jets (2 × 4) | 1.0 | 1.1 | 1.5 | 3.2 | 4.0 | 5.4 / 0.953 |
| latent-4 VAE: latents + jet masses | 1.0 | 1.8 | 8.0 | 8.9 | 9.2 | 11.0 / 0.974 |
| latent-4 VAE: anomaly scores + jet masses | 1.2 | 4.1 | 6.7 | 7.0 | 7.1 | 7.4 / 0.963 |
| latent-8 VAE: latents of both jets (2 × 8) | 1.0 | 1.1 | 1.5 | 4.6 | 6.6 | 10.9 / 0.968 |
| latent-8 VAE: latents + jet masses | 1.0 | 1.5 | 7.6 | 9.3 | 10.3 | 15.3 / 0.980 |

(`results/runs/ef_vae_s1/weak.json`, `results/runs/ef_vae8_s1/weak.json`, `figures/weak_iad.png`.)  The
permutation-invariant GNN representation — learned without labels, from the energy flow alone — carries
signal information comparable to the expert-designed HL4 features: with the two jet masses added the
supervised classifiers reach AUC 0.974 (latent 4) and 0.980 (latent 8) against 0.970 for HL4 (max SIC 11.0
and 15.3 against 11.2; single trainings, and max SIC sits at the 50-event floor where counting alone gives
≈ ±7 %).  In the weakly supervised search itself HL4 is better at every injection: the 10–18 dimensional
GNN representation needs more signal to be found (1.5–1.8 vs 7.0 at S/B = 0.8 %, the usual price of richer
inputs in weak supervision); from S/B ≈ 2 % on it reaches max SIC ≈ 8–10, close to HL4 — five to six times
the max SIC of the unsupervised GNN scores and about four times that of −τ21 (2.25).  Caveats: the data and
template events come from the background-training split, on which the autoencoder was trained (the injected
and test events were not); and our data and template samples are about three times smaller than in the
literature, so absolute SICs at fixed S/B are lower than published HL4 numbers (IAD / CATHODE max SIC
≈ 10–15 at S/B ≈ 0.6 % with ≈ 121k background events; see `docs/literature.md`).

## 8. Bump hunt

`gaehep bumphunt`: n signal events (test split) are injected into the 300k background test events, a cut on
the event score keeps a fraction ε of the *sideband* events (fixed threshold, or with `--mode qr` an
m_jj-dependent threshold fitted to the sideband score quantiles that keeps the background efficiency flat
in m_jj), the 4-parameter dijet function is fitted to the sidebands (2.7–5.0 TeV without the SR) and the
local significance in the SR is Z = √(2(n ln(n/b) − (n − b))), ignoring the uncertainty of the fit.  Mean Z
over 3 signal draws (fixed cut):

| injected (≈ 75 % in SR) | no cut | −τ21 (sum), ε = 10 % | ε = 2 % | EF-VAE (prod), ε = 10 % | ε = 2 % |
|---:|---:|---:|---:|---:|---:|
| 0 | −0.2 | −0.4 | −1.3 | −4.5 | −2.4 |
| 300 | +0.7 | +1.6 | +0.8 | −3.2 | −2.0 |
| 600 | +1.8 | +3.8 | +3.0 | −1.9 | −1.5 |
| 1200 | +3.7 | **+7.8** | +6.4 | +0.5 | −0.4 |

A −τ21 cut doubles the significance of the injected signal.  The EF-VAE cut increases the *response* to the
signal only modestly — Z(1200) − Z(0) = +5.1 with the 10 % cut against +3.9 without a cut (its SIC is 1.17
at ε_S = 0.3 and exceeds 1 only for ε_S ≳ 0.22) — and the result is dominated by a deficit at zero signal:
after the cut 3,479 events are observed in the SR against 3,754 expected from the sideband fit.  The
background efficiency of this particular cut is 0.0950 ± 0.0015 in the SR of the test background
(0.0948 / 0.0978 / 0.0928 / 0.0941 in its four 100-GeV bins) against 0.100 in the sidebands — but
0.101 ± 0.003 on the independent background-validation events, so this is a ≈ 3σ downward fluctuation of the
test background under this cut, not a systematic sculpting; the m_jj-dependent cut does not change it.
Either way the unsupervised score is too impure to help a bump hunt; the weakly supervised classifier of §7
would be the tool for that.

## 9. Conclusions

1. **The AIMS GAE could not represent individual objects.**  On complete graphs GCN and GIN produce exactly
   one embedding per event, and GAT and SAGE(sum) collapse within a few layers, also after training; the
   AIMS anomaly scores were an event-level spread statistic, and the ported model is close to random on
   LHCO (dev AUC 0.49–0.61).  Local k-NN graphs, EdgeConv, standardised relative features and a real
   bottleneck fix the architecture (all scores are tested to be invariant under node permutations and
   padding).
2. **A correctly permutation-invariant graph autoencoder is still the wrong tool if it reconstructs every
   constituent.**  Constituent-level reconstruction (Chamfer set AE, node-level GAE) is an *inverted*
   detector for this signal (SR AUC 0.35–0.49): colour-singlet signal jets are simpler than QCD jets of
   the same mass.  (A recent, not peer-reviewed report of inverted event-level graph-autoencoder AUCs on
   this dataset is listed, unverified, in `docs/literature.md`.)
3. **Reconstructing the energy flow instead works.**  With an IRC-safe, pT-weighted kernel distance (energy
   MMD) between the jet and its reconstruction, soft constituents barely enter the score.  The final
   latent-4 energy-flow VAE reaches SR AUC 0.829 ± 0.006 on the (tuning) 2-prong signal and 0.805 ± 0.002 on
   the 3-prong signal with the same event score, with m_jj sculpting close to the statistical floor (JSD
   0.008 at ε_B = 1 %), and keeps AUC 0.72–0.79 after the jet mass is decorrelated.
4. **Unsupervised scores reach about the level of an oracle background density.**  A histogram estimate of
   the background density in (m_J, pT_J, τ21) gives AUC 0.80; the energy-flow model matches or exceeds it.
   On the 2-prong dev split a tighter bottleneck and shorter training helped (a better background model also
   reconstructs the signal better), latent-space densities (flows, conditional flows, event-level flows) did
   not.  Unsupervised GNN scores give max SIC ≈ 1.5–1.7, below a dedicated observable on its own signal
   (τ21: 2.3) and too little to help a bump hunt.
5. **The learned representation carries signal information comparable to expert features.**  In the signal
   region, supervised classifiers on the permutation-invariant jet latents plus the two jet masses reach AUC
   0.974–0.980 against 0.970 for (m_J1, Δm_J, τ21_J1, τ21_J2).  In the idealised weakly supervised search
   HL4 is better at every signal fraction, much more so at low ones (SIC 7.0 vs 1.5–1.8 at S/B = 0.8 %); from
   S/B ≈ 2 % on the GNN representation reaches max SIC 8–10.

**Limitations.**  CPU-only training (2 cores): small models (48-wide EdgeConv, top-50 constituents), short
budgets and three seeds; the GPU notebook repeats the final models with five seeds and a scale study.  Design
choices were tuned on the 2-prong signal validation split, and earlier test results (including 3-prong) were
seen during development.  One signal model per topology; the 3-prong signal shares the colour-singlet
resonance structure.  Weak supervision is idealised (perfect background template, data and template drawn
from the autoencoder's training background, samples three times smaller than in the literature).
Significances in the bump hunt ignore fit and efficiency uncertainties.
