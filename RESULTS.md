# Results — permutation-invariant graph autoencoders on the LHC Olympics 2020 R&D dataset

> **Status: draft, updated as runs finish.**  Numbers marked *dev* come from the development protocol
> (background-validation vs 2-prong signal-validation events, short CPU training); numbers marked *test*
> come from the held-out test splits.  The 3-prong signal is never used for any design decision.

## 1. Setup in one paragraph

LHCO 2020 R&D: 1M QCD dijet events and 100k W′(3.5 TeV) → X(500 GeV) Y(100 GeV), X, Y → qq̄ events, plus
100k events of the 3-prong variant (X, Y → qqq).  Events are clustered into anti-kT R = 1.0 jets with
`fastjet` and the two leading jets are kept with their top-100 constituents; our jets agree with the
official feature files for every event (relative deviation < 1.1e-5).  Each jet is a graph whose nodes are its
constituents with direction-invariant features (Δη, Δφ, log pT, log E, log pT/pT_J, log E/E_J, ΔR),
standardised on background-training jets.  Splits: background 60/10/30 % (train/val/test), 2-prong signal
40/10/50 %, 3-prong signal test only.  Unsupervised models see background-training jets only.  Every anomaly
score is invariant to the order of constituents (unit tests permute and pad nodes) and the event score is a
symmetric function of the two jets: max, sum, min or **prod** = log F_B(s₁) + log F_B(s₂), where F_B is the
background CDF of jet scores (large only when *both* jets are in the background tail; label-free).  Metrics
are evaluated in the signal region SR = 3.3 < m_jj < 3.7 TeV: AUC, maximum significance improvement
SIC = ε_S/√ε_B over cuts keeping ≥ 50 background test events (≥ 20 for dev), rejection at ε_S = 0.3, and
sculpting of the m_jj spectrum (Jensen–Shannon divergence after cuts at ε_B = 10 % / 1 %).

## 2. Reference scores

**Single observables (test, SR).**  AUC with the best of the four event aggregations in brackets:

| observable | 2-prong AUC | 3-prong AUC | comment |
|---|---:|---:|---|
| −τ21 | 0.893 (sum) | 0.682 (min) | built for 2-prong jets |
| m_J | 0.811 (max) | 0.817 (max) | strong m_jj sculpting (JSD 0.25 at ε_B = 1 %) |
| −τ32 | 0.688 (sum) | 0.854 (sum) | built for 3-prong jets |
| multiplicity | 0.424 (min) | 0.596 (sum) | colour-singlet signal jets have *fewer* constituents |

No single observable is good for both signals; this is the point of model-agnostic (unsupervised) scores.

**Oracle background density (dev).**  Score = −log p_QCD(observables) with p_QCD the *true* background
density from a histogram of 1.2M background-training jets.  This is the best a purely density-based
anomaly score can do with these ideal low-dimensional features:

| density in | max | sum | min | prod |
|---|---:|---:|---:|---:|
| (m_J) | 0.704 | 0.660 | 0.515 | 0.585 |
| (m_J, τ21) | 0.697 | 0.733 | 0.750 | 0.761 |
| (m_J, τ21, τ32) | 0.734 | 0.775 | 0.777 | 0.790 |
| (m_J, pT_J, τ21) | 0.729 | 0.783 | 0.794 | 0.799 |

So "the signal is rare under the background" is worth an AUC of ≈ 0.80 here: the X jet (m = 500 GeV) sits in
the QCD mass tail, but so do very light QCD jets, and the Y jet (m = 100 GeV) sits near the QCD mass peak.
Observables do better on the signal they were designed for because they encode a signal hypothesis
("heavy", "2-prong").

## 3. The AIMS graph autoencoder collapses

The AIMS GAE (ported faithfully in `models/legacy_gae.py`: three PyG convolutions on *complete* graphs of
raw (pT, η, φ) node features, node-wise MLP decoder, MSE score) cannot represent individual nodes:

* `GCNConv` on a complete graph with self loops averages every node with weight 1/N, and `GINConv` with
  ε = 0 feeds x_i + Σ_{j≠i} x_j = Σ_j x_j to its MLP, so **every node of a graph gets exactly the same
  embedding** (`tests/test_legacy_collapse.py`).
* On a complete graph every message-passing layer averages over the whole jet, so differences between the
  nodes of a jet are washed out with depth ("over-smoothing" in one step).  On real LHCO jets (512
  background jets, 3 random initialisations, `gaehep collapse`), the share of the embedding variance that
  lies *within* jets after layers 1 / 2 / 3 is (input: 0.98)

  | conv | layer 1 | layer 2 | layer 3 (latent) |
  |---|---:|---:|---:|
  | GCN | 2e-14 | 3e-15 | 7e-16 |
  | GIN (ε = 0) | 1e-13 | 4e-14 | 1e-14 |
  | GAT | 1e-4 | 4e-10 | 4e-14 |
  | SAGE (sum) | 0.23 | 5e-4 | 1.5e-6 |

  GCN and GIN are identical across nodes from the first layer (machine precision); GAT and SAGE shrink the
  within-jet differences by orders of magnitude per layer (GAT's attention ranks the keys identically for
  every query, and in SAGE-sum the neighbour sum outweighs the root term by a factor ~N).  Only SAGE, with
  its separate root weight, can learn to undo this; trained models: *pending*.
* Consequence: the decoder reconstructs all objects of an event with one vector, so the AIMS anomaly score
  measured how far the raw (pT, η, φ) of the objects scatter around their event mean — an event-level
  statistic, not a learned graph representation.

GAE_HEP therefore uses local k-NN graphs in (Δη, Δφ), EdgeConv blocks with dynamic graphs, standardised
jet-relative features and an explicit bottleneck.  Legacy models on LHCO (dev): *pending*.

## 4. Constituent-level graph autoencoders are inverted anomaly detectors

| model | protocol | SR AUC max / sum / min / prod | 3-prong SR AUC (max) |
|---|---|---|---:|
| set AE, Chamfer loss on (Δη, Δφ, log pT) of all constituents | test, 25 epochs | 0.488 / 0.450 / 0.387 / – | 0.532 |
| set AE + latent flow | test | 0.569 / 0.421 / 0.339 / – | 0.583 |
| set AE, mass-quantile normalised | test | 0.380 / 0.346 / 0.363 / – | 0.417 |
| node-level GAE (2-d node bottleneck) | dev | 0.468 / 0.423 / 0.354 / 0.364 | – |
| Deep SVDD (same encoder) | dev | 0.629 / 0.635 / 0.605 / 0.625 | – |

Correctly permutation-invariant graph autoencoders that reconstruct the constituents rank the signal as
*more* normal than QCD.  The reconstructions (`results/runs/set_ae/figures/reconstructions.png`) show why:
the signal jets are colour-singlet decays, two (or three) clean, collimated prongs with little soft
radiation and fewer constituents, which an 8-dimensional jet latent reconstructs almost perfectly; a QCD jet
of the same mass gets its mass from wide-angle soft radiation that is hard to reconstruct point by point.
After removing the mass dependence (quantile normalisation) the score is anti-correlated with the signal
(AUC 0.38).  This is the "complexity bias" of autoencoders: reconstruction error measures how complicated
a jet is, not how unlikely it is.

## 5. Energy-flow graph autoencoder

Same EdgeConv encoder and 8-dimensional jet latent, but the decoder emits 32 points in (Δη, Δφ) **with pT
fractions** (softmax) and the loss/score is the energy-weighted maximum mean discrepancy between the
input energy flow {(Δη_i, Δφ_i), z_i = pT_i/ΣpT} and its reconstruction, with a sum of Gaussian kernels at
angular scales 0.02–0.4.  The distance is order-free on both sides, ignores padding, is exactly invariant
under collinear splittings and changes by O(z) under a soft emission of energy fraction z (all unit-tested),
so the score depends on the hard prong structure rather than on soft multiplicity.

*dev*, 5 epochs × 120k jets on one CPU core (17 min):

| variant | SR AUC max / sum / min / prod | max SIC (prod) | ρ(score, m_J) on QCD |
|---|---|---:|---:|
| latent 8, kernels 0.02–0.4 | 0.740 / 0.774 / 0.786 / 0.805 | 1.50 | +0.76 |
| latent 8, kernels 0.05–0.4 | 0.698 / 0.720 / 0.762 / 0.779 | 1.45 | +0.84 |
| **latent 4**, kernels 0.02–0.4 | 0.769 / 0.805 / 0.810 / **0.829** | 1.55 | +0.71 |
| latent 8 + latent flow, −log p(z) | 0.714 / 0.770 / 0.748 / – | – | +0.83 |
| latent 8 + conditional latent flow, −log p(z \| m_J, pT_J) | 0.674 / 0.752 / 0.747 / – | – | +0.80 |
| latent 8 + event-level flow, −log p(z_light, z_heavy) | 0.765 (event score) | 1.41 | +0.58 |

The inversion is gone and the energy-flow AE reaches (latent 8) and passes (latent 4) the oracle-density
level of §2 (0.799): a tighter bottleneck helps, because the autoencoder can then no longer reproduce
the rare, symmetric multi-prong energy flows it has not been trained on.  The fine kernel scale matters
(0.02 helps).  The score is strongly mass-correlated; densities on the latent space do not add
discrimination (per jet, per event, or conditioned on the jet mass), and conditioning on the jet mass does
not remove the mass correlation of the density score (the conditional entropy of the latent grows with
m_J), so mass decorrelation is done by quantile normalisation instead.

**Longer training hurts.**  The latent-8 model trained 5× longer (15 epochs × 200k jets, 80 min) reaches a
much lower background reconstruction loss but a *lower* signal efficiency: dev AUC (prod) 0.805 → 0.776,
heavier-jet AUC 0.745 → 0.698.  The better the autoencoder reproduces the background, the better it also
reproduces the signal's clean two-prong jets.  The training budget is therefore a hyper-parameter chosen on
the dev split like the others, and the final models use the short budget (5 epochs × 120k jets).

*test*, latent 8, long training (`results/runs/ef_ae`):

| signal | SR AUC max / sum / min / prod | max SIC (best) | 1/ε_B at ε_S = 0.3 (best) |
|---|---|---:|---:|
| 2-prong | 0.701 / 0.731 / 0.764 / 0.780 | 1.39 (prod) | 8.5 (prod) |
| 3-prong (never used for any choice) | 0.850 / **0.861** / 0.782 / 0.828 | 1.71 (sum) | 25.3 (max) |

The same unsupervised score that is moderate on the 2-prong signal is as good as the best dedicated
observable on the 3-prong signal (−τ32: 0.854; −τ21 only 0.682), and it barely sculpts the m_jj spectrum
(JSD 0.010 at ε_B = 1 % for max, 0.008 for prod, versus 0.251 for a jet-mass cut).

Sweep (dev, short budget): latent 2 → 0.821 (prod); adding a 0.01 kernel at latent 4 → 0.822 (no gain
over 0.829); decoder conditioned on (log m_J, log pT_J) at latent 4 → 0.797, and the score is *still*
mass-correlated (ρ = +0.70): heavier jets have intrinsically more complex energy flows, whatever the
decoder is told about the mass.  **VAE**: making the latent variational (β = 10⁻³, latent 8) leaves the reconstruction score
unchanged (0.803) and adds the per-jet KL divergence of the posterior as a second, density-like score:
0.811 / 0.832 / 0.778 / 0.817 (max / sum / min / prod; heavier jet 0.810).  An equal-weight rank
ensemble of the two (sum of per-jet background log-CDFs) gives 0.839 (prod), max SIC 1.70.

At latent 4 the VAE's reconstruction score is as good as the plain autoencoder's (0.826 and 0.824 for two
seeds) but its KL score is weak (0.657 prod): with only four latent dimensions the posterior of every jet
is broad and the KL carries little information, so the equal-weight ensemble (0.798) does not help there.

Seed spread (dev, prod unless noted): latent-4 VAE reconstruction 0.826 / 0.824 / 0.837; latent-8 VAE
reconstruction 0.803 / 0.792 / 0.796 and KL 0.832 / 0.842 / 0.847 (sum), so at latent 8 the KL term is the
stronger and stable score, while at latent 4 it varies between seeds (0.66 / 0.71).

**Final models** (short budget, three seeds each, evaluated on the test splits with `gaehep promote`):
(a) latent-4 energy-flow VAE, reconstruction score (`ef_vae_s*`); (b) latent-8 energy-flow VAE,
reconstruction + KL rank ensemble (`ef_vae8_s*`), the best dev configuration (0.839, max SIC 1.70).

*test*, final model (a), mean ± standard deviation over three seeds (the "3-seed ensemble" averages the
per-jet background CDFs of the three seeds):

| signal | aggregation | SR AUC | max SIC | 1/ε_B at ε_S = 0.3 | 3-seed ensemble AUC |
|---|---|---:|---:|---:|---:|
| 2-prong | **prod** (chosen on dev) | **0.829 ± 0.006** | 1.57 ± 0.04 | 15.1 | 0.832 |
| 2-prong | min / sum / max | 0.811 / 0.804 / 0.764 | 1.48 / 1.42 / 1.29 | | |
| 3-prong | **prod** (same choice) | **0.805 ± 0.002** | 1.45 ± 0.01 | 10.7 | 0.807 |
| 3-prong | sum / max (not chosen beforehand) | 0.845 / 0.835 | 1.63 / 1.58 | | |

Dev (0.824–0.837) and test (0.825–0.836) agree seed by seed, so the dev protocol did not overfit the
validation signal.  Jet-level AUC 0.74 (2-prong) and 0.73 (3-prong); m_jj sculpting JSD = 0.009 (prod) at
ε_B = 1 %; distance correlation of the jet score with m_J 0.64.  After mass-quantile normalisation (score
decorrelated from m_J, dCor 0.02) the AUCs are 0.725 (2-prong, prod) and 0.794 (3-prong): roughly half of
the 2-prong separation is the jet mass, the rest — and most of the 3-prong separation — is substructure.

Comparison with the observables under the same protocol (aggregation chosen on the 2-prong dev split, then
applied unchanged to both test signals):

| score | 2-prong AUC | 3-prong AUC | worse of the two | m_jj sculpting (JSD @ ε_B = 1 %) |
|---|---:|---:|---:|---:|
| −τ21 (sum) | **0.893** | 0.658 | 0.658 | 0.018 |
| −τ32 (sum) | 0.688 | **0.854** | 0.688 | 0.014 |
| m_J (max) | 0.811 | 0.817 | **0.811** | 0.251 |
| multiplicity (min) | 0.424 | 0.595 | 0.424 | 0.021 |
| set AE, Chamfer (max) | 0.488 | 0.532 | 0.488 | 0.004 |
| **energy-flow VAE, latent 4 (prod)** | 0.829 | 0.805 | 0.805 | **0.009** |
| energy-flow VAE, mass-decorrelated | 0.725 | 0.794 | 0.725 | 0.015 |

The energy-flow GNN is the only learned score that is useful on both signals, and among single scores only
the jet mass is as balanced — at the price of strongly sculpting the m_jj spectrum (and with a higher max
SIC: 1.9 against 1.6).  It does not beat a dedicated observable on the signal that observable was designed
for.

Final model (b), latent-8 VAE with reconstruction + KL ensemble: *pending*.

## 6. Supervised reference — *pending*

## 7. Weak supervision on the GNN latents

Unsupervised scores give AUC ≈ 0.8 but only max SIC ≈ 1.5 — too little purity to make a small resonance
visible.  The standard way to win on this benchmark is weak supervision in the signal region (CATHODE, "Back
To The Roots").  `gaehep weak` runs the *idealised anomaly detector* (IAD): a classifier separates SR "data"
(36.3k background-train events + n injected 2-prong signal events) from an independent, perfectly modelled
background template (the other 36.3k background-train SR events); no event label is used.  Classifier:
bagged histogram gradient-boosted trees (3 models), evaluated on the test SR (36.6k background, 37.7k
signal), 3 injection repetitions.  Event features are symmetric in the two jets (ordered by mass):

| injected S/B in SR | 0.41 % | 0.83 % | 1.7 % | 3.3 % | 8.3 % | supervised, same features |
|---|---:|---:|---:|---:|---:|---:|
| HL4: m_J1, Δm_J, τ21_J1, τ21_J2 | 1.6 | **7.0** | **9.4** | **10.1** | **10.6** | 11.2 |
| GNN latents of both jets (2 × 4) | 1.0 | 1.1 | 1.5 | 3.2 | 4.0 | 5.4 |
| GNN latents + jet masses | 1.0 | 1.8 | 8.0 | 8.9 | 9.2 | 11.0 |
| GNN anomaly scores + jet masses | 1.3 | 4.1 | 6.7 | 7.0 | 7.1 | 7.4 |

(max SIC on the test events, mean of 3 repetitions; `results/runs/ef_vae_s1/weak.json`,
`figures/weak_iad.png`.)  The permutation-invariant GNN representation, learned without labels and
without any physics input beyond the energy flow, plus the two jet masses carries as much information about
the signal as the expert-designed HL4 features (supervised ceilings 11.0 vs 11.2), and with ≳ 2 % signal
the idealised weakly supervised search on it reaches max SIC ≈ 8–9 — five to six times the purity of any
unsupervised score.  At lower signal fractions the 10-dimensional representation needs more signal than
the 4-dimensional HL4 to be found (1.8 vs 7.0 at S/B = 0.8 %).  The 4-d latents alone miss part of the
mass information (ceiling 5.4).  Our data and template samples are about three times smaller than in the
literature, so absolute SICs at fixed S/B are lower than published HL4 numbers (IAD / CATHODE max SIC
≈ 10–15 at S/B ≈ 0.6 % with ≈ 121k background events; see `docs/literature.md`).

## 8. Bump hunt

`gaehep bumphunt`: n signal events (test split) are injected into the 300k background test events, a cut on
the event score keeps a fraction ε of the *sideband* events (fixed threshold, or with `--mode qr` an
m_jj-dependent threshold fitted to the sideband score quantiles that keeps the background efficiency flat
in m_jj), the 4-parameter dijet function is fitted to the sidebands (2.7–5.0 TeV without the SR) and the
local significance in the SR is Z = √(2(n ln(n/b) − (n − b))).  Mean Z over 3 signal draws (fixed cut):

| injected (≈ 75 % in SR) | no cut | −τ21 (sum), ε = 10 % | ε = 2 % | EF-VAE (prod), ε = 10 % | ε = 2 % |
|---:|---:|---:|---:|---:|---:|
| 0 | −0.2 | −0.4 | −1.3 | −4.6 | −2.4 |
| 300 | +0.8 | +1.6 | +0.8 | −3.2 | −2.0 |
| 600 | +1.8 | +3.8 | +3.0 | −1.9 | −1.5 |
| 1200 | +3.7 | **+7.8** | +6.4 | +0.6 | −0.4 |

A −τ21 cut doubles the significance of the injected signal.  The unsupervised EF-VAE score does not: its SIC
exceeds 1 only for loose cuts (ε_S ≳ 0.5, Fig. `results/figures/sic_2prong_SR.png`), so any cut tight enough to
matter removes signal faster than √background.  At zero injection the EF-VAE cut leaves a deficit in the
SR: the background efficiency of the cut is about 5 % lower in the two SR bins than in the neighbouring
sidebands (0.093–0.096 vs 0.099–0.105 for a 10 % cut), which with 36k SR events and a significance that
ignores the uncertainty of the fit and of the efficiency appears as −4σ.  The m_jj-dependent cut does not
change this (it equalises the efficiency *between* sidebands, not inside the SR).  A bump hunt with this
score would need the weakly supervised classifier of §7, not the raw unsupervised score.

## 9. Conclusions

1. **The AIMS GAE could not represent individual objects.**  On complete graphs GCN and GIN produce exactly
   one embedding per event, GAT and SAGE(sum) collapse within a few layers; the AIMS anomaly scores were an
   event-level spread statistic.  Local k-NN graphs, EdgeConv, standardised relative features and a real
   bottleneck fix the architecture (all scores are tested to be invariant under node permutations and
   padding).
2. **A correctly permutation-invariant graph autoencoder is still the wrong tool if it reconstructs every
   constituent.**  Constituent-level reconstruction (Chamfer set AE, node-level GAE) is an *inverted*
   detector for this signal (SR AUC 0.39–0.49): colour-singlet signal jets are simpler than QCD jets of
   the same mass.  (A recent, not peer-reviewed report of inverted event-level graph-autoencoder AUCs on
   this dataset is listed, unverified, in `docs/literature.md`.)
3. **Reconstructing the energy flow instead works.**  An IRC-safe, pT-weighted kernel distance (energy MMD)
   between the jet and its reconstruction removes the dependence on soft multiplicity.  The final
   latent-4 energy-flow VAE reaches SR AUC 0.829 ± 0.006 on the 2-prong signal and 0.805 ± 0.002 on the
   unseen 3-prong signal with one fixed event score, sculpts the m_jj spectrum less than τ21 (JSD 0.009 at
   ε_B = 1 %), and keeps AUC 0.73–0.79 after the jet mass is decorrelated.
4. **Unsupervised scores are near their ceiling on this benchmark.**  An oracle density estimate of the
   background in (m_J, pT_J, τ21) gives AUC 0.80; the energy-flow model matches it.  Model choices matter
   in non-obvious ways: a tighter bottleneck and *shorter* training improve detection, because a better
   background model also reconstructs the signal better.  Latent-space densities (flows, conditional flows,
   event-level flows) add nothing.  Unsupervised scores give max SIC ≈ 1.5–1.7, below a dedicated
   observable on its own signal (τ21: 2.3) and too little for a bump hunt.
5. **The learned representation is nevertheless as informative as expert features.**  Fed to an idealised
   weakly supervised classifier in the signal region, the permutation-invariant jet latents plus the two jet
   masses reach the same supervised ceiling as (m_J1, Δm_J, τ21_J1, τ21_J2) (max SIC 11.0 vs 11.2) and max
   SIC 8–9 once the signal fraction is ≳ 2 %; HL4 needs less signal (SIC 7.0 at 0.8 %).

**Limitations.**  CPU-only training (2 cores): small models (48-wide EdgeConv, top-50 constituents) and short
budgets; the GPU notebook repeats the final models with five seeds and a scale study.  One signal model per
topology; the 3-prong signal shares the colour-singlet resonance structure.  Weak supervision is idealised
(perfect background template, data samples three times smaller than in the literature).  Significances in
the bump hunt ignore fit and efficiency uncertainties.
