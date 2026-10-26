# Results — permutation-invariant graph autoencoders on the LHC Olympics 2020 R&D dataset

> Numbers marked *dev* come from the development protocol (`gaehep dev`: short CPU training, background
> validation events vs the 2-prong **signal validation** split); numbers marked *test* come from the held-out
> test splits.  **What was tuned on what:** models are *trained* on background only, but every design choice
> (latent size, kernel scales, training budget, event aggregation) was made with the 2-prong signal
> validation split, so 2-prong numbers are signal-tuned.  No choice was made with the 3-prong signal, but the
> test results (2- and 3-prong) of earlier models — the Chamfer set AE and the long-budget energy-flow AE —
> were seen during development; the training-budget choice was taken after the long run's test numbers were
> available (on its dev result, and against its better 3-prong number, §5).  The hyper-parameter study of §6
> made all of its choices on the same 2-prong validation split; its final model was scored on the test splits
> only after the choice.

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
(`ef_vae8_s*+ens`).  The hyper-parameter study of §6 replaces model (a) by the same VAE trained for 2 epochs
with k = 8 neighbours.

*test*, final model (a), mean ± standard deviation over three seeds; bootstrap (statistical) uncertainty of a
single seed in brackets; the "3-seed ensemble" averages the per-jet background CDFs of the three seeds:

| signal | aggregation | SR AUC | max SIC | 1/ε_B at ε_S = 0.3 | 3-seed ensemble AUC |
|---|---|---:|---:|---:|---:|
| 2-prong | **prod** (chosen on dev) | **0.829 ± 0.006** (±0.002) | 1.57 ± 0.04 (±0.01) | 15.1 | 0.832 |
| 2-prong | min / sum / max | 0.811 / 0.804 / 0.764 | 1.48 / 1.42 / 1.29 | | |
| 3-prong | **prod** (same choice) | **0.805 ± 0.002** (±0.002) | 1.45 ± 0.01 (±0.01) | 10.7 | 0.807 |
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

*test*, final model (b), latent-8 VAE, three seeds (mean ± standard deviation; SR AUC / max SIC):

| score | 2-prong, prod | 2-prong, sum | 3-prong, prod | 3-prong, max | JSD @ ε_B = 1 % (prod) |
|---|---|---|---|---|---:|
| reconstruction | 0.799 ± 0.006 / 1.46 | 0.773 / 1.35 | 0.795 ± 0.002 / 1.43 | 0.847 / 1.68 | – |
| KL | 0.827 ± 0.007 / 1.62 | 0.842 / 1.69 | 0.738 ± 0.006 / 1.25 | 0.793 / 1.49 | – |
| **rank ensemble (rec + KL)** | **0.841 ± 0.003 / 1.70** | 0.851 / 1.74 | 0.789 ± 0.002 / 1.42 | 0.854 / 1.74 | 0.028 |
| 3-seed ensemble of the above | 0.845 / 1.73 | 0.854 / 1.76 | 0.792 / 1.43 | 0.862 / 1.78 | 0.029 |

Model (b) is the best score on the (tuning) 2-prong signal (0.841 vs 0.829 for model (a)), but its KL part
is more mass-like (dCor with m_J 0.77 vs 0.64): it is worse on the 3-prong signal with the pre-chosen
aggregation (0.789 vs 0.805) and sculpts m_jj more (JSD 0.028 vs 0.008).  Model (a) is the more
model-agnostic of the two.

## 6. Hyper-parameter study

Protocol: every run is a dev run (2-prong signal validation split, SR, prod aggregation of the
reconstruction score; one CPU core per run, two runs in parallel; 78 runs, about 22 CPU-hours); leads were
repeated with 3–5 seeds (the seed-to-seed standard deviation of one configuration is 0.002–0.010, so
single-seed differences below ≈ 0.01 are within the noise); the 3-prong signal was never looked at, and the
test splits only for
the final configuration.  All tables: `results/dev/hp_study.md`; figures `results/figures/hp_effects.png`
and `hp_budget.png` (`scripts/hp_study.py`).  Reference: the previous final model (§5, latent-4 VAE,
β = 10⁻³, kernels 0.02–0.4, 5 epochs × 120k jets), dev AUC 0.829 ± 0.007 (3 seeds; ± = sample standard
deviation over seeds throughout).

**Round 1: one change to the previous final model** (21 variants; dev AUC, seeds in brackets):

| change | dev AUC | change | dev AUC |
|---|---:|---|---:|
| **2 epochs** | **0.847 ± 0.002 (5)** | β = 10⁻⁴ / 3·10⁻³ | 0.830 / 0.841 |
| **kernels 0.02–0.1** (without 0.2, 0.4) | **0.843 ± 0.005 (3)** | β = 3·10⁻² / 0.1 | 0.779 / 0.780 |
| **β = 10⁻²** | **0.841 ± 0.008 (3)** | latent 3 / 6 | 0.839 / 0.796 |
| 3 epochs | 0.841 ± 0.005 (3) | encoder width 32 / 64 | 0.838 / 0.830 |
| 10 epochs | 0.819 | encoder depth 2 / 4 | 0.836 / 0.825 |
| kernels 0.02–0.05 | 0.845 | kernel sets with a 0.01 scale (3 sets) | 0.824–0.831 |

**The three gains do not add up.**  Each pair of {2 epochs, kernels 0.02–0.1, β = 10⁻²}, and all three
together, gives 0.815–0.817 — *below* the reference.  The training-budget scan explains why
(`hp_budget.png`): the dev AUC is an inverted U in the background validation loss the model reaches,

| training (epochs × jets) | 1 × 120k | 2 × 60k | 2 × 120k | 3 × 120k | 2 × 240k | 5 × 120k | 10 × 120k |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation loss | 0.414 | 0.293 | 0.24–0.25 | 0.21 | 0.19 | 0.18 | 0.15 |
| dev AUC | 0.813 | 0.850 | 0.847 | 0.841 | 0.836 | 0.829 | 0.819 |

so the score is best at an *intermediate* fit of the background: an autoencoder that fits the background
better also reproduces the signal's clean two-prong jets better (§5), and one that fits it too poorly does
not yet resolve the jet substructure.  Narrower kernels and a stronger KL weight make the reconstruction
coarser in the same way; combined with each other or with short training they under-fit (a higher
validation loss than either change alone; the scores also become more mass-like, ρ(score, m_J) = +0.77–0.81
instead of +0.64–0.72).  The same mechanism explains the round-2 losses below (smaller batches mean more
steps and over-fit; a deeper encoder with a lower learning rate under-fits in 2 epochs).

**Round 2: one change to the 2-epoch model** (31 variants, reference 0.847 ± 0.002, 5 seeds): all within
−0.007 … +0.009 except the changes that move the fit level — 1 epoch 0.813, 2 × 240k jets 0.836, batch 128
0.835, encoder depth 4 with learning rate 3·10⁻⁴ 0.822 ± 0.002 — and 30 instead of 50 constituents per jet
(0.835; 100 constituents: 0.855, single seed, at four times the training time).  The graph (k = 6 … 24
neighbours, static instead of dynamic k-NN, max instead of mean aggregation), the encoder size (width 32–96,
depth 2–4), the latent size (3–6), β (10⁻⁴–10⁻³), VAE vs plain AE, the decoder (128–512 wide, 16–64
points) and the kernel set (0.02–0.05 … 0.05–0.8) do not matter at this level.  **Round 3**: the best round-2 variants with seeds —
k = 8 neighbours 0.853 ± 0.004 (5 seeds), encoder depth 4 0.852 ± 0.003, learning rate 3·10⁻⁴
0.852 ± 0.002 (3 seeds) — are within one standard deviation of each other; k = 8 (best mean, and the
cheapest) is the new final configuration (`configs/ef_vae_k8_e2.yaml`: the previous model trained for
2 epochs with k = 8; about 5 CPU-minutes of training per model).  Its margin over k = 12 (+0.006) is partly a
selection effect among three similar candidates.

**Score kinds.**  The rank ensemble of a VAE's reconstruction and KL scores reached 0.870 in one seed (kernels
0.02–0.1) but varied between 0.777 and 0.870 over three seeds: at latent 4 the KL score is weak, and after
2 epochs it is at or below random for most seeds (dev AUC 0.43–0.70) — whether the ensemble helps depends on
the seed.  Ensembles over seeds do not rescue it (2-epoch model: 0.868 for three seeds, 0.851 for five),
so the final model keeps the reconstruction score.  (A cross-model ensemble of the 2-epoch reconstruction
score with the KL score of the 5-epoch latent-8 VAE reaches 0.863 on dev but is strongly mass-correlated,
ρ = +0.78, like model (b) of §5.)

*test*, final model of the study (five seeds, mean ± standard deviation; ensemble = average of the five
per-jet background CDFs) against the previous final model of §5 (three seeds); SR, prod aggregation:

| model | 2-prong AUC | max SIC | 1/ε_B at ε_S = 0.3 | 3-prong AUC | max SIC | JSD @ ε_B = 1 % |
|---|---:|---:|---:|---:|---:|---:|
| previous (5 epochs, k = 12) | 0.829 ± 0.006 | 1.57 ± 0.04 | 15.1 | **0.805 ± 0.002** | 1.45 ± 0.01 | **0.008** |
| **new (2 epochs, k = 8)** | **0.853 ± 0.005** | **1.74 ± 0.05** | **20.4** | 0.799 ± 0.002 | 1.44 ± 0.01 | 0.012 |
| new, 5-seed ensemble | 0.856 | 1.76 | 20.9 | 0.802 | 1.45 | 0.013 |
| new, sum / max / min aggregation | 0.839 / 0.802 / 0.832 | | | 0.825 / 0.814 / 0.754 | | |
| previous, sum / max / min | 0.804 / 0.764 / 0.811 | | | 0.845 / 0.835 / 0.756 | | |
| new, mass-decorrelated (ensemble + qn) | 0.767 | 1.38 | | 0.767 | | 0.011 |
| previous, mass-decorrelated (3-seed ensemble + qn) | 0.725 | 1.15 | | 0.794 | | 0.015 |

The dev result carries over to the test splits seed by seed (dev 0.860 / 0.851 / 0.851 / 0.854 / 0.851, test
0.861 / 0.850 / 0.849 / 0.853 / 0.850; bootstrap uncertainty of one seed ±0.001): on the 2-prong signal the
study gains 0.024 in AUC, 11 % in max SIC and 35 % in background rejection at ε_S = 0.3, and the same
direction holds for the weak supervision (§8) and bump hunt (§9).  It does **not** carry over to the 3-prong
signal, which played no role in the study: the chosen aggregation is unchanged within 0.006, the sum and max
aggregations lose 0.02, and the mass-decorrelated score loses 0.03 there while gaining 0.04 on the 2-prong
signal; m_jj sculpting rises from 0.008 to 0.012 (random cuts 0.003, a jet-mass cut 0.25).  This is the
signal dependence of the training budget already seen in §5 (the long-trained latent-8 AE was the better
3-prong detector): the training length decides which kind of jet the autoencoder still reconstructs badly,
so tuning it on one signal model tunes the detector toward that signal.

### GPU continuation

`notebooks/kaggle_gaehep.ipynb` on a Kaggle T4 (3 h 52 min including data preparation; all outputs in
`results/dev/gpu_*`, `results/runs/gpu_*` and `results/gpu/`): a 72-point `gaehep sweep --score best` grid —
latent size 3 / 4 / 6 / 8 × β = 10⁻³ / 10⁻² × kernel scales 0.02–0.05 / 0.02–0.1 / 0.02–0.4 × 2 / 3 / 5 epochs,
every point scored with the reconstruction score and with the rec + KL rank ensemble (ranking in
`results/dev/gpu_sweep.md`) — then three seeds for the best three points, five seeds of the winner scored on
the test splits, and scale variants of the winner (two seeds each, dev).  It reproduces the CPU picture
independently:

* At β = 10⁻³ all 36 grid points lie between 0.79 and 0.857; the best are trained for 2 epochs with narrow
  kernels, and at that point the latent size hardly matters (2 epochs, kernels 0.02–0.05: 0.850–0.857 for
  latent 3–8).  2 epochs give the best reconstruction AUC in all 12 (latent, kernel) cells (one tie with
  3 epochs) and 5 epochs the lowest or equal-lowest in 11 of 12.
* At β = 10⁻² the reconstruction AUC sits at a fixed value whatever the latent size and training length —
  0.787–0.788 for all 12 points with kernels 0.02–0.05, 0.777–0.779 for 17 of the 24 with the wider kernel
  sets: the posterior collapses, the decoder emits one template jet whatever the input, and the score becomes
  the jet's distance to that template.  The points that escape (latent 6/8, and latent 4 with kernels
  0.02–0.4 and 5 epochs at 0.855) are unstable over seeds (the latter: 0.834 ± 0.021).
* The rec + KL ensemble again has single good points (0.853 for latent 8, kernels 0.02–0.4, 3 epochs) but is
  unstable over seeds (0.747–0.802 seed means for the three confirmed points), so the reconstruction score
  wins.
* Winner (seed-confirmed): latent 4, β = 10⁻³, kernels 0.02–0.05, 2 epochs, k = 12 — dev 0.855 ± 0.002
  (5 seeds; latent 8: 0.853 ± 0.002, 3 seeds), the same recipe as the CPU study's final model
  (0.853 ± 0.004), found by a different search.

*test*, five seeds (`gpu_best_s*`, ensemble `gpu_best_seeds`), SR, prod aggregation, next to the CPU
study's final model:

| model | 2-prong AUC | max SIC | 1/ε_B at ε_S = 0.3 | 3-prong AUC | JSD @ ε_B = 1 % |
|---|---:|---:|---:|---:|---:|
| CPU final (2 epochs, k = 8, kernels 0.02–0.4) | 0.853 ± 0.005 | 1.74 ± 0.05 | 20.4 | 0.799 ± 0.002 | 0.012 |
| GPU winner (2 epochs, k = 12, kernels 0.02–0.05) | 0.855 ± 0.002 | 1.74 ± 0.01 | 25.5 | 0.801 ± 0.004 | 0.018 |
| GPU winner, 5-seed ensemble | 0.859 | 1.77 | 26.9 | 0.805 | 0.017 |
| GPU winner, mass-decorrelated (ensemble + qn) | 0.783 | 1.50 | 24.0 | 0.755 | 0.005 |

The two searches end at the same 2-prong AUC and max SIC and the same flat 3-prong AUC; the narrower kernels
buy a higher background rejection at ε_S = 0.3 (25.5 vs 20.4) for slightly more m_jj sculpting.  The weak
supervision (`results/runs/gpu_best_s1/weak.json`, five bagged models instead of three, so the HL4 row is
2.0 / 8.1 / 9.4 / 10.1 / 10.7 here) gives the anomaly scores + masses 1.5 / 5.4 / 7.6 / 7.8 / 7.9 and the
latents + masses 1.1 / 2.9 / 8.5 / 9.1 / 9.5 at S/B = 0.41 … 8.3 %, the bump hunt a response
Z(1200) − Z(0) = +6.4 at ε = 10 % (−3.4 → +3.0), both comparable to the CPU final model.

Scale variants of the winner (dev, two seeds, against 0.855 ± 0.002): **100 constituents 0.876 / 0.866**
(max SIC 1.96 / 1.83), 64 decoder points 0.863 / 0.852, a 4-layer encoder 0.854 / 0.829, a 128-wide encoder
0.850 / 0.850, 300k instead of 120k jets per epoch 0.845 / 0.843 (more training again lowers the AUC).  More
constituents is the one direction of scale that helps — both seeds lie above all five 50-constituent seeds,
and the CPU study saw +0.008 with 100 constituents at 2 epochs — but it has not been run on the test splits.

## 7. Supervised reference

Same EdgeConv encoder with a binary head, trained per jet on 2-prong signal-training jets against
background-training jets (labels; 5 epochs × 200k jets, `results/runs/classifier`): the idealised upper bound
for the 2-prong signal.

| test, SR | 2-prong AUC / max SIC | 3-prong AUC / max SIC (not trained on) | JSD @ ε_B = 1 % |
|---|---|---|---:|
| sum of the two jet logits | 0.982 / 15.7 | 0.958 / 8.8 | 0.32 |
| max | 0.973 / 9.0 | 0.941 / 4.2 | 0.14 |

The unsupervised energy-flow VAE recovers about two thirds of the supervised AUC excess over 0.5 on the
2-prong signal ((0.829 − 0.5)/(0.982 − 0.5) = 0.68; 0.73 for the model of §6) but only ≈ 4 % of its max-SIC
excess over 1 (5 % for §6) — the unsupervised scores rank the bulk of the signal correctly but do not isolate
it.  The supervised classifier
transfers well to the 3-prong signal, but it has learned the signal's jet masses and sculpts m_jj
strongly.

## 8. Weak supervision on the GNN latents

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
| HL4: m_J1, Δm_J, τ21_J1, τ21_J2 | 1.6 | **7.0** | **9.4** | **10.1** | **10.6** | 11.2 / 0.970 |
| latent-4 VAE: latents of both jets (2 × 4) | 1.0 | 1.1 | 1.5 | 3.2 | 4.0 | 5.4 / 0.953 |
| latent-4 VAE: latents + jet masses | 1.0 | 1.8 | 8.0 | 8.9 | 9.2 | 11.0 / 0.974 |
| latent-4 VAE: anomaly scores + jet masses | 1.2 | 4.1 | 6.7 | 7.0 | 7.1 | 7.4 / 0.963 |
| latent-8 VAE: latents of both jets (2 × 8) | 1.0 | 1.1 | 1.5 | 4.6 | 6.6 | 10.9 / 0.968 |
| latent-8 VAE: latents + jet masses | 1.0 | 1.5 | 7.6 | 9.3 | 10.3 | 15.3 / 0.980 |
| §6 model (2 epochs, k = 8): latents of both jets | 1.1 | 1.4 | 2.2 | 4.2 | 4.8 | 5.8 / 0.950 |
| §6 model: latents + jet masses | 1.3 | 2.2 | 9.2 | 9.1 | 9.6 | 11.5 / 0.974 |
| §6 model: anomaly scores + jet masses | 1.7 | 6.2 | 8.0 | 8.2 | 8.2 | 8.5 / 0.967 |

(`results/runs/ef_vae_s1/weak.json`, `results/runs/ef_vae8_s1/weak.json`, `results/runs/hp_final_s1/weak.json`,
`figures/weak_iad.png`.)  The model of §6 improves every GNN row: its anomaly scores plus the jet masses match
HL4 at the lowest signal fraction (1.7 vs 1.6) and reach 6.2 at S/B = 0.8 % (4.1 for the latent-4 VAE of §5),
its latents plus masses 9.2 at S/B = 1.7 % (HL4 9.4).  The
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

## 9. Bump hunt

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
The model of §6 (5-seed ensemble, prod, fixed cut at ε = 10 %) gives Z = −3.5 / −1.9 / −0.3 / +3.1 for
0 / 300 / 600 / 1200 injected events: a response Z(1200) − Z(0) = +6.6, between the previous model's +5.1 and
−τ21's +8.2, with a similar deficit at zero signal (−4.0 / −1.5 for 0 / 1200 events with the 2 % cut).
Either way the unsupervised score is too impure to help a bump hunt; the weakly supervised classifier of §8
would be the tool for that.

## 10. Conclusions

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
   MMD) between the jet and its reconstruction, soft constituents barely enter the score.  The latent-4
   energy-flow VAE reaches SR AUC 0.829 ± 0.006 on the (tuning) 2-prong signal and 0.805 ± 0.002 on the
   3-prong signal with the same event score — 0.853 ± 0.005 and 0.799 ± 0.002 after the hyper-parameter
   study (§6) — with m_jj sculpting close to the statistical floor (JSD 0.008 / 0.012 at ε_B = 1 %), and keeps
   AUC 0.73–0.79 (after §6: 0.77 on both signals) when the jet mass is decorrelated.
4. **Unsupervised scores reach about the level of an oracle background density.**  A histogram estimate of
   the background density in (m_J, pT_J, τ21) gives AUC 0.80; the energy-flow model matches or exceeds it.
   On the 2-prong dev split a tighter bottleneck and shorter training helped (a better background model also
   reconstructs the signal better), latent-space densities (flows, conditional flows, event-level flows) did
   not.  Unsupervised GNN scores give max SIC ≈ 1.5–1.8, below a dedicated observable on its own signal
   (τ21: 2.3) and too little to help a bump hunt; against a supervised EdgeConv classifier (AUC 0.982,
   max SIC 15.7) they recover 68–73 % of the AUC excess but only 4–5 % of the SIC excess.
5. **The learned representation carries signal information comparable to expert features.**  In the signal
   region, supervised classifiers on the permutation-invariant jet latents plus the two jet masses reach AUC
   0.974–0.980 against 0.970 for (m_J1, Δm_J, τ21_J1, τ21_J2).  In the idealised weakly supervised search
   HL4 is better at almost every signal fraction, by far at low ones for the GNN latents (SIC 7.0 vs 1.5–2.2
   at S/B = 0.8 %), less so for the anomaly scores plus jet masses of the §6 model (6.2; equal to HL4 at
   S/B = 0.4 %); from S/B ≈ 2 % on the GNN representation reaches max SIC 8–10.
6. **The training budget is the hyper-parameter that matters, and its optimum depends on the signal.**  In
   about 80 dev runs the anomaly score is best at an intermediate fit of the background (an inverted U in
   the background validation loss); shorter training, narrower kernel scales and a stronger KL weight each
   move the fit toward that optimum and do not combine, and at the optimum the graph, encoder, decoder and
   latent-size choices barely matter.  The chosen model (2 epochs, k = 8; five seeds) improves the 2-prong
   test AUC by 0.024, max SIC by 11 %, the background rejection at ε_S = 0.3 by 35 % and the weakly
   supervised and bump-hunt results — but not the never-tuned 3-prong signal (0.799 vs 0.805), and it
   sculpts m_jj slightly more.  Tuning an anomaly detector's training length on one signal model tunes it
   toward that signal.  An independent 72-point GPU grid lands on the same recipe (2 epochs, latent 4,
   β = 10⁻³, narrow kernels; test 0.855 ± 0.002 / 0.801 ± 0.004) and finds one further direction: 100
   instead of 50 constituents per jet (+0.016 on the 2-prong dev split, two seeds).

**Limitations.**  CPU-only training (2 cores): small models (48-wide EdgeConv, top-50 constituents — 96-wide
encoders did not help, 100 constituents did on the dev split, §6), short budgets and three to five seeds;
one GPU run (Kaggle T4) repeated the study on a 72-point grid with scale variants.  Design choices — the whole hyper-parameter study included — were
tuned on the 2-prong signal validation split, and earlier test results (including 3-prong) were seen during
development.  The initial weights of the CPU study's runs were drawn from the process's random state (not
from `train.seed`; fixed since, so new runs are exactly reproducible); seeds of one configuration are
therefore independent but not bit-reproducible.  One signal model per topology; the 3-prong signal shares the colour-singlet
resonance structure.  Weak supervision is idealised (perfect background template, data and template drawn
from the autoencoder's training background, samples three times smaller than in the literature).
Significances in the bump hunt ignore fit and efficiency uncertainties.

## 11. Literature-informed variants (follow-up study)

After the hyper-parameter study (§6) the anomaly-detection and graph-autoencoder literature was surveyed for
ideas that could lift the energy-flow graph VAE (papers, what was read and what was not: `docs/literature.md`).
The transferable ideas were implemented as options of the existing model (the defaults reproduce the saved
models exactly; unit tests cover permutation/padding invariance of every new variant) and as dev-protocol tools
(`gaehep dev-kme`, `dev-profile`), and every variant was run with the dev protocol of §6 (2 epochs × 120k jets,
k = 8 unless stated, SR metrics against the 2-prong signal *validation* split).  Reference for all rows: the
final model of §6, dev prod-AUC **0.853 ± 0.004** (five seeds; seed 1: 0.860), max SIC 1.73 ± 0.04.

### 11.1 Variants that did not help (dev, single seed unless stated)

| idea (source) | what was done | dev AUC (prod) | max SIC |
|---|---|---:|---:|
| Likelihood instead of kernel distance (complexity bias: Finke et al., NAE, likelihood-based scores) | decoder also emits a width σ per point; energy-weighted cross-entropy −Σ z_i log Σ_j w_j N(x_i; μ_j, σ_j) replaces the MMD, β = 10⁻³ / 10⁻² | 0.808 / 0.790 | 1.61 / 1.64 |
| the same, scored with the MMD of the NLL-trained model | | 0.838 / 0.806 | 1.66 / 1.53 |
| MMD + NLL loss | β = 10⁻² | 0.803 | 1.65 |
| Energy exponent p (the energy flow weights pT^p instead of pT) | p = 0.5 / 2 | 0.860 / 0.784 | 1.77 / 1.34 |
| Symmetry-aware inputs (Atkinson et al.; Ostdiek): principal-axis alignment | jets rotated/reflected into their energy principal-axis frame; full / 1 epoch | 0.748 / 0.805 | 1.34 / 1.48 |
| Signal-region-local training (as resonant searches do) | train only on jets from events with m_jj in 3.3–3.7 / 3.0–4.0 TeV | 0.850 / 0.851 | 1.71 / 1.73 |
| Fixed-target score (Ostdiek; random-network distillation) | student EdgeConv net regresses a frozen random teacher on background jets | 0.485 | 1.00 |
| Training-free kernel-embedding score (Fraser et al.) | random-Fourier-feature embedding of the energy flow (the MMD kernel mean map), k-NN distance to background jets; raw / whitened | 0.816 / 0.841 | 1.60 / 1.77 |
| the same, Mahalanobis distance of the embedding | | 0.833 | 1.70 |
| Density of the error profile (complexity-bias correction) | GMM / Mahalanobis fitted to the per-kernel-scale MMD errors of the trained model (+ KL, + latent mean) | 0.37–0.63 | 1.00–1.09 |
| KL score of the VAE (Ostdiek) | score = KL(q(z|x) ‖ p(z)) instead of the reconstruction | 0.45 (five seeds) | 1.02 |

Observations.  (i) The same inverted U as in §6 shows up again: the likelihood loss, the energy exponents and
the kernel-embedding scores are all *different* ways of changing how well the background is fitted, and they
land at or below the final model, never above it — the plateau at ≈ 0.85–0.86 is shared by methods that look
unrelated.  (ii) The energy exponent is the one that came out equal: p = 0.5 gives 0.860 on one seed against
0.860 for the same seed of the baseline — no gain.  (iii) Training-free scores are competitive but not better
(whitened k-NN 0.841 without any training, 0.84 max SIC 1.77), consistent with Fraser et al.  (iv) Everything
that adds a density estimate on top of a trained model (error profiles, latent flows of §5) *removes* the
information the reconstruction score carries; the profile densities sit at chance or below.

### 11.2 An infrared- and collinear-safe encoder generalises better to the unseen signal

The energy-flow *loss* of §5 is IRC safe, but the EdgeConv encoder is not: its nodes carry log pT features
and its neighbourhoods are k-nearest-neighbour graphs, both of which change when a constituent splits into
two collinear halves or a soft particle is emitted.  Following Atkinson et al. (arXiv:2204.12231; see
`docs/literature.md`) `encoder: ewmp` replaces it by **energy-weighted message passing on radius graphs**
(`models/ewmp.py`): nodes carry only directions (Δη, Δφ, ΔR), a node's neighbours are all constituents
within ΔR < 0.3, messages are averaged with the neighbours' pT fractions as weights, and the readout is the
pT-weighted mean and mean-square of the node features.  The pT fractions enter only as weights, so the
encoder — and with it the whole score — is exactly invariant under collinear splittings (unit-tested, also
with the NLL loss and the principal-axis frame) and changes by O(z) under a soft emission of energy
fraction z; all scores stay permutation- and padding-invariant.  Same size as the baseline (126,648 vs
127,368 parameters), same training recipe (2 epochs, latent 4, β = 10⁻³, kernels 0.02–0.4); training
takes about three times as long (15 vs 5 min at 50 constituents on one core) because of the dense pairwise
messages.

*dev*, three seeds: **0.857 ± 0.002** (max SIC 1.74 ± 0.03) against 0.853 ± 0.004 (1.73 ± 0.04) for the
EdgeConv model — equal within the seed spread.  Radius 0.3 is what Atkinson et al. use; two radii (0.1 and
0.4) gave 0.850 and 1 / 3 epochs 0.854 / 0.844 (single seeds), i.e. the same flat dependence on the
training length as in §6.

*test*, SR, prod aggregation (EdgeConv: the five seeds of §6, `hp_final_s*`; IRC-safe: `lt_ewmp_s1–3`; the
3-prong signal was never used for any choice):

| model | 2-prong AUC | max SIC | 1/ε_B at ε_S = 0.3 | 3-prong AUC | max SIC | 1/ε_B at ε_S = 0.3 | JSD @ ε_B = 1 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| EdgeConv + k-NN (final model of §6), 5 seeds | 0.853 ± 0.005 | 1.74 ± 0.05 | 20.4 | 0.799 ± 0.002 | 1.44 ± 0.01 | 10.1 | 0.0124 |
| **IRC-safe radius-graph encoder**, 3 seeds | 0.857 ± 0.002 | 1.75 ± 0.02 | 19.8 | **0.817 ± 0.005** | **1.52 ± 0.01** | 11.0 | **0.0087** |
| EdgeConv, 5-seed ensemble | 0.856 | 1.76 | 20.9 | 0.802 | 1.45 | 10.3 | 0.0126 |
| IRC-safe, 3-seed ensemble | 0.860 | 1.77 | 20.1 | 0.820 | 1.52 | 11.1 | 0.0114 |
| EdgeConv, ensemble + m_J-decorrelation (qn) | 0.767 | 1.38 | | 0.767 | 1.28 | | 0.011 |
| IRC-safe, ensemble + m_J-decorrelation (qn) | 0.766 | 1.33 | | 0.800 | 1.41 | | 0.013 |

The 2-prong numbers — the signal everything was tuned on — are the same.  The never-tuned 3-prong signal
improves in every aggregation (prod +0.018, max +0.010, sum +0.010, min +0.018 AUC; the seed spread is
0.002–0.008), its max SIC by 0.08 and its background rejection at ε_S = 0.3 by 9 %, and the mass-decorrelated
3-prong AUC by 0.033; the seed-averaged m_jj sculpting at ε_B = 1 % drops by 26–30 % (prod 0.0124 → 0.0087; max
0.018 → 0.014; sum 0.019 → 0.013) — though the ensembles end up at the same sculpting level
(0.011–0.013).  The weakly supervised search (seed 1 of each model, `results/runs/lt_ewmp_s1/weak.json`) and the bump
hunt of §9 (the two seed ensembles) are unchanged: the anomaly score + masses reaches max SIC 1.4 / 6.4 / 8.3 / 8.2 / 8.4 at
S/B = 0.41 … 8.3 % (EdgeConv: 1.7 / 6.2 / 8.0 / 8.2 / 8.2), the latents + masses 1.1 / 2.9 / 8.5 / 9.2 / 9.5
(1.3 / 2.2 / 9.2 / 9.1 / 9.6), and the injected-signal response Z(1200) − Z(0) is +6.4 against +6.6 (zero-
signal bias −2.5 against −3.5 at the 10 % cut).

**Is the gain really about IRC safety?**  A direct measurement on the saved models (`results/irc_probe.json`,
8000 + 8000 signal-region background and signal-validation events, jet-level SR AUC; constituents are never
dropped, only split or added):

| perturbation | EdgeConv: ΔAUC / rank correlation of the background scores | IRC-safe: ΔAUC / rank correlation |
|---|---|---|
| hardest constituent split in two collinear halves | +0.001 / 0.994 | 0.000 / 1.000 |
| five hardest constituents split | +0.001 / 0.978 | 0.000 / 1.000 |
| every constituent split (50 → up to 100) | −0.027 / 0.901 | 0.000 / 1.000 |
| 10 soft emissions of 0.2 % pT each | −0.003 / 0.969 | −0.002 / 0.995 |
| 10 soft emissions of 1 % pT each | −0.065 / 0.565 | −0.034 / 0.823 |

The EdgeConv model's score is *not* a function of the jet's energy flow alone — it moves under perturbations
that leave the energy flow (or, for soft emissions, almost leave it) unchanged — whereas the IRC-safe
encoder is invariant by construction, and robust to soft emissions up to the O(z) change of the target.
The model with the sensitivity is the one that generalises worse to a signal with a different constituent
structure (3 prongs); we have not separated this from other differences of the two encoders (radius graph,
weighted pooling), so this is the best-supported explanation, not a proof.

<!-- S11.3 -->
### 11.3 100 constituents per jet, and both changes together

The GPU study of §6 found one direction of scale that helped on the dev split — 100 instead of 50
constituents per jet — and it was never run on the test splits.  `configs/ef_vae_gpu_n100.yaml` is that
recipe (EdgeConv k = 16, latent 4, β = 10⁻³, narrow kernels 0.02–0.05, 2 epochs, 32 decoder points) and was
trained on the CPU with three seeds; combined with the IRC-safe encoder (radius 0.3, same decoder, loss and
kernels; `ewmp_checkpoint` keeps the pairwise messages of 100 constituents at about 4 GB of memory at the cost
of about 30 % more training time) it gives the last row.

*dev*: EdgeConv 100 constituents **0.866 ± 0.003** (0.863 / 0.866 / 0.870; max SIC 1.84 ± 0.02) against
0.853 ± 0.004; IRC-safe 100 constituents 0.871 (max SIC 1.91, one seed; training takes 90 min on one core).

*test*, SR, prod aggregation, mean ± standard deviation over seeds (the first two rows repeat the baselines of
§6 and §11.2; the kernel set is what separates the GPU winner from the final model of §6):

| model | seeds | 2-prong AUC | max SIC | 1/ε_B at ε_S = 0.3 | 3-prong AUC | max SIC | 1/ε_B at ε_S = 0.3 | JSD @ ε_B = 1 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EdgeConv + k-NN, 50 constituents (final model of §6) | 5 | 0.853 ± 0.005 | 1.74 ± 0.05 | 20.4 | 0.799 ± 0.002 | 1.44 ± 0.01 | 10.1 | 0.0124 |
| EdgeConv + k-NN, 50 constituents, narrow kernels 0.02–0.05 (GPU winner of §6) | 5 | 0.855 ± 0.002 | 1.74 ± 0.01 | 25.5 | 0.801 ± 0.004 | 1.43 ± 0.01 | 11.0 | 0.0175 |
| IRC-safe radius-graph encoder, 50 constituents | 3 | 0.857 ± 0.002 | 1.75 ± 0.02 | 19.8 | 0.817 ± 0.005 | 1.52 ± 0.01 | 11.0 | 0.0087 |
| EdgeConv + k-NN, 100 constituents | 3 | 0.866 ± 0.003 | 1.86 ± 0.02 | 33.3 | 0.805 ± 0.006 | 1.44 ± 0.02 | 11.6 | 0.0158 |
| **IRC-safe encoder, 100 constituents** | 1 | 0.871 | 1.92 | 36.1 | 0.834 | 1.56 | 15.6 | 0.0113 |

Reading the table row by row:

* **100 constituents** (third row against the GPU winner of the second): +0.011 in 2-prong AUC, +7 % in max
  SIC and **+31 % in background rejection at ε_S = 0.3** (25.5 → 33.3, 20.4 → 33.3 against the final model of
  §6); on the 3-prong signal +0.004 in AUC and +5 % in rejection, i.e. not hurting where the training length
  of §6 did.  m_jj sculpting is slightly lower than the GPU winner's but above the final model's.  The more
  constituents the energy-flow target resolves, the closer the 50-constituent truncation (not an IRC-safe
  operation: the cut is on pT) comes to the exact energy flow.
* **IRC-safe encoder at 100 constituents** (fourth row): the 2-prong figures improve again and the 3-prong
  AUC jumps from 0.805 to 0.834 — more than the two changes give separately — and the combined model is
  the best in every column of the table except the sculpting of the 50-constituent IRC-safe model, including the never-tuned 3-prong signal (AUC 0.834 against 0.799 for the model
  that was the headline of §6; max SIC 1.56 against 1.44; background rejection 15.6 against 10.1) and the
  m_jj sculpting (0.0113 against 0.0124 and 0.0158).  The other aggregations tell the same story (3-prong AUC
  of the combined model with max / sum / min: 0.805 / 0.838 / 0.804, against 0.814 / 0.825 / 0.754 for the final model of §6).
* The m_J-decorrelated scores (quantile normalisation in m_J, single seeds; `+qn` runs): final model 2-prong
  AUC 0.773 / 3-prong 0.750 / JSD 0.0113, EdgeConv at 100 constituents 0.802 / 0.759 / 0.0065, IRC-safe at 50
  constituents 0.753 / 0.788 / 0.0146, **IRC-safe at 100 constituents 0.789 / 0.792 / 0.0048** — the decorrelated
  combined score is the only one at 0.79 on both signals and sculpts the spectrum close to the
  statistical floor (0.003).
* Bump hunt (§9; prod, injected-signal response Z(1200) − Z(0), one seed each): the combined model gives
  +7.3 at the 10 % cut and +6.0 at the 2 % cut (zero-signal bias −3.3 and −4.2), against +6.6 and +2.5 for the final
  model of §6 (the weakly supervised classifiers remain the tool for a real search).
* Weakly supervised search (§8, one seed each): the anomaly score + masses gains most — max SIC 1.6 / 6.9 / 9.0 / 9.1 / 9.4 at S/B = 0.41 … 8.3 %
  against 1.7 / 6.2 / 8.0 / 8.2 / 8.2 for the final model of §6 (HL4 in the same run: 2.4 / 7.9 / 9.3 / 10.1 / 10.7) —
  while the latents + masses are unchanged (1.3 / 3.9 / 8.6 / 9.0 / 9.4 against 1.3 / 2.2 / 9.2 / 9.1 / 9.6).

**What this does and does not show.**  The 100-constituent choice came from a grid that was scored on the
2-prong dev split, so the 2-prong columns are signal-tuned in the way every earlier row is; the 3-prong
numbers and the IRC-safe encoder were never tuned on (the encoder was added on a physics argument and its dev
result was only checked to be level), but the 3-prong test numbers of both were seen, so any further choice
that uses them would not be clean.  The two changes were compared with the earlier models seed by seed, not
by a joint hyper-parameter search; the narrow kernels of the 100-constituent recipe were not re-tuned for the
IRC-safe encoder.  The unsupervised max SIC remains ≈ 1.9 at best.
<!-- /S11.3 -->
