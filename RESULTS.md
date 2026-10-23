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
| kernels 0.02–0.4 | 0.740 / 0.774 / 0.786 / **0.805** | 1.50 | +0.76 |
| kernels 0.05–0.4 | 0.698 / 0.720 / 0.762 / 0.779 | 1.45 | +0.84 |
| + latent flow, −log p(z) | 0.714 / 0.770 / 0.748 / – | – | +0.83 |
| + conditional latent flow, −log p(z \| m_J, pT_J) | 0.674 / 0.752 / 0.747 / – | – | +0.80 |

The inversion is gone and the energy-flow AE reaches the oracle-density level of §2 (0.805 vs 0.799).  The
fine kernel scale matters (0.02 helps).  The score is strongly mass-correlated; a latent density does not
add discrimination, and conditioning the density on the jet mass does not remove the mass correlation of
the score (the conditional entropy of the latent grows with m_J), so mass decorrelation is done by
quantile normalisation instead.  Sweep (latent size, finer kernels, VAE/KL score, mass-conditioned
decoder): *pending*.  Full-scale training and test results (2-prong and 3-prong): *pending*.

## 6. Supervised reference — *pending*

## 7. Weak supervision on the GNN latents — *pending*

## 8. Bump hunt — *pending*

## 9. Conclusions — *pending*
