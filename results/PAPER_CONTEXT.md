# GAE_HEP: context for writing the paper

This file holds everything a writer (human or model) needs to draft the paper from this results folder.

**Facts and references.**
- Every number below comes from a file in `results/`; the file is named next to it.
- Before citing a reference, verify it on arXiv. The arXiv IDs below were checked on 31 Oct 2026, but titles and author lists were not re-checked when writing this file.
- Repository: github.com/raktim711/GAE_HEP (code in `src/gaehep`).
- Frozen protocol: `docs/STUDY_PROTOCOL.md`, including the deviations D1–D9. D9 is the follow-up of the bias audit of
  6 November (`study/audit_followup.md`), which narrowed several claims; section 7 below already reflects it.
- Results draft: `paper/results_draft.md`.

**Do not overclaim.** The paper's claims must stay within section 7 ("What the paper may claim").

---

## 1. The question and why it matters

Unsupervised anomaly detection at the LHC often uses autoencoders on jet constituents. Graph neural networks
(message passing) and transformers (attention) are commonly assumed to give better jet representations than set
models without interactions. Whether that holds for *unsupervised* detection had not been tested under controlled
conditions.

**Question:** inside one permutation-invariant energy-flow autoencoder, does the interaction step improve
unsupervised anomaly detection on signals nobody tuned on? And does infrared and collinear (IRC) safety matter?

The study holds everything fixed except the encoder's interaction step:
- decoder, loss, latent size, training recipe and seeds are shared;
- all hypotheses and decision rules were frozen in a protocol before any study run;
- results are judged on untuned signals: the R&D 3-prong signal and LHC Olympics black boxes 1 and 3, with black
  box 2 as a null test.

**Origin.** This continues the author's AIMS (Cape Town) work on graph autoencoders for the ATLAS High Level
Trigger. That earlier work found that GCN, GIN and GAT layers on complete graphs give every node the same
embedding ("collapse"). The present study includes a complete-graph GCN as a control for this effect.

## 2. Data

- **LHC Olympics 2020 R&D dataset** (Zenodo 6466204).
  - Background: 1M QCD dijet events.
  - 2-prong signal: W′ (3.5 TeV) → X (500 GeV) Y (100 GeV), X, Y → qq̄; 100k events.
  - 3-prong signal: same masses, X, Y → qqq; 100k events.
  - Generation: Pythia 8 + Delphes.
- **Jets.** Anti-kT, R = 1.0 (fastjet), keeping the two leading jets. Constituents are (pT, η, φ), sorted by pT
  and truncated at 50 (100 in the E3 runs).
- **Splits.** Seeded (20201) and identified by their sha256 hash (in the protocol).
  - Background: 600k train, 100k validation, 300k test.
  - 2-prong signal: 40k train, 10k validation, 50k test.
  - 3-prong signal: 100k, test only.
- **Black boxes** (Zenodo 4536624; Kasieczka et al. 2021, arXiv:2101.08320). Test only. Each box is subsampled to
  200k background events plus every signal event, with a fixed seed.

| box | content | window |
|---|---|---|
| BB1 | 834 signal events: Z′ (3.823 TeV) → X (732 GeV) Y (378 GeV), X, Y → qq̄. Pythia tune, PDF and Delphes are changed. | 3.6–4.0 TeV |
| BB2 | background only, Herwig++ (the null test) | – |
| BB3 | 4.2 TeV KK gluon. Dijet mode: 1200 events, KK gluon → d d̄ (Pythia card `KKg2qq`). Trijet mode: 2000 events, → g + radion (2.217 TeV) → ggg (card `KKg2gr`). | 4.0–4.4 TeV |

The R&D signal region is 3.3–3.7 TeV. The LHCO paper describes the BB3 dijet mode as gluonic, but the Pythia card
generates quarks (protocol D4).

## 3. Method

**Model.** An energy-flow variational autoencoder.
- Encoder: one of seven encoders (below), 3 layers of width 48, then pooling and an MLP [128] to a 4-dimensional
  latent (β = 10⁻³).
- Decoder: an MLP [256, 256] that emits 32 points, each with (Δη, Δφ, pT-fraction logit).
- Loss: an energy-weighted maximum mean discrepancy between the input and output energy flows, with Gaussian
  kernels σ ∈ {0.02, 0.05, 0.1, 0.2, 0.4}.
- Jet score: the reconstruction loss.
- Event score: `prod` = log F_B(s₁) + log F_B(s₂), where F_B is the background CDF of jet scores on the background
  validation split.
- Training: 2 epochs × 120k jets, batch 256, AdamW (lr 10⁻³), cosine schedule, one CPU thread, 5 seeds.

**Encoders (the only thing that changes).**

| | no interaction | local graph | attention |
|---|---|---|---|
| IRC-unsafe (7 node features, mean + max pooling) | Deep Sets | EdgeConv, k-NN k = 8, dynamic | Transformer, 4 heads, no position encoding |
| IRC-safe (directions only, pT-weighted pooling) | EFN-style Deep Sets | EWMP: energy-weighted messages on a radius graph, R = 0.3 | energy-weighted attention: weights ∝ z_j·exp(q·k) |

The control is a GCN on the complete graph (mean over all nodes). Every model is tested to be permutation- and
padding-invariant. The IRC-safe ones are tested to be exactly collinear-safe. k-NN ties are broken by pT, so the
graphs are exactly permutation-invariant.

**Baselines.**
- Single observables: m_J, n_const, −τ21, −τ32. τ is computed from the constituents (protocol D3).
- A normalising flow on (log m_J, log pT, τ21, τ32, log n).
- An EFP k-NN distance (20 prime EFPs, d ≤ 4).
- A Chamfer set autoencoder.
- A supervised EdgeConv classifier as the upper bound: 2-prong AUC 0.982, max SIC 15.7 (`RESULTS.md` §7).

**Metrics.**
- AUC in the signal window: the primary metric.
- Max SIC (with at least 50 background events).
- 1/ε_B at ε_S = 0.3.
- m_jj Jensen–Shannon divergence of the background at ε_B = 1 % (JSD: sculpting).
- Robustness to collinear splits and soft emissions (Spearman ρ of scores before and after).
- Latent probes: R² for regressing jet observables on the latent.

**Statistics.** A contrast A − B is significant if |Δ| > 2 SE over seeds. On BB1 and BB3, a paired event bootstrap
of the 5-seed rank ensembles must also exclude zero. The training-length runs must agree in sign (H4).

**Sessions** (all CPU, about 2 days of cloud compute):

| session | runs |
|---|---|
| E1 | main grid: 7 encoders × 5 seeds, plus the baselines |
| E2 | training length (1, 3, 5 epochs × 3 seeds) and 57 ablations (latent width, pooling, 30 constituents) |
| E3 | training on BB1's own events, bump hunts, 100 constituents |
| F1 | fixed τ baselines, EFP check, black-box diagnostics, reproducibility check |
| E4 | 42 ablation runs scored on the test signals, post hoc |
| E5 | bias-audit follow-up (D9): EFN and EWMP with mean + max pooling (3 seeds each, R&D only), score–mass correlations of the 98 E1 + E2 runs, a look-elsewhere null of the BB1 bump hunt on the R&D background |

## 4. Results (numbers and where they are)

### 4.1 Main comparison

Source: `study/study_table.md` and `study/fix_table.md`. AUC in the window, mean ± sd over 5 seeds.

| encoder | R&D 3-prong | BB1 | BB3 | 3-prong max SIC | BB1 max SIC | JSD |
|---|---|---|---|---|---|---|
| Deep Sets | 0.760 ± 0.038 | 0.916 ± 0.003 | 0.476 ± 0.003 | 1.38 | 3.10 | 0.0141 |
| EdgeConv | 0.800 ± 0.003 | 0.910 ± 0.004 | 0.473 ± 0.003 | 1.44 | 2.88 | 0.0117 |
| Transformer | 0.814 ± 0.004 | 0.899 ± 0.002 | 0.471 ± 0.001 | 1.47 | 2.46 | 0.0076 |
| GCN complete | 0.689 ± 0.017 | 0.921 ± 0.001 | 0.460 ± 0.004 | 1.17 | 3.95 | 0.0086 |
| EFN | 0.812 ± 0.006 | 0.913 ± 0.004 | 0.475 ± 0.001 | 1.50 | 2.88 | 0.0110 |
| EWMP | 0.819 ± 0.004 | 0.909 ± 0.002 | 0.474 ± 0.001 | 1.52 | 2.79 | 0.0099 |
| EW attention | 0.823 ± 0.002 | 0.902 ± 0.003 | 0.469 ± 0.000 | 1.52 | 2.61 | 0.0081 |
| m_J | 0.646 | 0.918 | 0.549 | 1.36 | 3.60 | 0.148 |
| −τ21 (computed τ) | 0.683 | 0.917 | 0.515 | 1.15 | 3.95 | – |
| −τ32 (computed τ) | 0.805 | 0.636 | 0.514 | 1.59 | 1.03 | – |
| jet-observable flow | 0.753 | 0.804 | 0.735 | 1.26 | 1.41 | – |
| EFP k-NN | 0.462 | 0.490 | 0.554 | 1.00 | 1.00 | 0.031 |

The rankings flip between signals. The best encoders on 3-prong are the weakest on BB1, and on BB1 jet mass alone
matches every encoder. Every encoder is below 0.5 on BB3.

### 4.2 Central finding: background fit quality, on the 3-prong signal

Sources: `study/study_posthoc.md`, `study/audit_followup.md` (D9) and `../paper/figures/fig2_auc_vs_fit.pdf`.

Across 98 runs (E1 + E2: seven encoders, 1–5 epochs), the 3-prong AUC is close to a single function of the
background validation loss, whatever the encoder.
- The 3-prong AUC falls from about 0.83 to 0.67 as the loss rises from 0.13 to 0.6; the BB1 AUC rises from about
  0.89 to 0.92 over the same range.
- Training longer moves every encoder along the curve: 3-prong goes up and BB1 goes down (`study/stct_table.md`).

**How much it explains** (D9). Predicting each run while leaving it out, the quadratic in the loss explains 95 % of
the variance of the 3-prong AUC between the 98 runs, more than the mean of the other seeds of its configuration
(encoder × training length: 83 %). On the black boxes the configuration predicts better than the curve (BB1 89 % vs
81 %; BB3 83 % vs 61 %), and the curve predicts the 42 E4
ablation runs poorly there (out-of-sample R² 0.23 on BB1 and −0.85 on BB3, against 0.86 on 3-prong). The black-box
residuals of about ±0.005 are small only because the black-box AUCs barely vary (sd 0.010 and 0.006 between runs).

**Fixed-architecture evidence** (D9). Seeds change the fit at fixed architecture and training length. Within each
configuration, better-fitting seeds detect the 3-prong signal better: Spearman ρ(loss, AUC) = −0.71 (p = 2 × 10⁻¹⁶;
−0.59 without Deep Sets). On BB1 the within-configuration relation is weak (+0.29).

**Contrasts at equal fit.** Residuals of the curve (post hoc, D2 and D9). The 95 % intervals use the larger of the
Welch SE and a bootstrap SE that refits the curve; † = significant for all five curve forms tried.

| contrast | 3-prong, raw → equal fit [95 %] | BB1, raw → equal fit [95 %] |
|---|---|---|
| EdgeConv − Deep Sets | +0.040* → +0.008 [−0.007, +0.022] | −0.006* → +0.002 [−0.001, +0.005] |
| EWMP − EFN | +0.007* → +0.002 [−0.002, +0.007] | −0.004* → −0.002 [−0.005, +0.001] |
| Transformer − Deep Sets | +0.054* → +0.000 [−0.014, +0.015] | −0.017* → −0.001 [−0.003, +0.001] |
| EW attention − EFN | +0.012* → −0.012 [−0.017, −0.007] † | −0.011* → −0.001 [−0.005, +0.003] |
| EWMP − EdgeConv | +0.019* → +0.011 [+0.007, +0.016] † | −0.002 → +0.001 [−0.002, +0.004] |
| EFN − Deep Sets | +0.051* → +0.017 [+0.002, +0.031] † | −0.003 → +0.006 [+0.003, +0.008] † |
| EW attention − Transformer | +0.009* → +0.004 [−0.001, +0.008] | +0.003 → +0.006 [+0.002, +0.009] † |

\* significant under the protocol rule.

No interaction contrast is significantly positive at equal fit; the intervals still allow up to +0.022 on 3-prong.
The encoders compared never overlap in validation loss, so "equal fit" is read off the pooled curve.

**What the IRC-safe design's edge is made of** (E5, D9; `study/audit_followup.md` section 3). The IRC-safe encoders
differ from their partners in their inputs (directions only) and their pooling (pT-weighted). In the set family,
EFN with mean + max pooling fits much worse (loss 0.41 vs 0.235; raw 3-prong AUC 0.755 vs 0.812) but has the same
equal-fit residual as EFN (+0.012 vs +0.011): the pooling acts through the fit, and the equal-fit edge comes from the
direction-only inputs (+0.018 ± 0.007 with mean + max pooling; +0.018 to +0.025 for every curve form). That edge does
not need IRC safety: the mean + max EFN is not IRC safe. Graph family: ⟨…⟩

### 4.3 Pre-registered verdicts

Source: `study/study_posthoc.md`. The checks added after E1 are paired seeds, a Holm correction, dropping the
badly fitted runs, and the sign at 1, 3 and 5 epochs.

| hypothesis | verdict | robustness |
|---|---|---|
| H1: interactions help on untuned signals | inconclusive | positive on 3-prong, significantly negative on BB1 |
| H2a: IRC safety generalises | inconclusive | 3-prong only, and 3-prong is not blind for H2 (D5) |
| H2b: IRC safety sculpts less | supported by the frozen rule | fails with paired seeds and Holm; one contrast flips at 5 epochs |
| H3: complete-graph GCN collapses | supported | holds under every check |
| H5: width and constituents | descriptive | latent 4–16 equivalent, 2 and 32 worse; 30 constituents slightly worse; 100 change nothing important |

In the H3 latent, the MLP probe recovers τ21 with R² 0.24, against 0.50–0.61 for the other encoders
(`study/study_table.md`, latent probes).

### 4.4 IRC robustness

Source: `study/study_table.md` (IRC probe) and `study/audit_followup.md` section 4. Spearman ρ of the background
scores before and after each perturbation, mean over 5 seeds.

| encoder | split every constituent | 10 soft emissions of 0.2 % pT | 10 soft emissions of 1 % pT |
|---|---|---|---|
| Deep Sets | 0.96 | 0.982 | 0.73 ± 0.08 |
| EFN (safe) | 1.000 | 0.988 | 0.75 ± 0.16 |
| EdgeConv | 0.92 | 0.968 | 0.53 |
| EWMP (safe) | 1.000 | 0.993 | 0.79 |
| Transformer | 0.95 | 0.978 | 0.74 |
| EW attention (safe) | 1.000 | 0.999 | 0.98 |
| GCN complete | 0.83 | 0.867 | 0.47 |

The IRC-safe encoders are exactly collinear safe. Under soft emissions, only the two with an energy-weighted
interaction (EWMP, EW attention) are significantly more robust than their partners; EFN is not more robust than
Deep Sets (difference +0.005 ± 0.007 at 0.2 %, +0.016 ± 0.080 at 1 %). Ten emissions of 1 % add 10 % of the jet pT,
far from the soft limit.

### 4.5 Black boxes

**BB1** (`study/bb1_diagnostics.md`, `study/e3_summary.md`, `study/study_bb1data_table.md`):
- The encoders reach 0.90–0.92, matched by jet mass and by −τ21.
- Encoder scores correlate with m_J (ρ ≈ 0.7) on both R&D and BB1.
- Trained on BB1's own events (label-blind, 3 seeds), the AUC is 0.904–0.908 including the training events and
  0.894–0.899 on the held-out validation part (`study/e3_summary.md`).
- The bump hunt with a sideband-defined cut at the known window reaches a quoted local Z of 2.3–2.5 (10 % cut) and
  3.2–3.8 (2 % cut; per seed 1.6–4.4). Calibrated against the no-signal null (D9): ⟨…⟩
- BB2 test: the R&D-trained runs give a largest |Z| of 3.3 over 360 trials. They were not trained on BB2, so this
  does not test the train-on-data procedure behind the BB1 numbers.

**BB3** (`study/bb3_diagnostics.md`):
- The box background is not shifted relative to R&D: KS ≈ 0.02, and 10–11 % of jets lie above the R&D 90 %
  quantile.
- The dijet (quark) jets have fewer constituents than the background (n_const jet AUC 0.44), and the encoders score
  them as ordinary (jet AUC 0.44; mean over seed 1 of the seven encoders).
- The trijet mode is detected (jet AUC ≈ 0.70), but only 137 of its 2000 events fall in the window.
- The jet-observable flow reaches 0.735 through jet pT (jet AUC of pT 0.76).

### 4.6 Reproducibility

- Rescoring seed 1 of every encoder from the committed weights reproduces all 288 committed metrics of each run
  exactly (F1).
- No main-grid or curve run was flagged by the quality rule. Two of 57 ablation runs were flagged; they stay in the
  means.

## 5. Development history (appendix material; `RESULTS.md`)

All of this was done on the 2-prong validation split before the protocol; 3-prong test numbers were seen.

- **Collapse of complete-graph GNNs**, inherited from the AIMS code: GCN and GIN embeddings are identical across
  nodes (`RESULTS.md` §3).
- **Inverted constituent reconstruction.** The Chamfer set AE scores signal *as more normal*: AUC 0.488
  (`RESULTS.md`, set AE rows).
- **Choice of the recipe.** The energy-flow VAE's recipe (2 epochs, k = 8, latent 4, kernels, `prod`) was tuned on
  the 2-prong validation split. The final development model reached 0.853 (2-prong) and 0.799 (3-prong).
- **Unsupervised vs supervised.** The unsupervised model recovers about two thirds of the supervised AUC excess on
  2-prong, but only about 4 % of its max-SIC excess.

## 6. Protocol deviations (`docs/STUDY_PROTOCOL.md`)

| | deviation |
|---|---|
| D1 | `promote` refuses runs paused at the time budget; no number affected |
| D2 | post-hoc analyses: paired seeds, Holm, low-fit exclusion, equal-fit residuals; "contradicted" defined as the mirror rule |
| D3 | N-subjettiness computed from the constituents for every sample (the black boxes had none); baselines rerun as `fix_*` |
| D4 | black-box diagnostics; the BB3 dijet mode is quarks per the Pythia card |
| D5 | disclosure: the IRC-safety hypothesis was formed after development results on 3-prong had been seen |
| D6 | E1 session conditions (environment, one restart, a repeated download); no number affected |
| D7 | EFP k-NN checked: no bug, just a weak event-level baseline |
| D8 | E4: ablation runs scored on the test signals after the results were seen, for reporting only |
| D9 | bias-audit follow-up: curve diagnostics, robust equal-fit contrasts, the inputs × pooling runs (E5), the mass check, the BB1 look-elsewhere null; black-box parts left for session F2 (`docs/F2_SESSION.md`) |

## 7. What the paper may claim, and what it may not

Revised after the bias audit (D9). Numbers in section 4.

**Supported:**
- On the R&D 3-prong signal, how well an energy-flow autoencoder fits the background explains most of the
  differences between the seven encoders: 95 % of the AUC variance between 98 runs (leave one out; 83 % for knowing
  the configuration), across encoders, training lengths
  and seeds at fixed architecture (within-configuration ρ = −0.71). It does not explain all of them: width and
  pooling cost detection at equal fit.
- The direction of the fit effect depends on the signal: a better fit helps 3-prong and hurts BB1.
- At equal fit, message passing and attention give no detectable gain. Quote the limits: up to +0.022 on 3-prong
  (EdgeConv over Deep Sets), ±0.005 on BB1. Energy-weighted attention is even below EFN on 3-prong.
- IRC-safe encoders are exactly robust to collinear splits. The two with an energy-weighted interaction (EWMP,
  energy-weighted attention) are much more robust to soft emissions than their partners; EFN is not.
- The IRC-safe design keeps a small equal-fit edge on some signals. In the set family that edge comes from the
  direction-only inputs, not from IRC safety or the pooling. ⟨graph family⟩
- The complete-graph GCN collapses (H3).
- On BB1 the autoencoders detect the signal about as well as jet mass alone; they miss the quark-dijet mode of BB3.
  This is not a generator shift.

**Not supported (do not write):**
- "Graphs (or attention) improve unsupervised anomaly detection."
- "IRC safety improves generalisation" or "IRC safety improves detection": H2a is inconclusive, and in the set family
  the equal-fit edge comes from the inputs.
- "IRC-safe encoders are more robust to soft radiation" as a general statement: EFN is not.
- "Fit quality explains the black-box results", or "the curve holds out of sample on BB1/BB3": it predicts the
  black-box AUCs of new runs poorly (R² 0.23 and −0.85).
- "Fit quality causes the performance" or "architecture acts only through the fit": the seed test varies the fit
  only by chance, and the equal-fit comparison conditions on a mediator.
- "A poorly fitting model acts as a mass tagger": ⟨mass check⟩
- Any significance for BB1 beyond what section 4.5 gives after calibration and look-elsewhere correction; nothing
  discovery-level.

## 8. Framing options

| | framing | headline |
|---|---|---|
| **A (recommended, narrowed)** | "Fit quality, not architecture, on the signal where encoders differ" | Lead with the curve on 3-prong (95 % of the variance, seed test at fixed architecture, no detectable interaction gain at equal fit), then show that on the black boxes all encoders sit within 0.03 AUC of a jet-mass tagger. Strongest additional evidence: vary the fit at fixed architecture by design (learning rate, training data size), and the black-box checks of `docs/F2_SESSION.md`. |
| B | "A controlled graph-vs-set study" | Report the pre-registered verdicts as they are. Mostly inconclusive. |
| C | "Black-box stress test of unsupervised autoencoders" | BB1–BB3 results; narrower, partly known. |

The author has not yet chosen a framing.

## 9. Limitations to state

- The recipe was tuned for EdgeConv on 2-prong. Under the fit-quality reading this matters through which encoders
  the recipe lets fit well; the training-length curves show every encoder moving along the same curve, which limits
  but does not remove the concern.
- Fit quality is a post-hoc summary, not an intervention; the encoders compared at equal fit never overlap in loss.
- The curve explains the 3-prong signal, not the black boxes.
- The IRC-safety contrasts test a design (inputs + pooling + energy weighting), not IRC safety alone.
- There are few signal events in the windows: 512 for BB1, 137 for the BB3 trijet mode. The BB1 bump hunt's quoted
  local Z is not calibrated and ignores the look-elsewhere effect (section 4.5).
- All signals are Pythia; BB2 (Herwig) is the only generator change on the background side.
- H4 rests on 3 seeds per training length; E2–E5 have 3 seeds per configuration.
- The analysis is jet-level with a two-jet event score; there is no event-level embedding.
- There is no comparison with weakly supervised methods, which reach max SIC 15–50 on LHCO.
- Black-box checks not yet run: E5 on BB1/BB3, blind BB1 scan, BB2-trained null, BB1 at fixed jet masses
  (`docs/F2_SESSION.md`).

## 10. Key references

Verify each before citing.

**Data.**
- LHC Olympics 2020: Kasieczka et al., arXiv:2101.08320.
- R&D dataset: Zenodo 6466204.
- Black boxes: Zenodo 4536624.

**Closest prior work.**
- IRC-safe graph autoencoder: Atkinson et al., arXiv:2204.12231.
- Subjet-graph autoencoder on LHCO: Araz et al., arXiv:2506.19920.
- Optimal-transport embeddings: Cai, Bhargava & Nachman, arXiv:2512.04839.
- Pretrained jet foundation models: OmniLearn, arXiv:2502.14652; OmniLearned, arXiv:2510.24066.

**Weakly supervised and density methods (context for the regime).** Look up their arXiv numbers on INSPIRE.
- CATHODE
- CWoLa hunting
- ANODE
- the "Back To The Roots" boosted-decision-tree weak-supervision study

**Architectures and theory.**
- Energy Flow Networks: Komiske, Metodiev & Thaler, arXiv:1810.05165.
- Energy flow polynomials: arXiv:1712.07124.
- Deep Sets: arXiv:1703.06114.
- DGCNN / EdgeConv: arXiv:1801.07829.
- ParticleNet: arXiv:1902.08570.
- Set Transformer (PMA pooling): arXiv:1810.00825.
- Limits of sum decomposition: Wagstaff et al., arXiv:1901.09006.
- N-subjettiness: Thaler & Van Tilburg, arXiv:1011.2268.

## 11. Files in this folder

| file | contents |
|---|---|
| `study/study_table.md` / `.csv` | E1 main grid: all metrics, probes, IRC probe, flagged runs |
| `study/study_contrasts.json` | every pairwise contrast with SE and the black-box bootstrap |
| `study/study_posthoc.md` / `.json` | verdicts, paired / Holm / low-fit / equal-fit checks, H4 signs, E4 residuals |
| `../paper/figures/fig1–fig6 .pdf/.png` | **the paper figures**: journal style, final print size, vector PDF; draft captions and the colour/marker encoding in `../paper/figures/CAPTIONS.md` |
| `study/figures/study_auc_vs_valloss.png` | diagnostic version of Fig. 2 from the analysis code (not for the paper) |
| `study/figures/overview/*.png` | presentation versions of the figures, for the results deck (not for the paper) |
| `study/stct_table.md`, `curves.png`, `ablations.md` | E2 training length and ablations |
| `study/study_n100_table.md`, `study_bb1data_table.md`, `e3_summary.md` | E3: 100 constituents, BB1 train-on-data, bump hunts, BB2 null |
| `study/fix_table.md` | F1 baselines with computed τ. These replace the `obs_tau*` and `hldensity` rows of `study_table.md`, which used τ = 0 on the black boxes (D3). |
| `study/bb1_diagnostics.md`, `bb3_diagnostics.md` | F1 black-box diagnostics |
| `runs/<run>/metrics.json`, `bb*_metrics.json`, `history.json`, `config.yaml` | per-run results; `study_*` = E1, `stct_*` = E2, `stat_*` = E4, `fix_*` = F1 |
| `study/audit_followup.md` / `.json` | **bias-audit follow-up (D9)**: what the curve explains, equal-fit robustness, inputs × pooling, IRC contrasts, mass check, BB1 look-elsewhere null |
| `runs/e5_*`, `runs/*/mass_corr.json`, `runs/study_efn_s*/rnd_scan.json` | E5 runs and the D9 scoring outputs |
| `../docs/F2_SESSION.md` | the black-box checks still to run (needs zenodo access) |
| `../docs/STUDY_PROTOCOL.md` | the frozen protocol and deviations D1–D9 |
| `../paper/results_draft.md` | the results section, already drafted |
| `../RESULTS.md` | development history before the protocol |
