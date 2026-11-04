# GAE_HEP: context for writing the paper

This file holds everything a writer (human or model) needs to draft the paper from this results folder.

**Facts and references.**
- Every number below comes from a file in `results/`; the file is named next to it.
- Before citing a reference, verify it on arXiv. The arXiv IDs below were checked on 31 Oct 2026, but titles and author lists were not re-checked when writing this file.
- Repository: github.com/raktim711/GAE_HEP (code in `src/gaehep`).
- Frozen protocol: `docs/STUDY_PROTOCOL.md`, including the deviations D1–D8.
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

### 4.2 Central finding: background fit quality

Sources: `study/study_posthoc.md` and `../paper/figures/fig2_auc_vs_fit.pdf`.

Across 98 runs (E1 + E2: seven encoders, 1–5 epochs), AUC is close to a single function of the background
validation loss, whatever the encoder.
- The 3-prong AUC falls from about 0.83 to 0.67 as the loss rises from 0.13 to 0.6.
- The BB1 AUC rises from about 0.89 to 0.92 over the same range.
- A quadratic in the loss leaves a residual spread of 0.011 (3-prong), 0.0045 (BB1) and 0.0034 (BB3).

Training longer moves every encoder along the curve: 3-prong goes up and BB1 goes down (`study/stct_table.md`).
The two Deep Sets seeds with a poor fit (loss 0.42 against 0.29) are the ones with a low 3-prong AUC (0.72 against
0.79).

**Contrasts at equal fit.** These are the residuals of that curve (post hoc, protocol D2):

| contrast | 3-prong, raw → equal fit | BB1, raw → equal fit |
|---|---|---|
| EdgeConv − Deep Sets | +0.040* → +0.008 | −0.006* → +0.002 |
| EWMP − EFN | +0.007* → +0.002 | −0.004* → −0.002 |
| EWMP − EdgeConv | +0.019* → +0.011* | −0.002 → +0.001 |
| EFN − Deep Sets | +0.051* → +0.017* | −0.003 → +0.006* |
| EW attention − Transformer | +0.009* → +0.004 | +0.003 → +0.006* |

\* significant.

At equal fit the interaction contrasts vanish, and a small IRC-safety advantage remains. This comparison conditions
on a variable the encoder itself influences, so it explains the result but does not replace the pre-registered
verdicts.

**Out-of-sample test (E4, D8).** The 42 ablation runs were not used to fit the curve:

| signal | mean residual | within 2 sd of the curve |
|---|---|---|
| BB1 | −0.005 | 90 % |
| BB3 | +0.002 | 88 % |
| 3-prong | −0.011 | 60 % |

On 3-prong, about half of the latent-width and pooling variants fall below the curve; the 30-constituent runs fall
on it. Conclusion: fit quality explains most, but not all, of the differences.

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

Source: `study/study_table.md`, IRC probe. The table gives the Spearman ρ of the background scores before and after
each perturbation.

| encoder | split every constituent | 10 soft emissions of 1 % pT |
|---|---|---|
| EdgeConv | 0.92 | 0.53 |
| GCN complete | 0.83 | 0.47 |
| Deep Sets | 0.96 | 0.73 |
| EWMP | 1.000 | 0.79 |
| EFN | 1.000 | 0.75 |
| EW attention | 1.000 | 0.98 |

### 4.5 Black boxes

**BB1** (`study/bb1_diagnostics.md`, `study/e3_summary.md`, `study/study_bb1data_table.md`):
- The encoders reach 0.90–0.92, matched by jet mass and by −τ21.
- Encoder scores correlate with m_J (ρ ≈ 0.7) on both R&D and BB1.
- Trained on BB1's own events (label-blind, 3 seeds), the AUC is 0.904–0.908.
- The bump hunt with a sideband-defined cut reaches local Z 2.3–2.5 (10 % cut) and 3.2–3.8 (2 % cut). There is no
  look-elsewhere correction, so these are optimistic.
- BB2 null test: the largest |Z| is 3.3 over 360 trials, consistent with chance.

**BB3** (`study/bb3_diagnostics.md`):
- The box background is not shifted relative to R&D: KS ≈ 0.02, and 10–11 % of jets lie above the R&D 90 %
  quantile.
- The dijet (quark) jets have fewer constituents than the background (n_const jet AUC 0.44), and the encoders score
  them as ordinary (jet AUC 0.44).
- The trijet mode is found (jet AUC ≈ 0.70), but only 137 of its 2000 events fall in the window.
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

## 7. What the paper may claim, and what it may not

**Supported:**
- In permutation-invariant energy-flow autoencoders, how well the model fits the background explains most of the
  differences between encoders. On BB1 and BB3 this holds out of sample to about ±0.005 AUC.
- The direction of the effect depends on the signal: a better fit helps 3-prong and hurts BB1.
- Message passing and attention give no measurable gain at equal fit.
- IRC-safe encoders are exactly robust to collinear splits, much more robust to soft radiation (energy-weighted
  attention most of all), and keep a small detection advantage at equal fit on some signals.
- Complete-graph GCN collapses (H3).
- Unsupervised autoencoders behave close to mass taggers on BB1 and miss the quark-dijet mode of BB3. This is not a
  generator shift.

**Not supported (do not write):**
- "Graphs improve anomaly detection."
- "IRC safety improves generalisation" (H2a is inconclusive).
- "Fit quality fully determines performance": on 3-prong, width and pooling matter at equal fit.
- "Fit quality causes performance": the study has no intervention on fit at fixed architecture.
- Any discovery-level significance.

## 8. Framing options

| | framing | headline |
|---|---|---|
| **A (recommended)** | "Fit quality, not architecture" | Lead with the curve; the controlled encoder study is the evidence. Needs a causal test (vary fit at fixed architecture, e.g. learning rate or data size) and a mass-decorrelated check of the BB1 trend. |
| B | "A controlled graph-vs-set study" | Report the pre-registered verdicts as they are. Mostly inconclusive. |
| C | "Black-box stress test of unsupervised autoencoders" | BB1–BB3 results; narrower, partly known. |

The author has not yet chosen a framing.

## 9. Limitations to state

- The recipe was tuned for EdgeConv on 2-prong (the training-length curves partly address this).
- There are few signal events in the windows: 512 for BB1, 137 for the BB3 trijet mode.
- All signals are Pythia; BB2 (Herwig) is the only generator change on the background side.
- The equal-fit analysis is post hoc and conditions on a mediator.
- H4 rests on 3 seeds per training length.
- The analysis is jet-level with a two-jet event score; there is no event-level embedding.
- There is no comparison with weakly supervised methods, which reach max SIC 15–50 on LHCO.

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
| `../docs/STUDY_PROTOCOL.md` | the frozen protocol and deviations D1–D8 |
| `../paper/results_draft.md` | the results section, already drafted |
| `../RESULTS.md` | development history before the protocol |
