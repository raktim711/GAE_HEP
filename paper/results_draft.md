# Results (draft)

Draft of the results section. Every number comes from the committed tables: `results/study/study_table.md`,
`stct_table.md`, `study_posthoc.md`, `fix_table.md`, `study_n100_table.md`, `study_bb1data_table.md`,
`e3_summary.md`, `bb1_diagnostics.md`, `bb3_diagnostics.md`, `ablations.md`, for E4 the ablation section of
`study_posthoc.md`, and for the bias-audit follow-up `audit_followup.md` (D9). Unless stated otherwise:

- the event score is `prod`;
- each value is a mean ± sample standard deviation over 5 seeds (E1) or 3 seeds (E2–E5);
- AUC is quoted in the signal window: 3.3–3.7 TeV for R&D, 3.6–4.0 TeV for BB1, 4.0–4.4 TeV for BB3.

The R&D 2-prong signal is the one the recipe was tuned on, and it is never used as evidence. Deviations from the
protocol are D1–D9 in `docs/STUDY_PROTOCOL.md`; D9 collects the analyses added after the bias audit of 6 November.

## 1. Main comparison

Table 1 compares the seven encoders, which share decoder, loss, latent size and training recipe, with the
baselines.

| encoder / baseline | interaction | IRC-safe | R&D 3-prong AUC | BB1 AUC | BB3 AUC | R&D 3-prong max SIC | BB1 max SIC | JSD (ε_B = 1 %) |
|---|---|---|---|---|---|---|---|---|
| Deep Sets | none | no | 0.760 ± 0.038 | 0.916 ± 0.003 | 0.476 ± 0.003 | 1.38 ± 0.06 | 3.10 ± 0.06 | 0.0141 ± 0.0021 |
| EdgeConv (k-NN) | local graph | no | 0.800 ± 0.003 | 0.910 ± 0.004 | 0.473 ± 0.003 | 1.44 ± 0.02 | 2.88 ± 0.12 | 0.0117 ± 0.0008 |
| Transformer | attention | no | 0.814 ± 0.004 | 0.899 ± 0.002 | 0.471 ± 0.001 | 1.47 ± 0.01 | 2.46 ± 0.04 | 0.0076 ± 0.0008 |
| GCN, complete graph | mean over the jet | no | 0.689 ± 0.017 | 0.921 ± 0.001 | 0.460 ± 0.004 | 1.17 ± 0.01 | 3.95 ± 0.09 | 0.0086 ± 0.0008 |
| EFN-style Deep Sets | none | yes | 0.812 ± 0.006 | 0.913 ± 0.004 | 0.475 ± 0.001 | 1.50 ± 0.02 | 2.88 ± 0.20 | 0.0110 ± 0.0011 |
| EWMP (radius graph) | local graph | yes | 0.819 ± 0.004 | 0.909 ± 0.002 | 0.474 ± 0.001 | 1.52 ± 0.01 | 2.79 ± 0.11 | 0.0099 ± 0.0017 |
| energy-weighted attention | attention | yes | 0.823 ± 0.002 | 0.902 ± 0.003 | 0.469 ± 0.000 | 1.52 ± 0.01 | 2.61 ± 0.09 | 0.0081 ± 0.0009 |
| m_J | – | – | 0.646 | 0.918 | 0.549 | 1.36 | 3.60 | 0.1475 |
| −τ21 (computed τ, D3) | – | – | 0.683 | 0.917 | 0.515 | 1.15 | 3.95 | – |
| −τ32 (computed τ) | – | – | 0.805 | 0.636 | 0.514 | 1.59 | 1.03 | – |
| flow on jet observables (computed τ) | – | – | 0.753 ± 0.001 | 0.804 ± 0.001 | 0.735 ± 0.000 | 1.26 ± 0.00 | 1.41 ± 0.00 | – |
| EFP k-NN | – | – | 0.462 | 0.490 | 0.554 | 1.00 | 1.00 | 0.0312 |

**The three signals tell different stories.**

- **R&D 3-prong.** The interaction and IRC-safe encoders reach 0.80–0.82. Deep Sets and the complete-graph GCN
  are clearly worse. No single observable or baseline comes close, except −τ32 at 0.805.
- **BB1.** BB1 (Z′ → XY with heavy X, Y) is detected by every encoder at AUC 0.90–0.92. Jet mass alone (0.918) and
  −τ21 alone (0.917) do just as well, and the weakest encoders on the 3-prong signal (GCN, Deep Sets) are the
  strongest here.
- **BB3.** Every encoder is below 0.5 on BB3 (section 6).

The EFP k-NN baseline reproduces exactly when rerun and has no bug. It is weak at event level, because signal
events rarely have two anomalous jets (D7).

## 2. Background fit explains most of the 3-prong differences

Figure 2 (`paper/figures/fig2_auc_vs_fit.pdf`) plots each run's AUC against its background validation loss. It
covers all 98 autoencoder runs of E1 and E2 (seven encoders at 1, 2, 3 and 5 epochs), plus the 42 E4 ablation runs
as crosses.

- **Opposite directions.** On the R&D 3-prong signal the AUC rises as the background fit improves: from about
  0.67 at loss 0.6 to 0.82–0.83 below loss 0.16. On BB1 it does the opposite: the worst-fitting runs are the best
  detectors (0.92 at loss ≥ 0.4, against 0.89 at loss 0.13).
- **Training length moves each encoder along the same curve.** EdgeConv, for example, goes from 0.776 to 0.808 on
  the 3-prong signal and from 0.919 to 0.894 on BB1 between 1 and 5 epochs.

**How much the curve explains** (Table 2; post hoc, D9). Each run's AUC is predicted in two ways, leaving the run
out each time: by a quadratic in the validation loss fitted to the other runs, and by the mean of the other seeds of
its configuration (encoder × training length). On the 3-prong signal the curve explains 95 % of the variance of the
AUC between the 98 runs, the configuration 83 %. So on this signal the background fit predicts a run better than
knowing its encoder and training length. The black boxes are different: there the configuration predicts better
than the curve.

| signal | AUC sd between runs | R² of the curve | R² of the configuration | within-configuration ρ(loss, AUC) | E4 runs out of sample: R² (RMSE) |
|---|---|---|---|---|---|
| R&D 3-prong | 0.052 | 0.95 | 0.83 | −0.71 (p = 2 × 10⁻¹⁶) | 0.86 (0.021) |
| BB1 | 0.010 | 0.81 | 0.89 | +0.29 (p = 0.003) | 0.23 (0.006) |
| BB3 | 0.006 | 0.61 | 0.83 | +0.24 (p = 0.02) | −0.85 (0.005) |

R² values are leave-one-out; the in-sample values are 0.96 / 0.91 (3-prong), 0.82 / 0.94 (BB1) and 0.65 / 0.91 (BB3).

**A test at fixed architecture.** Seeds change the fit without changing the architecture or the training length.
Within each configuration, the runs that fit better detect the 3-prong signal better: Spearman ρ = −0.71 between
loss and AUC after removing every configuration's mean (98 runs, 28 configurations). The relation holds without
Deep Sets (ρ = −0.59, p = 4 × 10⁻⁹), whose two badly fitting E1 seeds are the clearest case (AUC 0.72 against
0.79 at loss 0.42 against 0.29). This is the closest the study comes to varying the fit at fixed architecture. It is
not a randomised intervention: a seed that fits worse may also differ in other ways. On BB1 the within-configuration
relation is weak (ρ = +0.29).

**Out of sample** (E4, D8). The 42 Deep Sets and EdgeConv ablation runs (latent width 2–32, pooling, 30
constituents) were not used to fit the curve.
- **On the 3-prong signal** the curve predicts them with R² 0.86. They nevertheless fall 0.011 below it on average,
  and about half of the width and pooling variants lie more than two residual sd below.
- **On the black boxes** their AUCs spread by only 0.007 (BB1) and 0.003 (BB3), and the curve predicts that spread
  poorly: R² 0.23 on BB1, and −0.85 on BB3, which is worse than predicting their mean. Their residuals of about
  ±0.005 are small only because the range is small.

**Conclusion.** Fit quality explains most of the differences between models on the 3-prong signal: across
encoders, training lengths and seeds. It does not explain all of them, since width and pooling cost detection at
equal fit. Nor does it explain the small differences on the black boxes, which depend on the encoder beyond its fit.

### 2.1 Contrasts at equal fit

Table 3 repeats the pre-registered contrasts on the residuals of the curve, that is, at equal fit (post hoc, D2 and
D9). Because the fit quality is itself a property of the encoder, this is a conditional comparison, not a
replacement for Table 1. The 95 % intervals use the larger of two standard errors: the Welch SE, and a bootstrap SE
that refits the curve in every replica.

The encoders compared never overlap in validation loss: EdgeConv spans 0.235–0.256 against 0.288–0.418 for Deep
Sets. "Equal fit" is therefore read off the pooled curve. † marks the contrasts that stay significant for every
curve tried: linear, quadratic, cubic, quadratic in log(loss), and quadratic without the GCN runs.

| contrast (A − B) | R&D 3-prong: raw → at equal fit [95 % interval] | BB1: raw → at equal fit [95 % interval] |
|---|---|---|
| EdgeConv − Deep Sets | +0.040* → +0.008 [−0.007, +0.022] | −0.006* → +0.002 [−0.001, +0.005] |
| EWMP − EFN | +0.007* → +0.002 [−0.002, +0.007] | −0.004* → −0.002 [−0.005, +0.001] |
| Transformer − Deep Sets | +0.054* → +0.000 [−0.014, +0.015] | −0.017* → −0.001 [−0.003, +0.001] |
| e.w. attention − EFN | +0.012* → **−0.012** [−0.017, −0.007] † | −0.011* → −0.001 [−0.005, +0.003] |
| EWMP − EdgeConv | +0.019* → **+0.011** [+0.007, +0.016] † | −0.002 → +0.001 [−0.002, +0.004] |
| EFN − Deep Sets | +0.051* → **+0.017** [+0.002, +0.031] † | −0.003 → **+0.006** [+0.003, +0.008] † |
| e.w. attention − Transformer | +0.009* → +0.004 [−0.001, +0.008] | +0.003 → **+0.006** [+0.002, +0.009] † |
| GCN complete − Deep Sets | −0.071* → +0.014 [+0.000, +0.029] | +0.005 → +0.001 [−0.001, +0.003] |

\* significant under the protocol rule. Bold: the 95 % interval excludes zero.

**At equal fit, message passing and attention give no detectable gain.** No interaction contrast is significantly
positive on any signal. These are limits, not zeros:
- **On the 3-prong signal,** the intervals still allow a gain of up to 0.022 (EdgeConv over Deep Sets), 0.007 (EWMP
  over EFN) and 0.015 (Transformer over Deep Sets). Energy-weighted attention is even below EFN.
- **On BB1,** every interaction contrast lies within ±0.005.

Conditioning on a fit that the encoder itself drives does not show that the architecture acts only through the fit.

### 2.2 What the IRC-safe design's edge is made of (E5, D9)

The IRC-safe encoders differ from their unsafe partners in two ways at once:
- **inputs:** they read only the constituent directions (Δη, Δφ, ΔR), instead of the seven standard features,
  which include the constituents' log pT and log E;
- **pooling:** they pool with pT weights, instead of mean + max.

IRC safety needs both. E4 already had the unsafe inputs with weighted pooling; E5 adds the safe inputs with mean +
max pooling: EFN and EWMP with mean + max pooling, 3 seeds each. Changing only the pooling keeps every parameter
shape, so a seed starts from the same weights as the native encoder. Neither mixed cell is IRC safe: its score
changes when the five hardest constituents are split collinearly (ρ = 0.966–0.999, against exactly 1 for the
IRC-safe encoders).

Table 4 gives the 3-prong AUC and the residual from the curve of section 2. The E5 runs were scored on R&D only;
the black boxes were not reachable from the session (`docs/F2_SESSION.md`).

| family | inputs / pooling | runs | loss | AUC | residual at equal fit |
|---|---|---|---|---|---|
| set | unsafe / mean + max (Deep Sets) | 5 | 0.342 | 0.760 ± 0.038 | −0.006 ± 0.007 |
| set | unsafe / weighted (E4) | 3 | 0.379 | 0.736 ± 0.074 | −0.016 ± 0.018 |
| set | safe / mean + max (E5) | 3 | 0.410 | 0.755 ± 0.008 | +0.012 ± 0.002 |
| set | safe / weighted (EFN) | 5 | 0.235 | 0.812 ± 0.006 | +0.011 ± 0.002 |
| graph | unsafe / mean + max (EdgeConv) | 5 | 0.244 | 0.800 ± 0.003 | +0.002 ± 0.002 |
| graph | unsafe / weighted (E4) | 3 | 0.206 | 0.819 ± 0.005 | +0.009 ± 0.002 |
| graph | safe / mean + max (E5) | 3 | 0.282 | 0.811 ± 0.002 | **+0.025 ± 0.001** |
| graph | safe / weighted (EWMP) | 5 | 0.219 | 0.819 ± 0.004 | +0.013 ± 0.001 |

**Set family.** EFN with mean + max pooling fits much worse than EFN (loss 0.41 against 0.235), and its raw 3-prong
AUC drops from 0.812 to 0.755. At equal fit, however, it is exactly as good as EFN (+0.012 against +0.011).
- **Pooling** acts through the fit: at equal fit, weighted pooling adds nothing with direction-only inputs
  (−0.001 ± 0.003; between −0.014 and −0.002 for the other curve forms).
- **The inputs** carry the edge: with mean + max pooling, direction-only inputs give +0.018 ± 0.007 (+0.018 to
  +0.025 for every curve form).

So in the set family the equal-fit edge of the IRC-safe encoder comes from not giving the network the constituents'
energies, and it does not need IRC safety.

**Graph family.** EWMP with mean + max pooling also fits worse than EWMP (loss 0.282 against 0.219, raw AUC 0.811
against 0.819). At equal fit, however, it has the largest residual of all eight cells: +0.025 against +0.013 for
EWMP.
- **The encoder side carries the edge.** With mean + max pooling, EWMP beats EdgeConv by +0.023 ± 0.002 at equal fit
  (+0.023 to +0.024 for every curve form). In this family that factor also includes EWMP's energy-weighted radius
  graph, not only the inputs.
- **Pooling acts both ways.** pT-weighted pooling helps EdgeConv at equal fit (+0.007 ± 0.003) and hurts EWMP
  (−0.012 ± 0.001), in every curve form. Its raw benefit for EWMP comes through the fit.

In both families, then, the equal-fit edge of the IRC-safe design comes from the encoder side (direction-only
inputs), not from the pT-weighted pooling that IRC safety also requires. The best detectors at equal fit are not IRC
safe.

## 3. Hypothesis verdicts

| hypothesis | pre-registered rule | paired seeds | Holm | without low-fit runs | H4: signs at 1, 3, 5 epochs |
|---|---|---|---|---|---|
| H1 interaction helps | inconclusive | inconclusive | inconclusive | inconclusive | 3-prong contrasts robust; EdgeConv − Deep Sets on BB1 flips |
| H2a IRC safety generalises | inconclusive | inconclusive | inconclusive | inconclusive | EWMP − EdgeConv flips at 1 epoch; attention pair flips at 5 |
| H2b IRC safety sculpts less | supported | inconclusive | inconclusive | supported | EFN − Deep Sets flips at 5 epochs, so only EWMP − EdgeConv is robust |
| H3 complete-graph GCN collapses | supported | supported | supported | supported | robust |

The pre-registered rules give:

- **H1 (interactions help): inconclusive.** Interactions help on the 3-prong signal and hurt on BB1. Section 2
  shows both effects follow the fit quality.
- **H2a (IRC safety generalises): inconclusive.** It is supported on the 3-prong signal only, and that signal is
  not blind for H2 (D5). The contrasts test the IRC-safe design as a whole; section 2.2 shows that in both families
  its equal-fit edge comes from the encoder side (direction-only inputs), not from IRC safety.
- **H2b (IRC safety sculpts less): supported but fragile.** It fails with paired seeds and with the Holm
  correction, and after the H4 check only EWMP − EdgeConv keeps its sign at every training length. The protocol
  drops a contrast that flips, which leaves H2b inconclusive.
- **H3 (complete-graph GCN collapses): supported** under every analysis. Its latent space also loses
  substructure: the MLP probe recovers τ21 with R² 0.24, against 0.50–0.61 for the other encoders.

The low-fit rule of D2 (1.25 × the configuration's median loss) was set after the two badly fitting Deep Sets
seeds had been seen, and it removes exactly those two. No verdict depends on it.

## 4. IRC safety in practice

Spearman ρ of the background jet scores before and after a perturbation, mean over the five E1 seeds (Figure 5;
pairwise contrasts in `audit_followup.md`, section 4).

- **Collinear splits.** The IRC-safe encoders are exactly invariant when constituents are split collinearly:
  ρ = 1.000 for the top-1, top-5 and all-constituent splits. The unsafe encoders are not: splitting every
  constituent gives ρ = 0.96 for Deep Sets, 0.95 for the transformer, 0.92 for EdgeConv and 0.83 for the GCN.
- **Soft emissions.** Two of the three IRC-safe encoders are clearly more robust than their unsafe partners, the
  third is not:

  | probe (ten emissions) | EFN vs Deep Sets | EWMP vs EdgeConv | e.w. attention vs Transformer | GCN |
  |---|---|---|---|---|
  | 0.2 % of the jet pT each | 0.988 vs 0.982 (n.s.) | 0.993 vs 0.968 | 0.999 vs 0.978 | 0.867 |
  | 1 % of the jet pT each | 0.75 ± 0.16 vs 0.73 ± 0.08 (n.s.) | 0.79 vs 0.53 | 0.98 vs 0.74 | 0.47 |

  The robust ones are the two whose interaction step is energy weighted. Ten emissions of 1 % each add 10 % of the
  jet pT, far from the soft limit that IRC safety is about. With 0.2 % each, every encoder except the GCN stays
  above ρ = 0.96.
- **Sculpting.** The dijet-mass sculpting at ε_B = 1 % is low for every autoencoder (JSD 0.008–0.014), far below
  a cut on jet mass (0.148).

## 5. BB1: detected about as well as jet mass alone

On BB1 the encoders do no better than jet mass or −τ21:

- The encoder scores correlate with m_J as strongly on the black box as on R&D (Spearman ρ ≈ 0.7, BB1
  diagnostics).
- Their BB1 background is barely shifted from the R&D background (KS 0.03–0.04).

**Is a worse fit a better mass tagger?** Within an architecture, yes (D9). The check uses 10 000 R&D background jets
per run and the Spearman correlation of the jet score with m_J (ρ_mass):
- **Within each encoder**, ρ_mass rises with the validation loss (Spearman +0.87 over 98 runs). It rises even
  between seeds of one configuration (+0.58).
- **The more mass-like the score, the higher the BB1 AUC**: +0.75 within encoder, +0.54 within configuration.
- **Across encoders** the relation holds without the complete-graph GCN (+0.75). The GCN fits worst but is the least
  mass-like (ρ_mass 0.68 against 0.70–0.77), which fits its collapse.

So the BB1 trend of section 2 is the score turning into a jet-mass tagger as the fit gets worse. Whether any encoder
adds information beyond the jet masses needs the black boxes (`docs/F2_SESSION.md`, step 4).

**Trained on BB1's own events** (E3, label-blind, 3 seeds), the encoders reach AUC 0.904–0.908 in the window.
These AUCs include the events the models were trained on. On the held-out validation part alone the AUC is
0.894–0.899 (`e3_summary.md`).

**Bump hunt.** A sideband-defined score cut at the known window gives a local significance of 2.3–2.5σ (10 % cut)
and 3.2–3.8σ (2 % cut; per seed 1.6–4.4σ), as computed by the analysis code. This quoted local Z overstates the
evidence. The D9 null test repeats the procedure without signal: three R&D-trained EFN runs scan their own training
sample, the whole R&D background (1 000 000 events, as many as BB1), over 14 windows of 400 GeV × the same cuts, with
3000 toy experiments.
- **Calibration.** Without signal, the quoted local Z at a fixed window is spread with sd 1.34, not 1, because it
  ignores the uncertainty of the background fitted in the sidebands. The calibrated local significance of the BB1
  result (2 % cut, mean per encoder) is 2.4–2.8σ.
- **Look-elsewhere.** A blind scan over these windows and cuts finds a largest local Z of 2.6 (median) and above 4.0
  in 5 % of background-only experiments. The real null scans give 2.2–2.7, in line with the toys, so the procedure
  does not create spurious bumps. Even if BB1's own scan peaked at the true window, its global significance would be
  0.8–1.4σ (p = 0.08–0.21 per encoder; the best single seed 1.9σ).
- **Fit convergence** is not an issue on these spectra: the analysis fit and a fully converged fit agree to 0.000
  in Z.

So BB1 is detected at the AUC level, but the bump hunt does not establish a significant excess.

**The BB2 test.** The R&D-trained main runs applied to BB2 (background only, Herwig) give a largest |Z| of 3.3
over 360 window × cut combinations. These runs were not trained on BB2, so this tests the R&D-trained procedure,
not the train-on-data hunt that produced the BB1 numbers.

## 6. BB3: the dijet mode looks more ordinary than QCD

Every encoder ranks BB3 signal below the background in the 4.0–4.4 TeV window (AUC 0.46–0.48). The diagnostics
rule out two artefacts:

- *Generator shift.* The BB3 background scores agree with the R&D background (KS ≈ 0.02; 10–11 % of jets above
  the R&D 90 % quantile).
- *Labels.* The modes are identified by their counts, and the Pythia cards confirm the 1200-event mode is the
  quark decay.

The low AUC comes from the 1200-event dijet mode. Its jets come from a KK gluon decaying to d d̄ (card
`KKg2qq`). They have fewer constituents than the background in the window (jet AUC of n_const 0.44), and the
encoders score them as more ordinary (jet AUC 0.44, the mean over seed 1 of the seven encoders).

The trijet mode (KK gluon → g + radion → ggg) is detected (jet AUC ≈ 0.71), but only 137 of its 2000 events fall
in the window. The flow on jet observables reaches 0.735 because it also uses pT, where the dijet mode differs
(jet AUC of pT 0.76). Note for the text: the LHCO paper's description of the dijet mode as gluonic does not match
the card (D4).

## 7. Ablations (H5, descriptive; 2-prong validation only)

- **Latent width.** Width 4–16 is equivalent. Width 2 is worse for every encoder (AUC 0.81–0.82 against
  0.84–0.86 at width 4), and width 32 is worse and less stable.
- **Pooling.** EdgeConv with pT-weighted pooling is the best configuration of the study on the dev signal (AUC
  0.854 ± 0.003). PMA pooling hurts Deep Sets badly (0.778).
- **Constituents.** 30 constituents are slightly worse than 50.
- **100 constituents** (E3, R&D-trained) change nothing that matters: EdgeConv 0.801, EWMP 0.820, transformer
  0.817 on the 3-prong signal; BB1 0.904–0.912; BB3 below 0.5 again.

## 8. Reproducibility

- **Rescoring.** Rescoring seed 1 of every encoder from the committed weights reproduces all 288 committed
  metrics of each run exactly (F1).
- **Splits.** All sessions verified the splits hash.
- **Run quality.** No main-grid or curve run was flagged. Two of 57 ablation runs were flagged; they stay in the
  means.

## Limitations to state

- **The recipe.** It was tuned for EdgeConv on the 2-prong signal. If detection follows the fit, the recipe matters
  through which encoders it lets fit well, so this concern is sharper, not weaker, under the fit-quality reading.
  The training-length curves limit it: from 1 to 5 epochs every encoder moves along the same curve, and EdgeConv is
  not singled out.
- **Fit quality is a summary, not a cause.** The curve is post hoc and conditions on a property of the encoder. The
  seed test (section 2) varies the fit at fixed architecture only by chance, and the encoders compared at equal fit
  never overlap in loss. A direct test would vary the fit at fixed architecture by design (learning rate, training
  data size).
- **The curve explains the 3-prong signal, not the black boxes.** On BB1 and BB3 the AUCs differ by less than 0.03,
  the curve explains less than the configuration, and it predicts new runs poorly (R² 0.23 and −0.85).
- **The IRC-safety contrasts test a design, not IRC safety.** In both families the equal-fit edge comes from the
  direction-only inputs, not the IRC-safe pooling (section 2.2).
- **Statistics.** The black boxes give few signal events in the window (BB1: 512; BB3 trijet: 137). The BB1 bump
  hunt's quoted local Z is not calibrated; after calibration and the look-elsewhere effect it is 0.8–1.4σ
  (section 5).
- **Generator dependence.** All signals are Pythia + Delphes.
- **Seeds.** H4 rests on 3 seeds per training length; E2–E5 have 3 seeds per configuration.
- **Unfinished black-box checks.** The E5 runs on BB1 and BB3, a blind BB1 scan, a BB2-trained null and BB1 at fixed
  jet masses need the black boxes (`docs/F2_SESSION.md`).
