# Results (draft)

Draft of the results section. Every number comes from the committed tables: `results/study/study_table.md`,
`stct_table.md`, `study_posthoc.md`, `fix_table.md`, `study_n100_table.md`, `study_bb1data_table.md`,
`e3_summary.md`, `bb1_diagnostics.md`, `bb3_diagnostics.md`, `ablations.md`, and for E4 the ablation
section of `study_posthoc.md`. Unless stated otherwise:

- the event score is `prod`;
- each value is a mean ± sample standard deviation over 5 seeds (E1) or 3 seeds (E2, E3);
- AUC is quoted in the signal window: 3.3–3.7 TeV for R&D, 3.6–4.0 TeV for BB1, 4.0–4.4 TeV for BB3.

The R&D 2-prong signal is the one the recipe was tuned on, and it is never used as evidence. Deviations from the
protocol are D1–D7 in `docs/STUDY_PROTOCOL.md`.

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
- **BB1.** BB1 (Z′ → XY with heavy X, Y) is found by every encoder (AUC 0.90–0.92). Jet mass alone (0.918) and
  −τ21 alone (0.917) do just as well, and the weakest encoders on the 3-prong signal (GCN, Deep Sets) are the
  strongest here.
- **BB3.** Every encoder is below 0.5 on BB3 (section 6).

The EFP k-NN baseline reproduces exactly when rerun and has no bug. It is weak at event level, because signal
events rarely have two anomalous jets (D7).

## 2. Background fit sets the detection performance

Figure 1 (`results/study/figures/study_auc_vs_valloss.png`) plots each run's AUC against its background
validation loss. It covers all 98 autoencoder runs of E1 and E2 (seven encoders at 1, 2, 3 and 5 epochs), plus
the 42 E4 ablation runs as squares.

- **One curve, not seven.** Encoders of every kind fall on a single curve.
- **Opposite directions.** On the R&D 3-prong signal, the AUC rises steadily as the background fit improves:
  from about 0.67 at loss 0.6 to 0.82–0.83 below loss 0.16. On BB1 it does the opposite: the worst-fitting runs
  are the best detectors (0.92 at loss ≥ 0.4, against 0.89 at loss 0.13).
- **Training length moves each encoder along the same curve.** EdgeConv, for example, goes from 0.776 to 0.808 on
  the 3-prong signal and from 0.919 to 0.894 on BB1 between 1 and 5 epochs.
- **Seeds do too.** The two low-AUC Deep Sets seeds of E1 (0.72 against 0.79) are the two that fit worst (loss
  0.42 against 0.29).

A quadratic in the validation loss fitted to all 98 runs leaves a residual spread of 0.011 (3-prong), 0.0045
(BB1) and 0.0034 (BB3). Table 2 repeats the pre-registered contrasts on these residuals, that is, at equal fit.
This is post hoc (D2). Because the fit quality is itself a property of the encoder, it is a conditional
comparison, not a replacement for Table 1.

| contrast | R&D 3-prong, raw → at equal fit | BB1, raw → at equal fit |
|---|---|---|
| EdgeConv − Deep Sets (interaction, unsafe) | +0.040* → +0.008 | −0.006* → +0.002 |
| EWMP − EFN (interaction, safe) | +0.007* → +0.002 | −0.004* → −0.002 |
| EWMP − EdgeConv (IRC safety, graph) | +0.019* → **+0.011*** | −0.002 → +0.001 |
| EFN − Deep Sets (IRC safety, no interaction) | +0.051* → **+0.017*** | −0.003 → **+0.006*** |
| e.w. attention − transformer (IRC safety, attention) | +0.009* → +0.004 | +0.003 → **+0.006*** |
| GCN complete − Deep Sets (collapse) | −0.071* → +0.014 | +0.005 → +0.001 |

\* significant (raw: protocol rule; at equal fit: |Δ| > 2 SE of the residual difference)

**At equal fit, message passing and attention bring nothing measurable.** Neither primary interaction contrast is
significant on any signal. Part of the IRC-safety advantage remains: EFN − Deep Sets, EWMP − EdgeConv and
energy-weighted attention − transformer all keep a positive residual on at least one untuned signal, never a
significantly negative one.

**Out-of-sample test of the curve (E4, post hoc, D8).** The 42 Deep Sets and EdgeConv ablation runs (latent width
2–32, pooling, 30 constituents) were not used to fit the curve. Scored on the test signals afterwards, they give:

| signal | curve residual sd (E1 + E2) | mean residual of the ablations | ablations within 2 sd |
|---|---|---|---|
| R&D 3-prong | 0.011 | −0.011 | 60 % (width 54 %, pooling 50 %, 30 constituents 100 %) |
| BB1 | 0.0045 | −0.005 | 90 % |
| BB3 | 0.0034 | +0.002 | 88 % |

On the black boxes the curve predicts new runs to about ±0.005. On the 3-prong signal it overestimates the
ablations by 0.011 on average, and about half of the width and pooling variants fall below it. Fit quality
therefore explains most of the differences between models, but not all of them: on the 3-prong signal, latent
width and pooling also cost detection at equal fit.

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
  not blind for H2 (D5).
- **H2b (IRC safety sculpts less): supported but fragile.** It fails with paired seeds and with the Holm
  correction, and after the H4 check only EWMP − EdgeConv keeps its sign at every training length. The protocol
  drops a contrast that flips, which leaves H2b inconclusive.
- **H3 (complete-graph GCN collapses): supported** under every analysis. Its latent space also loses
  substructure: the MLP probe recovers τ21 with R² 0.24, against 0.50–0.61 for the other encoders.

## 4. IRC safety in practice

- **Collinear splits.** The IRC-safe encoders are exactly invariant when a constituent is split collinearly:
  Spearman ρ of the background scores is 1.0000 for the top-1, top-5 and all-constituent splits. The unsafe
  encoders are not. Splitting every constituent drops EdgeConv's ρ to 0.92 and the GCN's to 0.83.
- **Soft emissions.** Ten soft emissions of 1 % of the jet pT each are the hardest test: ρ is 0.53 for EdgeConv,
  0.47 for the GCN, 0.79 for EWMP, 0.75 for EFN and 0.98 for energy-weighted attention.
- **Sculpting.** The dijet-mass sculpting at ε_B = 1 % is low for every autoencoder (JSD 0.008–0.014), far below
  a cut on jet mass (0.148).

## 5. BB1: found as a heavy-jet signal

On BB1 the encoders do no better than jet mass or −τ21:

- The encoder scores correlate with m_J as strongly on the black box as on R&D (Spearman ρ ≈ 0.7, BB1
  diagnostics).
- Their BB1 background is barely shifted from the R&D background (KS 0.03–0.04).

**Trained on BB1's own events** (E3, label-blind, 3 seeds), the encoders reach AUC 0.904–0.908 in the window.
A bump hunt with a sideband-defined score cut finds the resonance at a local significance of 2.3–2.5σ (10 % cut)
and 3.2–3.8σ (2 % cut, seed range 1.6–4.4σ). There is no look-elsewhere correction and no fit-parameter
uncertainty, so this is optimistic, and it is below 5σ.

**The null test passes.** The same procedure applied to BB2 (background only, Herwig) with all 40 R&D-trained main
runs gives a largest |Z| of 3.3 over 360 window × cut combinations, as expected from chance.

## 6. BB3: the dijet mode looks more ordinary than QCD

Every encoder ranks BB3 signal below the background in the 4.0–4.4 TeV window (AUC 0.46–0.48). The diagnostics
rule out two artefacts:

- *Generator shift.* The BB3 background scores agree with the R&D background (KS ≈ 0.02; 10–11 % of jets above
  the R&D 90 % quantile).
- *Labels.* The modes are identified by their counts, and the Pythia cards confirm the 1200-event mode is the
  quark decay.

The low AUC comes from the 1200-event dijet mode. Its jets come from a KK gluon decaying to d d̄ (card
`KKg2qq`). They have fewer constituents than the background in the window (jet AUC of n_const 0.44), and the
encoders score them as more ordinary (jet AUC 0.44).

The trijet mode (KK gluon → g + radion → ggg) is found (jet AUC ≈ 0.71), but only 137 of its 2000 events fall
in the window. The flow on jet observables reaches 0.735 because it also uses pT, where the dijet mode differs
(jet AUC of pT 0.76). Note for the text: the LHCO paper's description of the dijet mode as gluonic does not
match the card (D4).

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

- **The recipe.** It was tuned for EdgeConv on the 2-prong signal. Section 2 suggests this matters less than
  feared, because the fit quality, not the recipe, decides.
- **Statistics.** The black boxes give few signal events in the window (BB1: 512; BB3 trijet: 137).
- **Generator dependence.** All signals are Pythia + Delphes.
- **The equal-fit comparison** is post hoc and conditions on a mediator.
- **The fit-quality curve is not universal.** It holds out of sample on the black boxes, but on the 3-prong
  signal the latent-width and pooling ablations fall about 0.011 below it (section 2). The headline should be
  "fit quality explains most of the differences", not "fit quality decides".
- **Seeds.** H4 rests on 3 seeds per training length.
