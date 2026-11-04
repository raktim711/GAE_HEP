# Paper figures: files and draft captions

Regenerate with `python scripts/make_paper_figs.py`; the style is in `src/gaehep/eval/paper_style.py`.

**Files and format.** Each figure exists as a `.pdf` and a `.png`. The PDF is vector, with fonts embedded, and is
the one to submit. The PNG is a 300 dpi preview.

**Sizes.** Figures are made at final print size: double column is 7.0 in (17.8 cm), single column is 3.4 in
(8.6 cm). Include them without scaling, e.g. `\includegraphics[width=\textwidth]{fig1_auc_by_encoder.pdf}` in a
`figure*`, or `\columnwidth` for figures 5 and 6.

**Encoder encoding, the same in every figure.**
- Colour and marker = the interaction step: none (blue circle), local graph (green triangle), attention
  (vermillion square), complete-graph GCN (black diamond).
- Filled markers = IRC-safe; open markers = IRC-unsafe.

The colours are Okabe–Ito, safe for colour-vision deficiency. Shape and fill also keep the figures readable in
grey-scale.

| file | width | draft caption |
|---|---|---|
| `fig1_auc_by_encoder` | double | AUC in the signal window for each encoder on (a) the R&D 3-prong signal, (b) black box 1 and (c) black box 3. Markers are the mean over five seeds and bars the standard deviation. Dashed lines are single-observable and density baselines, with N-subjettiness computed from the constituents; "flow" is the normalising flow on jet observables. |
| `fig2_auc_vs_fit` | double | AUC of every autoencoder run against its background validation loss: 2 epochs × 5 seeds, plus 1, 3 and 5 epochs × 3 seeds. The line is a quadratic fit to these runs. Crosses are the latent-width, pooling and 30-constituent ablations, which were not used in the fit (out-of-sample test). |
| `fig3_training_length` | double | AUC against training length on (a) R&D 3-prong and (b) black box 1. Mean ± standard deviation over seeds: 5 at 2 epochs, 3 otherwise. Solid lines are IRC-safe, dashed IRC-unsafe. Points are offset horizontally for legibility. |
| `fig4_equal_fit` | double | Pre-registered encoder contrasts (first minus second) as measured (open squares) and at equal background fit (filled circles; residuals of the quadratic in Fig. 2). Bars are ±2 standard errors over seeds. Unpaired Welch SE for "as measured". |
| `fig5_irc_robustness` | single | Spearman rank correlation of the background jet scores before and after (a) splitting every constituent into two collinear halves and (b) adding ten soft particles of 1 % of the jet pT each. Mean ± standard deviation over five seeds. |
| `fig6_bb3_modes` | single | Jet-level AUC in the black box 3 window (4.0–4.4 TeV) for the two decay modes: dijet, KK gluon → q q̄; trijet, KK gluon → g + radion → ggg. "Encoders" is the mean (± sd) over all 35 encoder runs; the other bars are single observables. Bars start at AUC = 0.5 (random). |
