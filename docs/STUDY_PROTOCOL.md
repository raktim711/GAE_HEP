# GAE_HEP study protocol

This file freezes the design of the GAE_HEP study before any study run. The freeze date is the date of the commit that adds this file to `main`, and the git history records every later change. Every session reads this file first and follows it exactly. Anything that departs from it goes into [Deviations](#deviations) with the date, the reason and the effect on the numbers, and the paper lists every deviation.

## 1. Question

Take one permutation-invariant energy-flow autoencoder. Does its interaction step (message passing on a particle graph, or attention) give a better unsupervised anomaly embedding than a set model without interactions? And does infrared and collinear (IRC) safety matter?

Everything except the encoder's interaction step is shared and frozen (sections 3 and 4). Claims rest on signals that no design choice has seen (section 5).

## 2. Hypotheses and decision rules

**Metrics.** The primary metric is the event-level AUC in the signal's m_jj window, using the `prod` event score. The secondary metrics are:
- max SIC, computed with at least 50 background events;
- 1/ε_B at ε_S = 0.3;
- JSD: the Jensen–Shannon divergence between the background m_jj distributions before and after a cut at ε_B = 1 %.

**Signals.** The untuned signals are the R&D 3-prong signal, BB1 and BB3 (both modes combined). The R&D 2-prong signal is reported for every run, but it is never evidence for a hypothesis, because the recipe was tuned on it.

**Significance.** A contrast A − B on one signal and one metric is significant if |Δ| > 2·SE.
- Δ is the difference of the seed means.
- SE = √(s_A²/n_A + s_B²/n_B), with sample standard deviations.
- On BB1 and BB3 there is a second requirement. The difference between the two 5-seed rank ensembles must have a paired event bootstrap (200 resamples of the box's events, seed 20201) whose 95 % interval excludes zero.

**Verdicts.** Each hypothesis is judged supported, contradicted (a significant result goes the other way) or inconclusive.

| | Contrasts | Supported if |
|---|---|---|
| H1 interaction | Primary: edgeconv − deepsets, ewmp − efn. Secondary: transformer − deepsets, ew_attention − efn. All on AUC. | Both primary contrasts are significantly positive on at least two untuned signals, and significantly negative on none. |
| H2a IRC safety, generalisation | ewmp − edgeconv, efn − deepsets, ew_attention − transformer, on AUC | At least two contrasts significantly favour the IRC-safe encoder on at least two untuned signals, and none significantly favours the unsafe one. |
| H2b IRC safety, sculpting | The same three contrasts, on JSD of the R&D background test split | At least two contrasts have significantly lower JSD for the IRC-safe encoder, and none has significantly higher. |
| H3 collapse | gcn_complete − deepsets, on AUC | gcn_complete is not significantly better on any signal. |
| H4 training length | Every contrast found significant for H1–H3 | The 3-seed mean difference has the same sign at 1, 3 and 5 epochs. A contrast that flips is reported as not robust and left out of the H1–H3 verdicts. |
| H5 width and constituents | Latent width 2–32; 30, 50 and 100 constituents; pooling | Descriptive only, no verdict. |

Everything not listed is exploratory and is labelled so in the paper.

**Flagged runs.** A run is flagged when its `best_val_loss` (in `history.json`) is above 1.5 × the median of its configuration. A configuration is the same encoder, epochs, ablation setting and training sample. Flagged runs stay in every mean and every table, where they are marked. The paper also shows the main table without them.

## 3. Encoders

| `encoder` | Interaction | IRC-safe | Node inputs | Pooling |
|---|---|---|---|---|
| `deepsets` | none | no | 7 standard features | masked mean + max |
| `edgeconv` | EdgeConv on a k-NN graph, k = 8, first in (Δη, Δφ), then dynamic | no | 7 standard features | masked mean + max |
| `transformer` | self-attention: 4 heads, pre-LayerNorm, MLP ratio 2, no position encoding | no | 7 standard features | masked mean + max |
| `gcn_complete` | mean over all real nodes of the jet, itself included, then an MLP (the collapse control) | no | 7 standard features | masked mean + max |
| `efn` | none | yes | (Δη, Δφ, ΔR) | pT-fraction weighted (EWMP pooling) |
| `ewmp` | energy-weighted messages on a radius graph, R = 0.3 | yes | (Δη, Δφ, ΔR) | pT-fraction weighted |
| `ew_attention` | attention with log z_j added to the logits, so that the weights ∝ z_j · exp(q_i·k_j/√d) | yes | (Δη, Δφ, ΔR) | pT-fraction weighted |

- **Standard features.** The 7 standard node features are Δη, Δφ, log pT, log E, log(pT/pT_jet), log(E/E_jet) and ΔR, standardised on background-training nodes.
- **Size.** Every encoder has 3 layers of width 48. Parameter counts differ between encoders; they are reported, not matched.
- **IRC safety.** IRC-safe means the encoder is exactly invariant when a constituent is split collinearly. A soft emission carrying pT fraction z changes it by O(z). Tests check both properties.
- **k-NN ties.** Distance ties in k-NN graphs are broken by larger pT, then by (Δη, Δφ), which makes every k-NN graph exactly permutation-invariant. This changes `edgeconv` only on jets with ties, which are 0.1 % of jets at k = 8.
- **Configs.** `configs/study/<encoder>.yaml` equals `configs/ef_vae_k8_e2.yaml` except for `model_args.encoder`, plus `radii: [0.3]` for `ewmp`. Encoders without a k-NN graph ignore `k`.
- **Pooling ablation.** For `deepsets` and `edgeconv` only, `model_args.pool` can be `meanmax` (the default), `weighted` (EWMP's pT-fraction weighting) or `pma` (one learned seed attending over the nodes).

## 4. Training and scoring

The recipe is the same for every encoder (`configs/ef_vae_k8_e2.yaml`).

**Model.**
- The pooled encoder output goes through an MLP [128] to a latent of size 4. This is a VAE with β = 10⁻³.
- The decoder is an MLP [256, 256] that emits 32 points, each with (Δη, Δφ, pT-fraction logit).
- The loss is an energy-weighted MMD with Gaussian kernels σ ∈ {0.02, 0.05, 0.1, 0.2, 0.4}, on the 50 leading constituents of each jet.

**Training.**
- Each run is 2 epochs of 120 000 jets drawn from `bkg_train` each epoch, with `default_rng([seed, epoch])`, and batch size 256.
- The optimiser is AdamW with learning rate 10⁻³ and weight decay 10⁻⁴, a linear warm-up and cosine decay over the whole run.
- The best weights are chosen by the background loss on 20 000 `bkg_val` jets.
- Each run uses one thread. Each call has a 100-minute budget; rerun the same command to resume.
- A training length is a full run with its own schedule, not a checkpoint of a longer run.

**Seeds.** Seeds 1–5 for the main grid; seeds 1–3 for the training-length curves, the ablations and E3.

**Commands.**
- Development run: `gaehep dev configs/study/<enc>.yaml --name <run> --set train.epochs=2 --set train.seed=<s> --set train.threads=1`
- Test scores of `bkg_val`, `bkg_test`, `sig_test` and `sig3_test`: `gaehep promote <dev run> <study run>`
- Rank ensemble (the average over runs of per-jet `bkg_val` percentiles): `gaehep ensemble <runs> --out <name>`

**Warning.** `gaehep dev` sets `train.epochs` to the 5-epoch development budget, whatever the config says. `--set train.epochs=2` restores the recipe, so it goes on every study command except the training-length curves, which set their own epochs.

**Scores.**
- The jet score is the energy-MMD reconstruction (key `score`). The KL extra score is not used for claims.
- The event score is `prod` = log F_B(s₁) + log F_B(s₂). F_B is the empirical CDF of the per-jet scores on the validation split of the training sample. For R&D-trained runs that split is `bkg_val`; for runs trained on BB1 it is BB1's validation part. Labels never enter F_B.

## 5. Data

### R&D (development and test)

LHCO 2020 R&D (Zenodo 6466204) and its 3-prong signal. Jets are anti-kT with R = 1.0, keeping the two leading jets. Constituents are sorted by pT and truncated at 50 (100 for the 100-constituent runs).

| Split | Events | Use |
|---|---|---|
| `bkg_train` | 600 000 | training |
| `bkg_val` | 100 000 | early stopping, F_B, development metrics |
| `bkg_test` | 300 000 | test |
| `sig_train`, `sig_val` | 40 000, 10 000 | supervised reference; `sig_val` is the only development signal |
| `sig_test` | 50 000 | 2-prong test |
| `sig3_test` | 100 000 | 3-prong test |

The splits are in `processed/splits.npz`, made with seed 20201 from events with at least two jets. Their content hash is the sha256 over the sorted split names, each followed by its dtype string and its raw bytes:

```python
import hashlib
import numpy as np
from gaehep.config import processed_dir

f = np.load(processed_dir() / "splits.npz")
h = hashlib.sha256()
for k in sorted(f.files):
    a = f[k]
    h.update(k.encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
assert h.hexdigest() == "5217cca54367efda6b33a1626da8cdb21e87ef7b6fd6b2abfa73d8ac510f7af2"
```

P2 adds this as `gaehep.data.splits.content_hash()` and records it in every run's metadata as `splits_sha256`. Every session checks the hash before its first run and stops if it differs. The R&D splits must stay bit-identical.

### Black boxes (test only)

Source: Zenodo 4536624 (CC BY 4.0). Each box holds 1M events, each with 700 particles × (pT, η, φ), zero-padded and without a label column. The master keys hold one label per line, in file order. The md5 checksums below were taken from the record page on 31 Oct 2026:

| File | md5 |
|---|---|
| `events_LHCO2020_BlackBox1.h5` | `847be306afd923a4ba30543070113627` |
| `events_LHCO2020_BlackBox1.masterkey` | `c5d9b1e75ce26fd820c1c99dfc874e92` |
| `events_LHCO2020_BlackBox2.h5` | `2e5dce598238789938cb31e9cd3e7a46` |
| `events_LHCO2020_BlackBox3.h5` | `ffc53e0c0b42e0f2c752ce48db74ce27` |
| `events_LHCO2020_BlackBox3.masterkey` | `51e275f122eb48b10ccc2c858dda358e` |

The box contents are from Kasieczka et al. 2021, arXiv:2101.08320.

| Sample | Content | Role | Window |
|---|---|---|---|
| `bb1` | 834 signal events: Z′ (3.823 TeV) → X (732 GeV) Y (378 GeV), X, Y → qq̄. Pythia tune, PDF, scales and Delphes are changed. | untuned test; training sample of the train-on-data scenario | 3.6–4.0 TeV |
| `bb2` | background only, generated with Herwig++ and a changed Delphes card | null test | – |
| `bb3` | a 4.2 TeV resonance with a dijet mode (1200 events) and a trijet mode through a 2.217 TeV radion (2000 events). Pythia is changed. | untuned test | 4.0–4.4 TeV |

- **Labels.** 0 is background and 1 is signal. In `bb3`, labels 1 and 2 are the two modes, identified by their counts: 1200 for the dijet mode, 2000 for the trijet mode. Labels are used only for evaluation and to draw the subsample.
- **Processing.** Exactly as for R&D: the same clustering, constituents and tables, as samples `bb1`, `bb2` and `bb3`. The raw files are deleted after a verified run.
- **Splits.**
  - `bb<i>_test`: every event with at least two jets.
  - `bb<i>_test_sub`: 200 000 background events drawn without replacement with `default_rng(20201)`, plus every signal event.
  
  Results use the subsample; scoring the full box is optional.
- **Windows.** Each window spans 400 GeV around the resonance mass, aligned with the bump hunt's 100 GeV bins. They use the true masses, so they serve the encoder comparison, not a blind search. The full m_jj range is always reported too, because most `bb3` trijet events fall outside the window.

### Train-on-data scenario (E3)

`data.train_sample: bb1` takes all BB1 events with at least two jets and splits them 90/10 into training and validation with `default_rng(20201)`. The labels are never read for training or for F_B, and the recipe is unchanged. Metrics are reported on all BB1 events and, as a check, on the validation part alone. The bump hunt is the existing `gaehep.eval.bumphunt.search` with the box's window. It uses sideband-defined cuts that keep 10 % and 2 % of sideband events, and reports the local significance with and without the cut. On BB2 the same search runs in all three windows (3.3–3.7, 3.6–4.0 and 4.0–4.4 TeV). It is expected to find nothing, and any excess is reported.

## 6. Baselines

Every baseline gives per-jet scores, where larger means more anomalous. It then uses `prod` and the metrics of section 2.

| Baseline | Definition | Runs |
|---|---|---|
| single observables | m_J, n_const, −τ21 and −τ32, from `evaluate.observable_scores`, written by `gaehep obs-baselines` | `study_obs_mj`, `study_obs_nconst`, `study_obs_tau21`, `study_obs_tau32` |
| jet-observable flow | a zuko NSF with `LatentFlow`'s defaults (4 transforms, hidden (128, 128)) on standardised (log m_J, log pT_J, τ21, τ32, log n_const). It is trained on `bkg_train` jets with `LatentFlow`'s fit defaults and early-stopped on the `bkg_val` NLL. The score is −log p. | `study_hldensity_s1` … `_s5` |
| EFP k-NN | the prime EFPs with d ≤ 4 (`energyflow` `EFPSet('d<=4', 'p==1')`, hadronic measure, β = 1, normalised), log-transformed and standardised on the reference set. The reference set is 100 000 `bkg_train` jets drawn with `default_rng(20201)`. The score is the mean Euclidean distance to the k = 10 nearest reference jets. | `study_efpknn` |
| Chamfer set AE | the existing `results/runs/set_ae`, from the development history | `set_ae` |
| supervised EdgeConv | the existing `results/runs/classifier`, trained on 2-prong `sig_train`. It is the upper bound for the 2-prong signal and a mismatched reference elsewhere. | `classifier` |
| OmniLearned-small features | frozen features plus the latent flow (stretch goal, on Kaggle) | later |

### Scorer interface (P2 and P3)

`gaehep.baselines.SCORERS` maps a run config's `model` value (`observable`, `hl_density` or `efp_knn`) to a function `(run_dir: Path, data: Data, split: str) -> np.ndarray`. The function returns float64 per-jet scores of shape `(n_events, 2)` for any split that `Data` knows, including the black-box splits once P2 has registered them.

A baseline run directory holds the same files as a model run:
- `config.yaml`, with `model`;
- `scores.npz`, with `bkg_val`, `bkg_test`, `sig_test` and `sig3_test`;
- `metrics.json`.

`gaehep score-sample` uses `SCORERS` for baseline runs and `build_model` for everything else. P2 and P3 are written in parallel, so P2 guards the import and tests the dispatch with a stub.

## 7. Analysis outputs

**Black-box files.** Each run gets two files per box, written by P2:
- `<run>/bb<i>_scores.npz` holds:
  - `jet_scores`: `(n, 2)`, float32
  - `event_index`: `(n,)`, int64, the row in the box's event table
  - `mjj`: float32
  - `label`: int8
  - `subsample`: a bool scalar
- `<run>/bb<i>_metrics.json` holds:
  - top-level keys `name`, `sample`, `subsample`, `splits_sha256`, `git`, `time` and `metrics`;
  - `metrics` has `event`, `jet` and `sculpting` with the same structure as in `metrics.json`;
  - the event entries are keyed `"<signal>/<region>"`, where the signal is `bb1`, or one of `bb3`, `bb3_dijet` and `bb3_trijet`, and the region is `window` or `full`;
  - for BB2, `metrics` instead holds `ks` (the KS distance of the box's scores to the R&D `bkg_test` scores, per aggregation) and `bumphunt` (rows per window).

**Where the tables read.** The `headline` block of `metrics.json` uses `max` and is not used for the study.

| Signal or quantity | Field |
|---|---|
| R&D 2-prong | `metrics.json` → `metrics.event.prod["sig_test/SR"]` |
| R&D 3-prong | `metrics.event.prod["sig3_test/SR"]` |
| sculpting (JSD) | `metrics.sculpting.prod.mjj["jsd@0.01"]` |
| BB1 | `bb1_metrics.json` → `metrics.event.prod["bb1/window"]` |
| BB3 | `bb3_metrics.json` → `metrics.event.prod["bb3/window"]` |

The full m_jj range is read the same way, with `full` in place of the window.

**Probes.** `gaehep probes <run>` writes `probes.json`.
- Inputs: the saved latents (`latents.npz`) of the 20 000 jets in the first 10 000 `bkg_val` events.
- Standardised targets: m_J, pT_J, τ21, τ32 and n_const, plus three EFPs with hadronic measure and β = 1:
  - the wedge, with edges {(0,1), (1,2)};
  - the triangle, with edges {(0,1), (1,2), (2,0)};
  - the square, with edges {(0,1), (1,2), (2,3), (3,0)}.
- Models: ridge regression (α = 1) and an MLP with hidden layers (64, 64) and `random_state=20201`.
- Cross-validation: 5-fold `KFold(shuffle=True, random_state=20201)`.
- Output: R² mean and standard deviation over the folds.

**IRC probe.** `gaehep irc-probe <run>` writes `irc_probe.json`. It applies the perturbations of `scripts/irc_probe.py`, plus a random re-ordering of the stored constituents, and counts k-NN distance ties.

**Study table.** `gaehep study-table <prefix>` writes three files to `results/study/`:
- `<prefix>_table.md` and `.csv`: one row per configuration (encoder or baseline), and for each signal and metric the mean ± standard deviation over seeds, the rank-ensemble value and the number of flagged runs.
- `<prefix>_contrasts.json`: every pair of encoders × signal × metric, with Δ, SE, significance and, for BB1 and BB3, the ensemble bootstrap interval.

The command tolerates missing files. Ensembles and bootstrap intervals need the score files, which are not committed (section 8), so the session that has the score files computes them.

## 8. Runs, scheduling and commits

| Run name | What | Seeds | Session |
|---|---|---|---|
| `std_<enc>_s<s>` | main-grid development runs | 1–5 | E1 |
| `study_<enc>_s<s>`, `study_<enc>_ens` | their test promotions and 5-seed ensembles | 1–5 | E1 |
| `study_hldensity_s<s>`, `study_efpknn`, `study_obs_*` | baselines | 1–5, or none | E1 |
| `stc_<enc>_e<E>_s<s>` → `stct_<enc>_e<E>_s<s>` | training-length curves at E = 1, 3 and 5 epochs, and their test promotions | 1–3 | E2 |
| `stw_<enc>_l<d>_s<s>` | latent width d = 2, 8, 16, 32 for deepsets, edgeconv and ewmp (development only) | 1–3 | E2 |
| `stp_<enc>_<pool>_s<s>` | pooling `weighted` and `pma` for deepsets and edgeconv (development only) | 1–3 | E2 |
| `stn_<enc>_n30_s<s>` | 30 constituents for deepsets, edgeconv and ewmp (development only) | 1–3 | E2 |
| `std_bb1data_<enc>_s<s>` → `study_bb1data_<enc>_s<s>` | trained on BB1, for edgeconv, ewmp and efn | 1–3 | E3 |
| `std_n100_<enc>_s<s>` → `study_n100_<enc>_s<s>` | 100 constituents for edgeconv, ewmp and transformer, with `model_args.ewmp_checkpoint: true` for ewmp | 1–3 | E3 |

**Scheduling.**
- Each session sets up the data once (`make data`, then `make data-blackbox`) and checks the hash.
- Jobs are single-thread and run in the background, as many in parallel as `nproc`, and at most 3 at 100 constituents. Check `free -g` before adding a job.
- Every job is time-boxed and resumable by rerunning the same command. Training is time-boxed at 100 minutes per call; scoring resumes per chunk.
- Wait on the background jobs, and check progress at most every 20–30 minutes.

**Commits.**
- Commit the tracked files of `results/dev/<run>/` and `results/runs/<run>/`. The `.gitignore` excludes checkpoints, scores and latents.
- Commit trained weights as `results/models/<run>/model.pt` with `git add -f` (about 0.5 MB each), and everything in `results/study/`.
- Never commit data, score or latent files. They can be regenerated from the committed weights, and anything that needs them is computed in the session that has them.
- Every experiment PR states its wall time, flagged runs and deviations.

## 9. Test numbers and changes

The development history (RESULTS.md, sections 1–11) made its choices on the R&D 2-prong validation split and saw 3-prong test numbers. It goes into an appendix as development history. From the first study run on:

- No choice in sections 2–7 changes.
- Test, 3-prong and black-box numbers come only from the frozen pipeline.
- After the first test number, a bug fix is allowed only as a deviation, reported with the numbers before and after.
- No decision looks at any signal other than the 2-prong validation split, and none is planned.

## Deviations

Each entry gives the date, what changed, why, and the effect on the numbers. Entries D2–D5 were written on
1 Nov 2026, after the E1 results (PR #11) had been seen.

**D1 (E2 session, 31 Oct). `gaehep promote` refuses a run whose training was paused by the time budget.**
A paused `gaehep dev` exits with partial weights; promoting it would score an undertrained model. Every promoted
run of E1–E3 was checked to have `status: done` and `epochs_done` equal to its configured epochs, so no number
changed.

**D2 (1 Nov). Post-hoc robustness analyses of the verdicts (`gaehep posthoc`).** The verdicts of section 2 stay
the pre-registered ones. Four analyses are added after seeing E1, and are reported beside them, labelled post hoc:
- *Paired seeds.* Seed s of every encoder sees the same training jets (`default_rng([seed, epoch])`), so the seeds
  are paired; contrasts are also computed from per-seed differences.
- *Holm correction* over all primary tests of H1–H3 (Welch p-values, α = 0.05), because section 2 has no
  multiplicity correction.
- *Low-fit runs.* In E1, two of five Deep Sets seeds (2 and 4) ended at 1.4 × the median validation loss with a
  3-prong AUC of 0.72 against 0.79 for the other seeds; the 1.5 × flag did not catch them. Contrasts are repeated
  without runs above 1.25 × their configuration's median.
- *Equal background fit.* Across all encoders the AUC is close to a single function of the background
  validation loss (3-prong falling, BB1 rising; `results/study/figures/study_auc_vs_valloss.png`). Contrasts are
  repeated on the residuals of a quadratic fit of AUC in the validation loss. Because the fit quality is itself
  an outcome of the encoder, this is a conditional comparison ("at equal fit"), not a replacement.

Section 2 does not define "contradicted" exactly. `gaehep posthoc` uses the mirror image of each rule: H1 is
contradicted if both primary contrasts are significantly negative on at least two untuned signals and positive on
none; H2a/H2b likewise; H3 is contradicted if `gcn_complete` is significantly better than `deepsets` on any signal.

**D3 (1 Nov). N-subjettiness computed from the constituents (`gaehep tau`).** The black boxes have no LHCO
feature files, so their τ were NaN: the τ21/τ32 baselines sat at AUC 0.500 on BB1/BB3, and the jet-observable
flow replaced τ by training means there. τ1–τ3 (β = 1, unnormalised, minimised axes from exclusive-kT seeds) are
now computed from the stored constituents (`src/gaehep/data/nsubjettiness.py`). On 10 000 R&D jets they agree
with the official values to 1e-3 (relative) for 84 % of jets (90 % of jets with fewer than 100 stored
constituents), with correlation 0.9996 for τ21 and 0.998 for τ32. That is below the 99 % agreement set as the
threshold for mixing the two, so the computed values are used for **every** sample wherever τ enters a baseline
that is compared across samples. The affected baselines are rerun as `fix_obs_*` and `fix_hldensity_s*` with
`tau: computed`; E1's `study_obs_tau*` and `study_hldensity_s*` stay as the before-numbers. The encoders do not
use τ.

**D4 (1 Nov). Black-box diagnostics (`gaehep bb-diagnose`).** Every encoder is below AUC 0.5 on BB3 (dijet mode
0.43). Before any interpretation, the diagnostics report single-observable AUCs per mode, the shift of the
box-background scores against the R&D background, and the correlation of the scores with m_J. The BB3 modes are
identified by their counts (1200 dijet, 2000 trijet). The LHCO paper describes the dijet mode as a decay to
gluons; the Pythia card in the Zenodo record is named `KKg2qq`. The cards are checked before the paper
describes the mode.

**D5 (1 Nov). Disclosure: the IRC-safety hypothesis was formed after seeing 3-prong results.** No hyper-parameter
was chosen with the 3-prong signal (the radius 0.3 is from Atkinson et al.), but the IRC-safe encoder's 3-prong
advantage was seen during development (RESULTS.md §11) before H2 was written. The 3-prong signal is therefore not
a blind test of H2, nor of the H1 contrast `ewmp − efn`. BB1 and BB3 are the confirmatory tests for these, and
the paper says so.

**D6 (E1 session, 31 Oct–1 Nov). Session conditions.** The environment was built with the PyPI torch wheel (CPU
use only), one container restart interrupted the first runs (they resumed from their checkpoints), and the bb3
download was repeated after a collision of two downloads (md5 verified). No number is affected (PR #11).
