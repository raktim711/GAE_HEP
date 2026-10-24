# GAE_HEP — permutation-invariant graph autoencoders for anomaly detection at the LHC

Unsupervised anomaly detection on jets of the **LHC Olympics 2020 R&D dataset** with graph neural networks
whose anomaly scores are **invariant to the ordering of jet constituents**.  This continues the AIMS project
[`AIMS-project---Anomaly-Detection`](https://github.com/raktim711/AIMS-project---Anomaly-Detection)
(graph autoencoders on ATLAS HLT objects), fixing the problems found there:

* On complete graphs `GCNConv` and `GINConv` (ε = 0) give **every node of a graph the same embedding**
  (verified in `tests/test_legacy_collapse.py`), and GAT / SAGE(sum) collapse within a few layers on real
  jets (`gaehep collapse`), so the AIMS autoencoders reconstructed all objects of an event with one vector;
  this repo uses local k-NN graphs and EdgeConv instead.
* Raw, unscaled features let pT dominate the MSE → jet-relative, standardised node features.
* No real bottleneck → graph-level latent (set autoencoder) or a node latent narrower than the node's
  degrees of freedom.
* Hyper-parameters tuned on test-set AUC, best weights never restored → training and early stopping on
  background only (validation loss, deep-copied best state), design choices on a separate signal
  *validation* split, test splits for the reported numbers.
* Data vs MC, prescale weights, selection-driven AUCs → a fully simulated, labelled benchmark with
  literature reference numbers and SIC / sculpting metrics.

See [`RESULTS.md`](RESULTS.md) for the results and their discussion, [`PLAN.md`](PLAN.md) for the plan and
[`docs/literature.md`](docs/literature.md) for reference results.

## Results in brief

Signal-region (3.3 < m_jj < 3.7 TeV) event AUC on the test splits; the event aggregation of each score was
chosen on the 2-prong *validation* split and applied unchanged to both signals (no choice was made with the
3-prong signal).  Details, uncertainties, figures and the exact protocol in [`RESULTS.md`](RESULTS.md).

| score | 2-prong AUC | 3-prong AUC | m_jj sculpting (JSD, ε_B = 1 %) |
|---|---:|---:|---:|
| −τ21 (2-prong observable) | **0.893** | 0.658 | 0.018 |
| −τ32 (3-prong observable) | 0.688 | **0.854** | 0.014 |
| jet mass | 0.811 | 0.817 | 0.251 |
| set autoencoder, Chamfer on constituents | 0.488 | 0.532 | 0.004 |
| energy-flow graph VAE (latent 4, 5 epochs; 3 seeds) | 0.829 ± 0.006 | **0.805 ± 0.002** | **0.008** |
| **energy-flow graph VAE after the hyper-parameter study** (2 epochs, k = 8; 5 seeds) | **0.853 ± 0.005** | 0.799 ± 0.002 | 0.012 |
| same, GPU grid winner (2 epochs, kernels 0.02–0.05; 5 seeds) | **0.855 ± 0.002** | 0.801 ± 0.004 | 0.018 |

* Constituent-level graph autoencoders are *inverted* detectors here (signal jets are simpler than QCD);
  reconstructing the IRC-safe energy flow instead fixes this and reaches the level of an oracle (histogram)
  estimate of the background density in (m_J, pT_J, τ21) (AUC ≈ 0.80).
* Hyper-parameter study (78 dev runs, `results/dev/hp_study.md`): the anomaly score is best at an
  *intermediate* fit of the background — an autoencoder that fits QCD better also reconstructs the signal's
  clean two-prong jets better.  Training for 2 instead of 5 epochs is the one change that matters (narrower
  kernels or a stronger KL weight do the same and do not combine with it); graph, encoder, decoder and
  latent-size choices barely matter at the optimum.  The tuned model gains 0.024 AUC, 11 % max SIC and 35 %
  background rejection on the 2-prong signal (test), but not on the never-tuned 3-prong signal: the
  training length tunes the detector toward the signal it was tuned on.  A 72-point GPU grid (Kaggle T4)
  reached the same recipe independently and found one more lever on the dev split: 100 instead of 50
  constituents per jet (+0.016, two seeds; not yet on the test splits).  Latent-space flows added nothing.
* Unsupervised max SIC is only ≈ 1.6–1.8.  In the signal region, supervised classifiers on the GNN's
  permutation-invariant jet latents plus jet masses reach AUC 0.974–0.980 vs 0.970 for the standard
  (m_J1, Δm_J, τ21_J1, τ21_J2) features; in an idealised weakly supervised search those standard features
  remain better at almost every signal fraction, and the GNN representation reaches max SIC 8–10 for
  S/B ≳ 2 %.

## Data

LHC Olympics 2020 R&D, Zenodo record [6466204](https://zenodo.org/records/6466204): 1M QCD dijet events +
100k W′(3.5 TeV) → X(500 GeV) Y(100 GeV), X, Y → qq, plus 100k events of a 3-prong variant (X, Y → qqq)
used only as an *unseen* signal.  Events are clustered into anti-kT R = 1.0 jets (pT > 20 GeV) with
scikit-hep `fastjet`; the two leading jets are kept with their top-100 constituents.  Our jets match the
official feature files for all events within 1e-3 (relative), 99.9 % of them within 1.2e-5.

Splits (seeded): background 60 / 10 / 30 % (train / val / test); 2-prong signal 40 / 10 / 50 %; 3-prong
signal test only.  Unsupervised models are trained (and early-stopped) on background only; the 2-prong signal
validation split drives design choices (`gaehep dev`), the 2-prong signal training split trains the
supervised reference and is the injection pool of the weak-supervision study.

## Models (all scores permutation invariant; enforced by tests)

| id | model | score |
|---|---|---|
| M0 | `legacy_gae` — faithful port of the AIMS GAE (GCN/SAGE/GIN/GAT on complete graphs, raw features) | mean node MSE |
| M1 | `node_gae` — EdgeConv node-level GAE, 2-d node bottleneck | mean node MSE |
| M2 | `set_ae` — EdgeConv encoder → jet latent → set decoder, masked Chamfer loss (+ multiplicity) | Chamfer distance |
| M2c | `set_ae` with `cond_dim: 2` — decoder conditioned on (log m, log pT): mass-decorrelated | Chamfer distance |
| M2e | `ef_ae` — energy-flow graph AE: same encoder, decoder emits points **with pT fractions**, IRC-safe multi-scale energy MMD loss (optionally a VAE latent / mass-conditioned decoder) | energy MMD (or KL) |
| M3 | normalising flow (zuko NSF) on a model's jet latents, optionally conditioned on (log m, log pT) | −log p(z), −log p(z \| m) |
| M4 | `deep_svdd` — one-class Deep SVDD, same encoder | distance to centre |
| ref | `classifier` — supervised EdgeConv (idealised upper bound) | logit |
| ref | single observables m_j, τ21, τ32, multiplicity | value |

Event score = max (also min, sum) over the two leading jets.  Post-hoc variants: latent flows (`gaehep flow
[--conditional]`), mass-quantile normalisation (`gaehep qn`), rank ensembles (`gaehep ensemble`).

**Development protocol.** Design choices are compared with `gaehep dev <config>` (short training, metrics on
background *validation* events vs the 2-prong *signal validation* split; `gaehep dev-report` →
`results/dev/summary.md`).  No choice was made with the test splits or the 3-prong signal, although test
results of earlier models were seen during development (see the note at the top of `RESULTS.md`).  Final
models: `gaehep dev` → `gaehep promote <dev run> <name>` (scores the same weights on the test splits).

**Weak supervision.** `gaehep weak <run>` trains idealised data-vs-background classifiers (IAD, as in CATHODE /
"Back To The Roots") in the signal region on the run's permutation-invariant jet latents and on the standard
(m_J1, Δm_J, τ21_J1, τ21_J2) features, for several injected signal fractions.

## Usage

```bash
pip install -e ".[dev]"            # Python >= 3.11; torch CPU is fine
make test                          # unit tests (no data needed)
make smoke                         # end-to-end run on a synthetic dataset

gaehep download                    # ~3.2 GB from Zenodo (md5-verified, resumable)
gaehep preprocess --sample rnd     # ~5 min on 2 cores
gaehep preprocess --sample qqq
gaehep tables --sample rnd && gaehep tables --sample qqq && gaehep splits
gaehep baselines                   # single observables -> results/runs/baseline_*/

# development protocol (validation splits only) and the final model of the hyper-parameter study, five seeds
for s in 1 2 3 4 5; do
  gaehep dev configs/ef_vae.yaml --name e2_k8_s$s --set train.epochs=2 --set model_args.k=8 \
         --set train.seed=$s                                                # ~9 min on one core
  gaehep promote e2_k8_s$s hp_final_s$s                                     # test splits -> results/runs/
done
gaehep ensemble hp_final_s1 hp_final_s2 hp_final_s3 hp_final_s4 hp_final_s5 --out hp_final_5seeds
gaehep qn hp_final_5seeds                       # mass-decorrelated variant
gaehep bumphunt hp_final_5seeds --agg prod      # sideband-fit bump hunt with injected signal
gaehep weak hp_final_s1 --n-models 3            # weak supervision (IAD) on the jet latents vs HL4
gaehep dev-ensemble RUN1 RUN2 ... --out NAME    # rank ensemble of dev runs (seeds, or rec + KL scores)
gaehep collapse                                 # legacy AIMS GAE collapse study
gaehep report && gaehep dev-report              # results/summary.md, results/dev/summary.md
python scripts/hp_study.py                      # hyper-parameter study tables and figures

# hyper-parameter sweep (dev protocol) -> seed confirmation -> best configuration on the test splits
gaehep sweep configs/ef_vae.yaml --prefix sw --score best --budget-hours 8 \
       --grid '{latent_dim: [3, 4, 6], train.epochs: [2, 3, 5], model_args.sigmas: {s3: [0.02, 0.05, 0.1]}}'

gaehep train configs/set_ae.yaml   # any config: train + score + evaluate -> results/runs/set_ae/
```

Data live outside the repo (`GAEHEP_DATA`, default `/home/claude/data/lhco`); results in `results/`.
[`notebooks/kaggle_gaehep.ipynb`](notebooks/kaggle_gaehep.ipynb) runs the GPU continuation of the
hyper-parameter study on Kaggle: a 72-point `gaehep sweep` grid (latent size × β × kernel scales × epochs, both
score kinds), seed confirmation, the best configuration with five seeds on the test splits (ensemble, mass
decorrelation, bump hunt, weak supervision) and scale variants of it (100 constituents, wider or deeper
encoder, more decoder points, more jets per epoch); every step is idempotent, so reruns reuse finished runs.

## Layout

```
src/gaehep/data      download, preprocess (fastjet), tables (validation), splits, dataset (graph batches)
src/gaehep/graphs    dense masked k-NN / gather / pooling
src/gaehep/models    edgeconv, set_ae, ef_ae, node_gae, deep_svdd, classifier, legacy_gae, latent_flow
src/gaehep/losses    masked Chamfer, Sinkhorn, energy MMD
src/gaehep/train     training loop (checkpoint / resume / time budget), run orchestration, dev protocol, weak supervision
src/gaehep/eval      metrics (AUC, SIC, rejection, bootstrap), sculpting (JSD, dCor), bump hunt, IAD, plots, report
configs/             one YAML per experiment
tests/               invariance, losses, data, metrics, legacy-collapse, training smoke tests
```
