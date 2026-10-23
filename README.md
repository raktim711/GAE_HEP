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
* Hyper-parameters tuned on test-set AUC, best weights never restored → label-free model selection on
  background validation loss, deep-copied best state.
* Data vs MC, prescale weights, selection-driven AUCs → a fully simulated, labelled benchmark with
  literature reference numbers and SIC / sculpting metrics.

See [`RESULTS.md`](RESULTS.md) for the results and their discussion, [`PLAN.md`](PLAN.md) for the plan and
[`docs/literature.md`](docs/literature.md) for reference results.

## Results in brief

Signal-region (3.3 < m_jj < 3.7 TeV) event AUC on the test splits; the event aggregation of each score was
chosen on the 2-prong *validation* split and applied unchanged to both signals (the 3-prong signal is never
used for any choice).  Details, uncertainties and figures in [`RESULTS.md`](RESULTS.md).

| score | 2-prong AUC | 3-prong AUC | m_jj sculpting (JSD, ε_B = 1 %) |
|---|---:|---:|---:|
| −τ21 (2-prong observable) | **0.893** | 0.658 | 0.018 |
| −τ32 (3-prong observable) | 0.688 | **0.854** | 0.014 |
| jet mass | 0.811 | 0.817 | 0.251 |
| set autoencoder, Chamfer on constituents | 0.488 | 0.532 | 0.004 |
| **energy-flow graph VAE** (latent 4, 3 seeds) | 0.829 ± 0.006 | 0.805 ± 0.002 | 0.009 |

* Constituent-level graph autoencoders are *inverted* detectors here (signal jets are simpler than QCD);
  reconstructing the IRC-safe energy flow instead fixes this and reaches the level of an oracle density
  estimate of the background (AUC ≈ 0.80) — the ceiling for background-only scores on this signal.
* Shorter training and a tighter bottleneck work better: a better background model also reconstructs the
  signal better.  Latent-space flows add nothing.
* Unsupervised max SIC is only ≈ 1.6; in an idealised weakly supervised search in the signal region the
  GNN's permutation-invariant jet latents plus jet masses carry at least as much signal information as the
  standard (m_J1, Δm_J, τ21_J1, τ21_J2) features (supervised max SIC 15.3 with 8-d latents vs 11.2) and give
  SIC 8–10 for S/B ≳ 2 %.

## Data

LHC Olympics 2020 R&D, Zenodo record [6466204](https://zenodo.org/records/6466204): 1M QCD dijet events +
100k W′(3.5 TeV) → X(500 GeV) Y(100 GeV), X, Y → qq, plus 100k events of a 3-prong variant (X, Y → qqq)
used only as an *unseen* signal.  Events are clustered into anti-kT R = 1.0 jets (pT > 20 GeV) with
scikit-hep `fastjet`; the two leading jets are kept with their top-100 constituents.  Our jets match the
official feature files for 100% of events (relative deviation < 4e-5).

Splits (seeded): background 60 / 10 / 30 % (train / val / test); 2-prong signal 40 / 10 / 50 % (train/val only
for the supervised reference); 3-prong signal test only.  Unsupervised models never see signal.

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
`results/dev/summary.md`).  The test splits and the 3-prong signal are only used for the final numbers.

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
gaehep baselines
gaehep train configs/set_ae.yaml   # train + score + evaluate -> results/runs/set_ae/
gaehep flow set_ae                 # latent-flow variant
gaehep report                      # results/summary.md
```

Data live outside the repo (`GAEHEP_DATA`, default `/home/claude/data/lhco`); results in `results/`.
[`notebooks/kaggle_gaehep.ipynb`](notebooks/kaggle_gaehep.ipynb) runs the full pipeline at GPU scale on Kaggle (100 constituents, wider models, 3 seeds); reruns resume unfinished runs and skip finished ones.

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
