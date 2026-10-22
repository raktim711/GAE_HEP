# GAE_HEP — permutation-invariant graph autoencoders for anomaly detection at the LHC

Unsupervised anomaly detection on jets of the **LHC Olympics 2020 R&D dataset** with graph neural networks
whose anomaly scores are **invariant to the ordering of jet constituents**.  This continues the AIMS project
[`AIMS-project---Anomaly-Detection`](https://github.com/raktim711/AIMS-project---Anomaly-Detection)
(graph autoencoders on ATLAS HLT objects), fixing the problems found there:

* On complete graphs `GCNConv` and `GINConv` (ε = 0) give **every node of a graph the same embedding**
  (verified in `tests/test_legacy_collapse.py`), so the AIMS GCN/GIN autoencoders reconstructed all objects
  of an event with one vector; this repo uses local k-NN graphs and EdgeConv instead.
* Raw, unscaled features let pT dominate the MSE → jet-relative, standardised node features.
* No real bottleneck → graph-level latent (set autoencoder) or a node latent narrower than the node's
  degrees of freedom.
* Hyper-parameters tuned on test-set AUC, best weights never restored → label-free model selection on
  background validation loss, deep-copied best state.
* Data vs MC, prescale weights, selection-driven AUCs → a fully simulated, labelled benchmark with
  literature reference numbers and SIC / sculpting metrics.

See [`PLAN.md`](PLAN.md) for the full plan and [`docs/literature.md`](docs/literature.md) for reference results.

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
| M3 | normalising flow (zuko NSF) on M2 latents | −log p(z) |
| M4 | `deep_svdd` — one-class Deep SVDD, same encoder | distance to centre |
| ref | `classifier` — supervised EdgeConv (idealised upper bound) | logit |
| ref | single observables m_j, τ21, τ32, multiplicity | value |

Event score = max (also min, sum) over the two leading jets.  Post-hoc variants: mass-quantile
normalisation (`gaehep qn`), rank ensembles (`gaehep ensemble`).

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
`notebooks/` will hold a Kaggle/Colab GPU notebook for full-scale runs.

## Layout

```
src/gaehep/data      download, preprocess (fastjet), tables (validation), splits, dataset (graph batches)
src/gaehep/graphs    dense masked k-NN / gather / pooling
src/gaehep/models    edgeconv, set_ae, node_gae, deep_svdd, classifier, legacy_gae, latent_flow
src/gaehep/losses    masked Chamfer, Sinkhorn
src/gaehep/train     training loop (checkpoint / resume / time budget), run orchestration
src/gaehep/eval      metrics (AUC, SIC, rejection, bootstrap), sculpting (JSD, dCor), plots, report
configs/             one YAML per experiment
tests/               invariance, losses, data, metrics, legacy-collapse, training smoke tests
```
