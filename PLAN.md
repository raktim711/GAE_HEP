# Plan: permutation-invariant graph autoencoders for anomaly detection on the LHC Olympics 2020 R&D data

## Context
The AIMS HLT project (repo `raktim711/AIMS-project---Anomaly-Detection`, read in full) trained graph autoencoders on ATLAS HLT objects. Its results were limited in two ways:
- **Data issues.** The background was real data and the signals were MC, which opens a MET shortcut. The event weights were dropped after resampling, and the "jet-pT threshold" cut was really an event selection.
- **Model issues.** On complete graphs, `GCNConv` gives every node exactly the same embedding, since the normalised adjacency is 1/N everywhere. `GINConv` with ε=0 does too (its MLP sees the sum over all nodes). `GATConv` nearly does: the receiving node only enters through the LeakyReLU kink. So 3 of the 4 GAEs reconstructed every object in an event with one shared vector. That is consistent with AUC ≈0.90 regardless of architecture, while SAGE, which keeps a root weight, reached ~30× lower training loss.
  - The features were unscaled, so raw pT dominated the MSE.
  - There was no real bottleneck.
  - Hyper-parameters were tuned on test-set AUC.
  - In graphs-with-edges, "best weights" were never restored, because the code kept a reference to the live `state_dict` rather than a copy.

Goal now (your direction): rebuild the method around **permutation-invariant GNNs** on the **LHC Olympics 2020 R&D** dataset. That dataset is fully simulated and labelled, so we get standard metrics and literature comparisons. Work runs in this cloud session (CPU: 2 vCPU, 7 GB RAM, 30 GB disk, no GPU), with a Kaggle GPU notebook for full-scale reruns. Code goes in a new repo `raktim711/GAE_HEP`, pushed regularly.

## Data
- **Source.** Zenodo record **6466204** (R&D v5): `events_anomalydetection_v2.h5` (2.9 GB, md5 629789d5…), `events_anomalydetection_v2.features.h5` (74 MB), and the 3-prong signal `events_anomalydetection_Z_XY_qqq.h5` (236 MB) with its features file.
- **Contents.** 1M QCD dijets plus 100k W′(3.5 TeV)→X(500)Y(100), with X,Y→qq. The 3-prong signal is a never-tuned-on generalisation test.
- **Reading.** Chunked `pandas.read_hdf(start, stop)`, as the official LHCO examples do, through the PyTables abi3 wheel. The full file is 17 GB in RAM, so this uses 50k-event chunks cast to float32.
- **Jets.** scikit-hep `fastjet` (cp313 wheel), awkward multi-event clustering, anti-kT with R=1.0 and ptmin 20 GeV, as in the official example. Keep the two leading jets by pT.
- **Validation.** Check our jet 4-momenta and masses against the official features file (target ≥99.9% match). Take τ1–τ3 from the features file once jet matching is confirmed; this also fixes the j1/j2 ordering convention.
- **Storage** (outside the repo, `/home/claude/data/lhco/`):
  - a float32 memmap `constituents[n_evt, 2, 100, 3]` (pT, η, φ, top-100 by pT) plus `n_const`;
  - a jet table (pT, η, φ, m, n_const, τ1–3) and an event table (mjj, label, sample).
- **Splits** (seeded, saved). Background 60/10/30 for train/val/test. Both signals go to test only.
  - Model selection uses the background-only validation loss.
  - Anything chosen with labels is reported separately as "oracle".
  - Metrics are given on the full mjj range and on the signal region (3.3–3.7 TeV).

## Graphs and models (all scores permutation-invariant by construction and tested)
- **Graph.** One graph per jet. Nodes are constituents (padded and masked).
  - Node features: Δη, Δφ relative to the jet axis, log pT, log E, log pT/pT_jet, log E/E_jet, ΔR. These are standardised on background-train nodes.
  - Edges: k-NN with k=16 in (Δη, Δφ), then dynamic k-NN in feature space (EdgeConv/DGCNN).
  - No index or position features.
  - Event score = max over the two jets (sum and min also reported), so it doesn't depend on jet order.
- **Implementation.** Dense masked tensors `[B,N,F]` with my own batched k-NN (`cdist`+`topk`) and EdgeConv. This avoids PyG's compiled extensions, which can't be installed here, and is faster on CPU for N≤100.
  - PyG (pure-python) is used only for the legacy port and to cross-check the dense EdgeConv.
- **Models.**
  - **M0 legacy port.** The AIMS `GraphAutoEncoder` (ModelTesting cell 50) and `GAEModel` (graphs-with-edges cell 15) on complete graphs with raw features. This is the baseline, and it demonstrates the GCN/GIN collapse.
  - **M1 node-level GAE (fixed).** EdgeConv encoder, per-node latent narrower than the 7 input features, node-wise decoder. Score = masked mean node MSE.
  - **M2 graph-level set AE (main model).** EdgeConv encoder → masked mean+max pooling → latent z (d=8 or 16) → set decoder generating N̂ particles, plus a multiplicity head.
    - Loss: masked **Chamfer distance**, with Sinkhorn-EMD as a variant. This loss is invariant to both input and output order.
    - Score = reconstruction loss.
  - **M3 latent density.** A normalising flow (zuko) fit on M2's background latents. Score = −log p(z), plus a rank-ensemble with M2.
  - **M4 Deep SVDD.** The same invariant encoder, bias-free, with a fixed centre.
  - **M5 VGAE** (if time allows). Reconstructs the k-NN adjacency plus node features.
  - **Mass decorrelation.** A conditional M2 with the decoder conditioned on m_j and pT_j, plus post-hoc quantile normalisation of the score vs m_j.
  - **References.** Simple observables (m_j, τ21, n_const), and a supervised EdgeConv classifier as the idealised upper bound.
- **Training.** AdamW with a cosine schedule. Early stopping on background validation loss, with best weights deep-copied.
  - Random 200k-jet subsets per epoch.
  - Checkpoint and resume, with time-boxed jobs under 2 h (the background-command limit).
  - At least 3 seeds for headline models.
  - Expected speed: about 2–5k jets/s training and about 10k jets/s scoring on CPU.

## Evaluation
- **Detection metrics.** AUC, the SIC curve, and max SIC (requiring at least 50 background test events past the cut), with bootstrap confidence intervals. Also 1/ε_B at ε_S = 0.3 and 0.5. Reported at jet and event level, on the full range and the SR, for 2-prong and 3-prong.
- **"Does the GNN learn more than jet mass?"** Correlation of each score with m_j, n_const and τ21, and performance after mass decorrelation.
- **Sculpting.** Jensen–Shannon divergence of the background mjj and m_j distributions before vs after cuts at ε_B = 10% and 1%.
- **Plots.** ROC and SIC overlays, score histograms, latent PCA, and η–φ input vs reconstruction. These reuse your rcParams (ModelTesting cell 3) and eta–phi plotter (cell 53).
- **Search demo (stretch, still GNN-based).**
  - Inject N_s ∈ {0, 500, 1000, 2000} signal events into held-out background.
  - Cut on the GNN score, with thresholds set in the sidebands.
  - Fit the 4-parameter dijet function to the mjj sidebands and compute the SR significance.
  - Check there is no fake bump at N_s=0.

## Repo `GAE_HEP` (local at /home/claude/GAE_HEP, pushed to github.com/raktim711/GAE_HEP)
- **Top level:** `pyproject.toml` (package `gaehep`, ruff, pytest), `Makefile` (`setup / data / smoke / test / train-* / eval / report`), `README.md`, `PLAN.md`, `RESULTS.md`, `configs/*.yaml`.
- **`src/gaehep/`:**
  - `data/` — download with md5 and resume, preprocess, splits, memmap dataset, features
  - `graphs/` — k-NN, masked ops
  - `models/` — `legacy_gae`, `edgeconv`, `node_gae`, `set_ae`, `deep_svdd`, `vgae`, `flows`, `classifier`
  - `losses/` — Chamfer, Sinkhorn
  - `train/` — loop, checkpoint
  - `eval/` — metrics, sculpting, bumphunt, plots, style
  - `cli.py`
- **Tests and notebooks:** `tests/` (pytest) and `notebooks/kaggle_gaehep.ipynb` for GPU full scale. Large arrays and checkpoints are git-ignored. Results are stored as JSON and PNG.
- **Environment:** a venv with torch CPU (download.pytorch.org once allowed; otherwise the PyPI wheel, ~3 GB), torch_geometric, fastjet, awkward, vector, tables, h5py, hdf5plugin, scikit-learn, lightgbm, zuko, matplotlib, pyyaml, pytest, ruff.
- **Sonnet subagents** handle the trivial jobs: the literature benchmark table (with citations, which I verify), boilerplate (pyproject, Makefile, README, plot-style module), simple-helper tests, and the Kaggle notebook wrapper. I write and review all model, data and evaluation code.

## Milestones (I report at each)
1. Environment, scaffold, data download and validation, EDA, simple-observable baselines.
2. M0 port with a numerical collapse demo; first M1 and M2 results.
3. Full comparison: M2–M4, decorrelation, seeds, 3-prong, sculpting, then a `RESULTS.md` draft.
4. Stretch: bump hunt, Kaggle notebook, final write-up.

## Needed from you
- **Network.** Apply the Custom network setting: allowed domains `zenodo.org` and `download.pytorch.org`, keeping the defaults. As of now Zenodo is still blocked. I'll do the PyPI-only setup first and poll until it opens.
- **Repo.** Create the empty repo `raktim711/GAE_HEP` with no README and tell me; I'll attach it with push access.

## Verification
- **Unit tests.**
  - Permutation invariance: random node permutations give the same scores to 1e-5, and swapping the jets gives the same event score.
  - Padding invariance.
  - k-NN vs brute force; dense EdgeConv vs PyG EdgeConv.
  - Chamfer properties: zero on identical sets, symmetric, order-free.
  - The GCN/GIN collapse on complete graphs.
  - Metrics against analytic Gaussians.
  - Split disjointness, and the scaler being fit only on background-train (leakage guard).
- **Data checks.** md5 sums, event counts (1M background / 100k signal), and the jet match vs the official features file.
- **Smoke test.** `make smoke`: end-to-end on 5k events before any full run.
- **Sanity checks.**
  - The supervised GNN should be at least as good as any unsupervised model.
  - The m_j baseline should behave as expected.
  - No excess at N_s=0.
  - Headline metrics recomputed from saved scores.
  - A fresh-eyes review before the final report.
