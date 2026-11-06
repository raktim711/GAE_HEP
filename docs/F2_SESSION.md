# Session F2: black-box checks of the bias-audit follow-up (Deviations D9)

The bias audit of 6 November left four checks that need the LHC Olympics black boxes (Zenodo 4536624). The session
that ran the rest of D9 could not reach zenodo.org, so they are collected here for one session that can. Every
command below exists on the branch `claude/results-draft` (code in `src/gaehep/eval/followup.py`, tested on
synthetic black boxes in `tests/test_followup.py`).

| step | settles | needs | CPU (2 cores) |
|---|---|---|---|
| 1 | the inputs × pooling 2×2 on BB1 and BB3 (finding B2) | the six `e5_*` runs | ~2.5 h, mostly the three EWMP runs |
| 2 | blind version of the BB1 bump hunt: does the largest excess sit at the true window? (B5) | the three EFN BB1-trained runs | ~45 min |
| 3 | procedure-matched null: the same hunt on BB2 with models trained on BB2 (B5) | three new EFN runs | ~1.5 h |
| 4 | BB1 AUC at fixed jet masses, and whether "worse fit → better BB1" survives there (B4) | committed weights | ~1.3 h for the 35 E1 runs, ~4 h for all 98 E1 + E2 runs |
| 5 (optional) | Figure 6's encoder bar over all 35 E1 runs instead of seed 1 (B7) | committed weights | ~2.5 h |

Steps 2 and 3 matter most for the paper: together with the R&D null of D9 they decide what can be said about the
BB1 significance.

## Before starting

- Start the session by hand (not as a scheduled task) in an environment whose network settings allow
  `zenodo.org`; scheduled sessions could not reach it before.
- Setup takes 2–3 h (download and processing of three black boxes, 2.5 GB each). Check `df -h` first: about 9 GB
  must be free while a box is processed.

## Prompt to paste

> Work in raktim711/GAE_HEP. Check out `claude/results-draft`, create the branch `claude/f2-blackbox-followup` from
> it, and run `docs/F2_SESSION.md` steps 1–4 (step 5 only if time is left), following the protocol's scheduling and
> commit rules (section 8): single-thread jobs in the background, as many in parallel as `nproc`, never commit data,
> scores or latents. Every `gaehep dev` command keeps `--set train.epochs=2`. When done, run `gaehep audit-followup`
> and `python scripts/make_paper_figs.py`, commit the tracked outputs listed in the doc, push, and open a pull request
> into `claude/results-draft` (or into `main` if that branch has been merged). Do not merge it.

## Setup

```bash
make setup                        # venv with the pinned packages
make data                         # R&D sample (Zenodo 6466204)
make data-blackbox                # BB1-BB3: download, preprocess (--delete-raw), tables, splits
gaehep tau --sample bb1 --workers 2   # computed N-subjettiness (D3): only for the -tau21 reference of step 4
python -c "from gaehep.config import splits_sha256; print(splits_sha256())"   # must print 5217cca5…
```

## Step 1: the E5 runs on BB1 and BB3

```bash
for r in e5_efn_meanmax_s{1,2,3} e5_ewmp_meanmax_s{1,2,3}; do
  gaehep prepare-run $r                       # committed weights -> run dir, bkg_val scores -> scores.npz
  gaehep score-sample $r --sample bb1 --subsample
  gaehep score-sample $r --sample bb3 --subsample
done
```

## Step 2: blind BB1 scan of the BB1-trained runs

```bash
for s in 1 2 3; do
  r=study_bb1data_efn_s$s
  gaehep prepare-run $r                       # F_B from BB1's validation part, as in E3
  gaehep score-sample $r --sample bb1         # all 1 000 000 events, no --subsample
  gaehep bb-scan $r --sample bb1              # 14 windows x 3 cuts + 1000 toys -> bb1_scan.json
done
```

## Step 3: the procedure-matched null on BB2

```bash
for s in 1 2 3; do
  gaehep dev configs/study/efn.yaml --name std_bb2data_efn_s$s --set train.epochs=2 --set train.seed=$s \
    --set train.threads=1 --set data.train_sample=bb2
  gaehep promote std_bb2data_efn_s$s study_bb2data_efn_s$s
  gaehep score-sample study_bb2data_efn_s$s --sample bb2       # all events
  gaehep bb-scan study_bb2data_efn_s$s --sample bb2            # -> bb2_scan.json
done
```

## Step 4: BB1 at fixed jet masses

```bash
RUNS=$(ls results/runs | grep -E '^study_(deepsets|efn|edgeconv|ewmp|transformer|ew_attention|gcn_complete)_s[1-5]$')
gaehep bb1-mass-check $RUNS                   # -> <run>/bb1_mass.json; add the stct_* runs if time allows
```

## Step 5 (optional): BB3 diagnostics over all E1 runs

```bash
for r in $RUNS; do gaehep prepare-run $r; gaehep score-sample $r --sample bb3 --subsample; done
gaehep bb-diagnose $RUNS --sample bb3         # rewrites results/study/bb3_diagnostics.{md,json}
```

## Report and commit

```bash
gaehep audit-followup                         # adds sections 6 (black-box scans) and 7 to audit_followup.md
python scripts/make_paper_figs.py
make test lint
```

Commit:
- the tracked files of `results/runs/<run>/`: `bb*_metrics.json`, `bb*_scan.json`, `bb1_mass.json`, `config.yaml`,
  `history.json`, `metrics.json`;
- `results/dev/std_bb2data_*`, `results/study/*`, `paper/figures/*`;
- the BB2-trained weights, force-added as `results/models/study_bb2data_efn_s<s>/model.pt`.

Never commit `*_scores*.npz`, `scores.npz`, `latents*.npz` or `model.pt` inside `results/runs/`.

Then add one paragraph with the step 1–4 results under D9 in `docs/STUDY_PROTOCOL.md`, update the numbers in
`paper/results_draft.md` (sections 2, 5 and the Limitations) and in `results/PAPER_CONTEXT.md`, and open the PR.
