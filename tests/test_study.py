"""``gaehep study-table``: collection, mean +- std over seeds, flagged runs, the significance rule, the paired
event bootstrap of the rank ensembles, and tolerance of missing files.  Everything is built from synthetic
metrics and score files in a temporary results directory (no model, no LHCO data)."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from gaehep import cli
from gaehep.eval import study
from gaehep.eval.evaluate import aggregate
from gaehep.eval.metrics import summarize
from gaehep.eval.study import (
    COLUMNS,
    SIGNALS,
    collect,
    combine,
    ensemble_event_scores,
    paired_bootstrap,
    roc_metrics,
    seed_contrast,
    study_table,
)
from gaehep.train import run as R

N_BKG, N_SIG = 2500, 500
BOXES = {  # box -> (label values of signal events, mjj range of the synthetic events)
    "bb1": ((1,), (3000.0, 5000.0)),
    "bb3": ((1, 2), (3000.0, 5000.0)),
}
# config -> (R&D 2-prong AUCs, 3-prong AUCs, mean jet-score shift of signal on the boxes, best val losses)
CONFIGS = {
    "alpha": ([0.80, 0.81, 0.82], [0.70, 0.71, 0.72], 1.5, [1.0, 1.0, 2.0]),
    "beta": ([0.70, 0.71, 0.72], [0.60, 0.61, 0.62], 0.3, [1.0, 1.0, 1.0]),
    "gamma": ([0.79, 0.81, 0.83], [0.69, 0.71, 0.73], 1.5, [1.0, 1.1, 1.0]),
}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def rd_metrics(name: str, auc2: float, auc3: float, jsd: float) -> dict:
    def entry(auc):
        return {"auc": auc, "max_sic": 10 * auc, "rej@0.3": 100 * auc, "n_sig": 10, "n_bkg": 100}

    return {
        "name": name,
        "metrics": {
            "event": {
                "prod": {
                    "sig_test/SR": entry(auc2),
                    "sig3_test/SR": entry(auc3),
                    "sig_test/full": entry(auc2 - 0.1),
                    "sig3_test/full": entry(auc3 - 0.1),
                }
            },
            "sculpting": {"prod": {"mjj": {"jsd@0.01": jsd}}},
        },
    }


def make_box(rng: np.random.Generator, box: str) -> dict[str, np.ndarray]:
    sig_labels, (lo, hi) = BOXES[box]
    n = N_BKG + N_SIG
    label = np.zeros(n, dtype=np.int8)
    label[N_BKG:] = np.resize(np.array(sig_labels, dtype=np.int8), N_SIG)
    return {
        "event_index": np.arange(n, dtype=np.int64),
        "label": label,
        "mjj": rng.uniform(lo, hi, n).astype(np.float32),
    }


def box_event_metrics(
    sig_key: str, window: tuple[float, float], bb: dict, scores: np.ndarray, ref: np.ndarray
) -> dict:
    ev = aggregate(scores, "prod", ref)
    inside = (bb["mjj"] >= window[0]) & (bb["mjj"] <= window[1])
    out = {}
    for region, sel in (("window", inside), ("full", np.ones_like(inside))):
        s = ev[sel & (bb["label"] > 0)]
        b = ev[sel & (bb["label"] == 0)]
        out[f"{sig_key}/{region}"] = summarize(s, b)
    return {"name": "", "metrics": {"event": {"prod": out}, "jet": {}, "sculpting": {}}}


def write_box_scores(run: Path, name: str, box: str, bb: dict, shift: float, rng, val: np.ndarray) -> None:
    sig = bb["label"] > 0
    jets = rng.normal(0, 1, (len(sig), 2)) + shift * sig[:, None]
    np.savez(
        run / f"{box}_scores.npz",
        jet_scores=jets.astype(np.float32),
        event_index=bb["event_index"],
        mjj=bb["mjj"],
        label=bb["label"],
        subsample=np.array(True),
    )
    sig_def = next(s for s in SIGNALS if s.short == box)
    ref = np.sort(val.astype(np.float64).ravel())
    m = box_event_metrics(sig_def.key, sig_def.window, bb, jets.astype(np.float32), ref)
    m["name"] = name
    write_json(run / f"{box}_metrics.json", m)


def build_tree(results: Path) -> dict[str, dict]:
    """study_{alpha,beta,gamma}_s{1,2,3}, their _ens runs, and the unseeded study_efpknn."""
    rng = np.random.default_rng(1)
    bbs = {box: make_box(rng, box) for box in BOXES}
    info = {}
    for cfg, (a2, a3, shift, losses) in CONFIGS.items():
        for i, seed in enumerate((1, 2, 3)):
            name = f"study_{cfg}_s{seed}"
            run = results / "runs" / name
            val = rng.normal(0, 1, (2000, 2)).astype(np.float32)
            run.mkdir(parents=True)
            np.savez(run / "scores.npz", bkg_val=val)
            write_json(run / "metrics.json", rd_metrics(name, a2[i], a3[i], 0.1 * (i + 1)))
            write_json(run / "history.json", {"status": "done", "best_val_loss": losses[i]})
            (run / "config.yaml").write_text(
                yaml.safe_dump({"model": "ef_ae", "model_args": {"encoder": cfg}})
            )
            write_json(
                run / "probes.json",
                {"targets": {"m_j": {"ridge": {"r2_mean": 0.5 + 0.1 * i}, "mlp": {"r2_mean": 0.9}}}},
            )
            write_json(
                run / "irc_probe.json",
                {
                    "auc": 0.8,
                    "split all": {"auc": 0.75, "rho": 0.9},
                    "knn_ties": {"bkg": {"jet_fraction": 0.001}},
                },
            )
            for box in BOXES:
                write_box_scores(run, name, box, bbs[box], shift, rng, val)
        ens = results / "runs" / f"study_{cfg}_ens"
        write_json(ens / "metrics.json", rd_metrics(f"study_{cfg}_ens", a2[-1] + 0.05, a3[-1] + 0.05, 0.05))
        write_json(
            ens / "bb1_metrics.json",
            box_event_metrics(
                "bb1",
                (3600.0, 4000.0),
                bbs["bb1"],
                rng.normal(size=(len(bbs["bb1"]["label"]), 2)).astype(np.float32),
                np.sort(rng.normal(size=4000)),
            ),
        )
        info[cfg] = {"a2": a2, "a3": a3, "losses": losses}
    single = results / "runs" / "study_efpknn"
    write_json(single / "metrics.json", rd_metrics("study_efpknn", 0.9, 0.85, 0.2))
    (single / "config.yaml").write_text("model: efp_knn\n")
    return info


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("study")
    build_tree(root)
    return root


@pytest.fixture(scope="module")
def configs(results):
    return {c.label: c for c in collect("study", results)}


@pytest.fixture(scope="module")
def tables(results, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    paths = study_table("study", results, out, n_boot=40, log=lambda *a: None)
    with open(paths["table.csv"], newline="") as fh:
        rows = {r["label"]: r for r in csv.DictReader(fh)}
    return {
        "paths": paths,
        "md": paths["table.md"].read_text(),
        "rows": rows,
        "contrasts": json.loads(paths["contrasts.json"].read_text()),
    }


# ----------------------------------------------------------------------------------------- the rule
def test_seed_contrast_applies_two_standard_errors():
    # delta = 0.02, SE = sqrt(1e-4/3 + 1e-4/3) = 0.00816 -> 2 SE = 0.0163 < 0.02
    c = seed_contrast([0.80, 0.81, 0.82], [0.78, 0.79, 0.80])
    assert c["delta"] == pytest.approx(0.02) and c["se"] == pytest.approx(np.sqrt(2e-4 / 3))
    assert c["significant_seed"] is True and (c["n_a"], c["n_b"]) == (3, 3)
    # delta = 0.01, SE = sqrt(2 * 4e-4 / 3) = 0.0163 -> 2 SE = 0.033 > 0.01
    c = seed_contrast([0.80, 0.82, 0.84], [0.79, 0.81, 0.83])
    assert c["significant_seed"] is False and c["delta"] == pytest.approx(0.01)
    # the sign does not matter, the sample standard deviation (ddof = 1) is used, one seed is undetermined
    assert seed_contrast([0.78, 0.79, 0.80], [0.80, 0.81, 0.82])["delta"] == pytest.approx(-0.02)
    one = seed_contrast([0.8], [0.7, 0.71])
    assert (
        one["se"] is None and one["significant_seed"] is None and one["delta"] == pytest.approx(0.8 - 0.705)
    )


@pytest.mark.parametrize(
    ("seed", "boot", "want"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
        (None, True, None),
        (True, None, None),
        (None, None, None),
        (None, False, False),
        (False, None, False),
    ],
)
def test_combine_needs_both_requirements(seed, boot, want):
    assert combine(seed, boot) is want


# --------------------------------------------------------------------------------- bootstrap pieces
def test_roc_metrics_is_summarize_from_one_roc():
    rng = np.random.default_rng(0)
    for tied in (False, True):
        s, b = rng.normal(0.8, 1, 300), rng.normal(0, 1, 3000)
        if tied:
            s, b = np.round(s), np.round(b)
        want = summarize(s, b)
        got = roc_metrics(s, b)
        assert got == pytest.approx((want["auc"], want["max_sic"], want["rej@0.3"]), rel=1e-12)
    assert all(np.isnan(roc_metrics(np.array([]), b)))


def test_ensemble_event_scores_follow_the_ensemble_command(results, configs):
    runs = configs["alpha"].runs
    got = ensemble_event_scores(runs, "bb1")
    acc_jet = acc_val = 0.0
    for r in runs:
        with np.load(r.dir / "scores.npz") as f:
            val = f["bkg_val"]
        with np.load(r.dir / "bb1_scores.npz") as f:
            jets = f["jet_scores"]
        ref = R._ecdf_ref(val.ravel())
        acc_jet = acc_jet + R._ecdf(ref, jets)
        acc_val = acc_val + R._ecdf(ref, val)
    ens_jet, ens_val = (acc_jet / 3).astype(np.float32), (acc_val / 3).astype(np.float32)
    want = aggregate(ens_jet, "prod", np.sort(ens_val.astype(np.float64).ravel()))
    assert np.array_equal(got["score"], want)
    with np.load(runs[0].dir / "bb1_scores.npz") as f:
        assert np.array_equal(got["label"], f["label"]) and np.array_equal(got["mjj"], f["mjj"])


def test_ensemble_event_scores_report_why_they_cannot_be_built(results, configs, tmp_path):
    runs = configs["beta"].runs
    # a different event table in one seed
    other = tmp_path / "other"
    other.mkdir()
    shutil.copy(runs[0].dir / "scores.npz", other / "scores.npz")
    with np.load(runs[0].dir / "bb1_scores.npz") as f:
        cols = {k: f[k] for k in f.files}
    cols["label"] = cols["label"][::-1].copy()
    np.savez(other / "bb1_scores.npz", **cols)
    bad = study.Run("other", other)
    msg = ensemble_event_scores([runs[0], bad], "bb1")
    assert isinstance(msg, str) and "different events" in msg
    assert "cannot read" in ensemble_event_scores([study.Run("nothing", tmp_path / "nothing")], "bb1")


def test_paired_bootstrap_is_seeded_paired_and_restricted_to_the_window(results, configs):
    sig = next(s for s in SIGNALS if s.short == "bb1")
    boxes = {k: ensemble_event_scores(configs[k].runs, "bb1") for k in ("alpha", "beta")}
    boxes["alpha_copy"] = boxes["alpha"]
    obs, boot = paired_bootstrap(boxes, sig, n_boot=25)
    _obs2, boot2 = paired_bootstrap(boxes, sig, n_boot=25)
    assert all(np.array_equal(boot[k], boot2[k]) for k in boot)
    assert boot["alpha"].shape == (25, 3) and np.array_equal(
        boot["alpha"], boot["alpha_copy"]
    )  # same resamples
    assert not np.array_equal(paired_bootstrap(boxes, sig, n_boot=25, seed=1)[1]["alpha"], boot["alpha"])
    # the observed numbers are those of the window events
    b = boxes["alpha"]
    lo, hi = sig.window
    inside = (b["mjj"] >= lo) & (b["mjj"] <= hi)
    want = roc_metrics(b["score"][inside & (b["label"] == 1)], b["score"][inside & (b["label"] == 0)])
    assert obs["alpha"] == pytest.approx(want)
    assert obs["alpha"][0] > obs["beta"][0] + 0.1 and np.nanmean(boot["alpha"][:, 0]) == pytest.approx(
        want[0], abs=0.03
    )


# ------------------------------------------------------------------------------------- collection
def test_collect_groups_seeds_ensembles_and_single_runs(results, configs):
    assert set(configs) == {"alpha", "beta", "gamma", "efpknn"}
    a = configs["alpha"]
    assert [r.name for r in a.runs] == ["study_alpha_s1", "study_alpha_s2", "study_alpha_s3"]
    assert a.ens is not None and a.ens.name == "study_alpha_ens" and a.kind == "alpha"
    e = configs["efpknn"]
    assert [r.name for r in e.runs] == ["study_efpknn"] and e.ens is None and e.kind == "efp_knn"
    assert [c.label for c in collect("study_alpha", results)] == [
        "study_alpha"
    ]  # nothing left after the prefix
    assert collect("nothing", results) == []


def test_collect_skips_extra_score_runs_and_runs_without_files(results, tmp_path):
    shutil.copytree(results, tmp_path / "r")
    (tmp_path / "r" / "runs" / "study_alpha_s1+kl").mkdir()
    write_json(tmp_path / "r" / "runs" / "study_alpha_s1+kl" / "metrics.json", rd_metrics("x", 0.5, 0.5, 0.1))
    (tmp_path / "r" / "runs" / "study_empty_s1").mkdir()
    assert {c.key for c in collect("study", tmp_path / "r")} == {
        "study_alpha",
        "study_beta",
        "study_gamma",
        "study_efpknn",
    }


def test_flagged_runs_are_above_one_and_a_half_times_the_median(configs):
    assert configs["alpha"].flagged == ["study_alpha_s3"]  # 2.0 > 1.5 x median 1.0
    assert configs["beta"].flagged == [] and configs["gamma"].flagged == []
    assert study.flagged_runs([]) == []
    neg = [
        study.Run(f"r{i}", Path("."), history={"best_val_loss": v}) for i, v in enumerate((-3.0, -3.0, 5.0))
    ]
    assert study.flagged_runs(neg) == []  # the rule needs a positive median


# ------------------------------------------------------------------------------------------- table
def test_table_means_stds_and_ensembles(tables):
    rows = tables["rows"]
    assert set(rows) == {"alpha", "beta", "gamma", "efpknn"}
    a = rows["alpha"]
    assert a["configuration"] == "study_alpha" and a["kind"] == "alpha" and a["n_runs"] == "3"
    assert float(a["rd2_SR_auc_mean"]) == pytest.approx(0.81) and float(a["rd2_SR_auc_std"]) == pytest.approx(
        0.01
    )
    assert float(a["rd3_SR_max_sic_mean"]) == pytest.approx(7.1) and float(
        a["rd3_full_auc_mean"]
    ) == pytest.approx(0.61)
    assert float(a["rd2_SR_auc_ens"]) == pytest.approx(0.87)
    assert float(a["jsd_mean"]) == pytest.approx(0.2) and float(a["jsd_ens"]) == pytest.approx(0.05)
    assert float(a["val_loss_mean"]) == pytest.approx(4 / 3)
    assert a["n_flagged"] == "1" and rows["beta"]["n_flagged"] == "0" and a["has_ensemble"] == "1"
    assert float(a["probe_mlp_m_j_mean"]) == pytest.approx(0.9) and float(
        a["probe_ridge_m_j_mean"]
    ) == pytest.approx(0.6)
    assert float(a["irc_rho_split all_mean"]) == pytest.approx(0.9)
    assert float(a["irc_dauc_split all_mean"]) == pytest.approx(-0.05)
    assert float(a["irc_ties_mean"]) == pytest.approx(0.001)
    e = rows["efpknn"]
    assert float(e["rd2_SR_auc_mean"]) == 0.9 and e["rd2_SR_auc_std"] == "" and e["rd2_SR_auc_ens"] == ""
    assert e["bb1_window_auc_mean"] == "" and e["has_ensemble"] == "0"
    # the BB numbers come from bb<i>_metrics.json: the shifted signal of alpha is found, beta's is nearly hidden
    assert float(a["bb1_window_auc_mean"]) > 0.8 > 0.65 > float(rows["beta"]["bb1_window_auc_mean"]) > 0.5
    assert float(a["bb1_full_auc_mean"]) > 0.8 and a["bb1_window_auc_ens"] != ""


def test_markdown_shows_cells_and_flags(tables):
    md = tables["md"]
    assert "# Study table: `study`" in md and "## AUC in the signal window" in md and "## IRC probe" in md
    assert "0.810 ± 0.010 [0.870]" in md  # mean ± std [ensemble]
    assert "| efpknn | 1 | 0 | 0.900 |" in md  # one run: no std, no ensemble
    assert "`study_alpha_s3`" in md.split("## Flagged runs")[1]
    assert "R&D 2-prong (tuned)" in md


def test_columns_cover_every_signal_region_and_metric():
    for sig in SIGNALS:
        for region in (sig.primary, "full"):
            for metric in ("auc", "max_sic", "rej@0.3"):
                assert f"{sig.short}_{region}_{metric}" in COLUMNS
    assert [s.window for s in SIGNALS if s.window] == [(3600.0, 4000.0), (4000.0, 4400.0)]


# ---------------------------------------------------------------------------------------- contrasts
def contrast(tables, a: str, b: str, sig: str, metric: str) -> dict:
    return tables["contrasts"]["contrasts"][f"{a} - {b}"][sig][metric]


def test_contrasts_cover_every_ordered_pair_signal_and_metric(tables):
    c = tables["contrasts"]["contrasts"]
    labels = ["alpha", "beta", "efpknn", "gamma"]
    assert {f"{a} - {b}" for a in labels for b in labels if a != b} == set(c)
    ab = c["alpha - beta"]
    assert set(ab) == {"sig_test/SR", "sig3_test/SR", "bb1/window", "bb3/window", "bkg_test/sculpting"}
    assert set(ab["bb1/window"]) == {"auc", "max_sic", "rej@0.3"} and set(ab["bkg_test/sculpting"]) == {
        "jsd@0.01"
    }
    meta = tables["contrasts"]["meta"]
    assert meta["bootstrap"]["n_boot"] == 40 and meta["bootstrap"]["seed"] == 20201
    assert meta["configurations"]["alpha"]["flagged"] == ["study_alpha_s3"]
    assert meta["windows_gev"] == {"bb1": [3600.0, 4000.0], "bb3": [4000.0, 4400.0]}


def test_contrast_on_the_rd_signals_is_the_seed_rule(tables):
    e = contrast(tables, "alpha", "beta", "sig3_test/SR", "auc")
    assert e["delta"] == pytest.approx(0.10) and e["se"] == pytest.approx(0.01 * np.sqrt(2 / 3))
    assert e["significant_seed"] is True and e["significant"] is True and "bootstrap" not in e
    e = contrast(tables, "alpha", "gamma", "sig3_test/SR", "auc")  # 0.71 vs 0.71: |delta| = 0 < 2 SE
    assert e["delta"] == pytest.approx(0.0, abs=1e-12) and e["significant"] is False
    e = contrast(tables, "alpha", "gamma", "bkg_test/sculpting", "jsd@0.01")
    assert e["significant"] is False and e["delta"] == pytest.approx(0.0)


def test_contrast_on_black_boxes_needs_the_bootstrap_too(tables):
    e = contrast(tables, "alpha", "beta", "bb1/window", "auc")
    b = e["bootstrap"]
    assert e["delta"] > 0.1 and b["delta_ensemble"] > 0.1 and b["lo"] > 0 and b["excludes_zero"] is True
    assert b["n_boot"] == 40 and b["n_finite"] <= 40 and b["lo"] <= b["hi"]
    assert e["significant"] is combine(e["significant_seed"], b["excludes_zero"])
    # same architecture, same shift: the intervals of the two ensembles cover zero
    g = contrast(tables, "alpha", "gamma", "bb3/window", "auc")["bootstrap"]
    assert g["excludes_zero"] is False and g["lo"] < 0 < g["hi"]
    assert contrast(tables, "alpha", "gamma", "bb3/window", "auc")["significant"] is False


def test_contrast_of_the_reverse_pair_is_the_mirror_image(tables):
    for sig, metric in (("bb1/window", "auc"), ("bb3/window", "rej@0.3"), ("sig_test/SR", "max_sic")):
        f, r = contrast(tables, "alpha", "beta", sig, metric), contrast(tables, "beta", "alpha", sig, metric)
        assert r["delta"] == pytest.approx(-f["delta"]) and r["se"] == pytest.approx(f["se"])
        assert r["significant"] == f["significant"]
        if f.get("bootstrap"):
            assert r["bootstrap"]["lo"] == pytest.approx(-f["bootstrap"]["hi"])
            assert r["bootstrap"]["hi"] == pytest.approx(-f["bootstrap"]["lo"])


def test_unseeded_configuration_is_undetermined(tables):
    e = contrast(tables, "alpha", "efpknn", "sig_test/SR", "auc")
    assert e["n_b"] == 1 and e["se"] is None and e["significant_seed"] is None and e["significant"] is None
    assert "bb1/window" not in tables["contrasts"]["contrasts"]["alpha - efpknn"]  # efpknn has no BB numbers


# ------------------------------------------------------------------------------- missing and odd files
def test_missing_files_are_tolerated(results, tmp_path):
    root = tmp_path / "r"
    shutil.copytree(results, root)
    runs = root / "runs"
    for p in runs.glob("study_*_s*/probes.json"):
        p.unlink()
    (runs / "study_beta_s2" / "bb1_scores.npz").unlink()  # beta's ensemble cannot be rebuilt on BB1
    for p in runs.glob("study_*/bb3_*"):
        p.unlink()  # no BB3 at all
    (runs / "study_gamma_s1" / "history.json").write_text("not json")
    (runs / "study_gamma_s2" / "irc_probe.json").write_text("")
    shutil.rmtree(runs / "study_gamma_s3")
    paths = study_table("study", root, tmp_path / "o", n_boot=20, log=lambda *a: None)
    with open(paths["table.csv"], newline="") as fh:
        rows = {r["label"]: r for r in csv.DictReader(fh)}
    assert rows["gamma"]["n_runs"] == "2" and rows["alpha"]["probe_mlp_m_j_mean"] == ""
    assert rows["alpha"]["bb3_window_auc_mean"] == ""
    c = json.loads(paths["contrasts.json"].read_text())
    assert "bb3/window" not in c["contrasts"]["alpha - beta"]
    ab = c["contrasts"]["alpha - beta"]["bb1/window"]["auc"]
    assert ab["bootstrap"] is None and ab["significant_seed"] is True and ab["significant"] is None
    assert "bb1_scores.npz" in c["meta"]["bootstrap_unavailable"]["bb1"]["beta"]
    assert "bb1" not in {k for k, v in c["meta"]["bootstrap_unavailable"].items() if "alpha" in v}
    assert "–" in paths["table.md"].read_text()


def test_runs_trained_on_a_black_box_skip_the_bootstrap(results, tmp_path):
    root = tmp_path / "r"
    shutil.copytree(results, root)
    for p in (root / "runs").glob("study_beta_s*/config.yaml"):
        p.write_text(yaml.safe_dump({"model": "ef_ae", "data": {"train_sample": "bb1"}}))
    paths = study_table("study", root, tmp_path / "o", n_boot=20, log=lambda *a: None)
    c = json.loads(paths["contrasts.json"].read_text())
    assert "black box" in c["meta"]["bootstrap_unavailable"]["bb1"]["beta"]
    assert c["contrasts"]["alpha - beta"]["bb1/window"]["auc"]["bootstrap"] is None


# ------------------------------------------------------------------------------------------------ CLI
def test_cli_study_table_writes_to_results_study(results, monkeypatch):
    monkeypatch.setenv("GAEHEP_RESULTS", str(results))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert cli.main(["study-table", "study", "--n-boot", "20"]) == 0
    files = {Path(line).name for line in out.getvalue().split() if line.endswith(("md", "csv", "json"))}
    assert files == {"study_table.md", "study_table.csv", "study_contrasts.json"}
    assert all((results / "study" / f).exists() for f in files)
    with pytest.raises(SystemExit, match="no runs named"):
        cli.main(["study-table", "nonexistent"])
