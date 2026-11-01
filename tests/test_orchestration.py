"""Orchestration on a tiny synthetic dataset: run / resumable scoring / Data cache, post-hoc variants,
report, reconstruction and EDA figures, and the command line.

Everything shares one module-scoped fake dataset and one trained model (CPU, a few seconds in total).
``GAEHEP_DATA`` / ``GAEHEP_RESULTS`` are set through a module-scoped ``MonkeyPatch.context`` so they never
leak into other test modules (``config.data_dir`` / ``results_dir`` read the environment on every call).
"""

from __future__ import annotations

import contextlib
import copy
import io
import re
import shutil
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pytest
import torch
import yaml

from gaehep import cli
from gaehep.config import SR_MJJ, load_json
from gaehep.data.fake import make_fake
from gaehep.eval import eda, recon, report  # imported here: their style.apply() runs before rc_context
from gaehep.models import build_model
from gaehep.train import loop
from gaehep.train import run as R
from gaehep.train.run import SCORE_SPLITS, Data, run

N_BKG, N_SIG, N_MAX = 600, 150, 30
LAT_EVENTS = 100  # bkg_train events whose latents are saved
CFG = {  # 1-epoch set AE in the style of configs/smoke.yaml, shrunk further
    "model": "set_ae",
    "model_args": {"hidden": [8, 8], "k": 4, "latent_dim": 2, "n_out": N_MAX, "dec_hidden": [16]},
    "data": {"n_const": N_MAX, "latent_train_events": LAT_EVENTS},
    "train": {
        "epochs": 1,
        "jets_per_epoch": 200,
        "batch_size": 64,
        "val_jets": 100,
        "prepare_batches": 2,
        "threads": 1,
        "device": "cpu",
    },
}
FIGURES = ("rejection_SR.png", "sic_SR.png", "score_hist.png", "sculpting_mjj.png", "history.png")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# Figures are only checked to be valid PNGs: no second 'tight' layout pass, no minor ticks, low dpi (halves
# the plotting time, which dominates this module).  Restored when the module's fixtures are torn down.
FAST_RC = {
    "savefig.dpi": 40,
    "savefig.bbox": "standard",
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
}


def quiet(*a: object) -> None:
    pass


def is_png(p: Path) -> bool:
    return p.is_file() and p.stat().st_size > 1000 and p.read_bytes()[:8] == PNG_MAGIC


def load_npz(p: Path) -> dict[str, np.ndarray]:
    with np.load(p) as f:
        return {k: f[k] for k in f.files}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Fake processed data + empty results dir, exported for the whole module only."""
    root = tmp_path_factory.mktemp("orch")
    make_fake(root / "data", N_BKG, N_SIG, N_MAX)
    with pytest.MonkeyPatch.context() as mp, mpl.rc_context(FAST_RC):
        mp.setenv("GAEHEP_DATA", str(root / "data"))
        mp.setenv("GAEHEP_RESULTS", str(root / "results"))
        yield root


@pytest.fixture(scope="module")
def runs(env) -> Path:
    return env / "results" / "runs"


@pytest.fixture(scope="module")
def trained(runs):
    """Run 'a' through the default stages once; keep its first outputs for later comparisons."""
    logs: list[str] = []
    summary = run(copy.deepcopy(CFG), "a", log=lambda *a: logs.append(" ".join(map(str, a))))
    return {
        "summary": summary,
        "logs": logs,
        "scores": load_npz(runs / "a" / "scores.npz"),
        "latents": load_npz(runs / "a" / "latents.npz"),
        "model": (runs / "a" / "model.pt").read_bytes(),
    }


@pytest.fixture(scope="module")
def cli_run(env, runs):
    """`gaehep train` end to end (all stages) with --set overrides; a second trained run for the report."""
    cfg_path = env / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(CFG))
    argv = ["train", str(cfg_path), "--name", "cli", "--set", "train.seed=2"]
    argv += ["--set", "model_args.latent_dim=3", "--set", "data.latent_train_events=50"]
    argv += ["--set", "train.lr=1e-3"]  # the default lr; used to reach AdamW as the string '1e-3'
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = cli.main(argv)
    return rc, out.getvalue()


# ----------------------------------------------------------------------------------------------- run()
def test_run_default_stages_outputs(trained, runs):
    """run() with default stages must leave finite float32 (n, 2) scores per split, metrics, figures, no _partial."""
    assert trained["summary"]["status"] == "done"
    d = Data(N_MAX)
    rd = runs / "a"
    sc = trained["scores"]
    assert set(sc) == set(SCORE_SPLITS)
    for split, s in sc.items():
        assert s.shape == (len(d.events(split)), 2) and s.dtype == np.float32, split
        assert np.isfinite(s).all(), split
    lat = trained["latents"]
    assert set(lat) == {*SCORE_SPLITS, "bkg_train"}
    assert lat["bkg_train"].shape == (LAT_EVENTS, 2, CFG["model_args"]["latent_dim"])
    assert all(v.dtype == np.float32 and np.isfinite(v).all() for v in lat.values())
    m = load_json(rd / "metrics.json")
    assert m["name"] == "a" and m["config"]["model"] == "set_ae"
    assert 0.0 <= m["headline"]["auc_2p_full"] <= 1.0 and "auc_3p_full" in m["headline"]
    assert set(m["metrics"]["event"]) == {"max", "min", "sum", "prod"}
    for f in FIGURES:
        assert is_png(rd / "figures" / f), f
    assert yaml.safe_load((rd / "config.yaml").read_text()) == CFG
    assert not (rd / "_partial").exists()
    assert not list(rd.glob("*.tmp*"))


def test_rerun_skips_training(trained, runs, monkeypatch):
    """A finished run must not be retrained on rerun, and rescoring the saved best weights reproduces scores."""

    def no_fit(*a, **k):
        raise AssertionError("fit() called for a finished run")

    monkeypatch.setattr(R, "fit", no_fit)
    logs: list[str] = []
    s = run(copy.deepcopy(CFG), "a", stages=("train", "score"), log=lambda *a: logs.append(str(a[0])))
    assert s["status"] == "done" and any("already trained" in x for x in logs)
    assert (runs / "a" / "model.pt").read_bytes() == trained["model"]
    again = load_npz(runs / "a" / "scores.npz")
    assert all(np.array_equal(again[k], v) for k, v in trained["scores"].items())


def test_score_stage_resumes_from_partial(trained, runs, monkeypatch):
    """An interrupted score stage must keep finished splits in _partial and score only the rest on restart."""
    src, dst = runs / "a", runs / "resume"
    dst.mkdir(parents=True)
    for f in ("model.pt", "history.json"):
        shutil.copy(src / f, dst / f)
    real = R.score_pool
    calls: list = []

    def flaky(model, pool, *a, **k):
        calls.append(pool)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return real(model, pool, *a, **k)

    monkeypatch.setattr(R, "score_pool", flaky)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(copy.deepcopy(CFG), "resume", stages=("train", "score"), log=quiet)
    part = dst / "_partial"
    assert sorted(p.name for p in part.iterdir()) == ["bkg_val.npz"]  # no half-written .tmp left either
    assert not (dst / "scores.npz").exists()

    calls.clear()

    def counting(model, pool, *a, **k):
        calls.append(pool)
        return real(model, pool, *a, **k)

    monkeypatch.setattr(R, "score_pool", counting)
    d = Data(N_MAX)
    run(copy.deepcopy(CFG), "resume", stages=("train", "score"), data=d, log=quiet)
    expected = [("bkg_test", None), ("sig_test", None), ("sig3_test", None), ("bkg_train", LAT_EVENTS)]
    assert len(calls) == len(expected)
    for pool, (split, cap) in zip(calls, expected, strict=True):
        assert np.array_equal(np.unique(pool.event), d.events(split, cap)), split
    assert d._pools == {}  # every scored pool released again
    assert not part.exists()
    sc, lat = load_npz(dst / "scores.npz"), load_npz(dst / "latents.npz")
    assert set(sc) == set(trained["scores"]) and set(lat) == set(trained["latents"])
    assert all(np.array_equal(sc[k], v) for k, v in trained["scores"].items())
    assert all(np.array_equal(lat[k], v) for k, v in trained["latents"].items())


# --------------------------------------------------------------------------- Data cache, eval batch size
def test_data_release_drops_only_its_key(env):
    """Data.release must drop exactly the pool cached by the same (split, cap, mjj) so memory is freed."""
    d = Data(N_MAX)
    p = d.pool("bkg_val")
    assert d.pool("bkg_val") is p and "bkg_val:None:None" in d._pools
    w = d.pool("bkg_val", mjj=SR_MJJ)
    assert len(w) < len(p)
    d.release("bkg_val")
    assert "bkg_val:None:None" not in d._pools and len(d._pools) == 1  # windowed pool untouched
    p2 = d.pool("bkg_val")
    assert p2 is not p and np.array_equal(p2.const, p.const) and np.array_equal(p2.event, p.event)
    d.release("bkg_val", mjj=SR_MJJ)
    d.release("bkg_val")
    d.release("sig_val", 5)  # never cached: no error
    assert d._pools == {}


def test_eval_batch_size_attribute_is_the_default(pool, scaler, monkeypatch):
    """score_pool / mean_loss must use a model's eval_batch_size (ewmp: 256) when batch_size is None."""
    seen: list[int] = []
    real = loop.iterate_batches

    def spy(pool_, scaler_, batch_size, *a, **k):
        seen.append(batch_size)
        return real(pool_, scaler_, batch_size, *a, **k)

    monkeypatch.setattr(loop, "iterate_batches", spy)
    torch.manual_seed(0)
    ew = build_model("ef_ae", encoder="ewmp", hidden=(8, 8), latent_dim=2, n_out=6, dec_hidden=(16,))
    assert ew.eval_batch_size == 256
    s_default = loop.score_pool(ew, pool, scaler)["score"]
    loop.mean_loss(ew, pool, scaler, np.arange(len(pool)))
    s_small = loop.score_pool(ew, pool, scaler, batch_size=7)["score"]
    sa = build_model("set_ae", hidden=(8,), k=4, latent_dim=2, n_out=30, dec_hidden=(16,))
    assert not hasattr(sa, "eval_batch_size")
    loop.score_pool(sa, pool, scaler)
    loop.mean_loss(sa, pool, scaler, np.arange(8))
    assert seen == [256, 256, 7, 1024, 1024]
    np.testing.assert_allclose(s_default, s_small, rtol=1e-4, atol=1e-6)  # per-jet scores: batch-independent


# ------------------------------------------------------------------------------------- post-hoc variants
def test_ensemble_and_quantile_variants(trained, runs, monkeypatch):
    """Rank ensemble / mass-quantile variants must write per-split (n, 2) CDF scores in [0, 1] that keep ranks."""
    evaluated: dict[str, dict] = {}
    monkeypatch.setattr(R, "evaluate_run", lambda name, **k: evaluated.setdefault(name, k))  # tested above
    sc = trained["scores"]
    R.ensemble_variant(["a", "a"], "a_ens2")
    ens = load_npz(runs / "a_ens2" / "scores.npz")
    assert evaluated["a_ens2"]["cfg"] == {"posthoc": "rank_ensemble", "sources": ["a", "a"]}
    ref = np.sort(sc["bkg_val"].ravel())
    for k, s in sc.items():
        assert ens[k].shape == s.shape and ens[k].dtype == np.float32
        np.testing.assert_allclose(ens[k], np.searchsorted(ref, s, side="right") / ref.size, atol=1e-6)
    assert ens["bkg_val"].max() == 1.0

    R.quantile_normalised_variant("a", n_bins=4)
    qn = load_npz(runs / "a+qn" / "scores.npz")
    assert evaluated["a+qn"]["cfg"]["source"] == "a" and set(qn) == set(sc)
    m = Data(N_MAX).table("bkg_val")["m"]
    edges = np.quantile(m.ravel(), [0.25, 0.5, 0.75])
    for k, s in sc.items():
        assert qn[k].shape == s.shape and qn[k].dtype == np.float32
        assert np.all((qn[k] >= 0) & (qn[k] <= 1)), k
    b = np.digitize(m, edges)
    for i in range(4):  # within a mass bin the transform is monotone in the raw score
        x, y = sc["bkg_val"][b == i], qn["bkg_val"][b == i]
        o = np.argsort(x)
        assert np.all(np.diff(y[o]) >= 0) and y.max() == 1.0


# ---------------------------------------------------------------------------------------------- report
def test_report_collect_table_compare_main(trained, cli_run, runs, env):
    """report.collect / markdown_table / compare / main must cover every evaluated run of a results dir."""
    rows = report.collect()
    names = [n for n, _ in rows]
    assert {"a", "cli"} <= set(names) and "resume" not in names  # runs without metrics.json are skipped
    assert all("auc_2p_SR" in h for _, h in rows)
    md = report.markdown_table(rows)
    lines = md.splitlines()
    assert len(lines) == 2 + len(rows) and lines[0].count("|") == lines[1].count("|")
    assert any(ln.startswith("| a |") for ln in lines) and any(ln.startswith("| cli |") for ln in lines)
    paths = report.compare(["a", "cli:prod"], env / "cmp")
    assert [p.name for p in paths] == [
        "rejection_2prong_SR.png",
        "sic_2prong_SR.png",
        "rejection_3prong_SR.png",
        "sic_3prong_SR.png",
    ]
    assert all(is_png(p) for p in paths)
    assert report.main() == md
    summary = (env / "results" / "summary.md").read_text()
    assert "## AUC in the SR" in summary and "| cli |" in summary and summary.endswith(md + "\n")


# --------------------------------------------------------------------------------------- recon and EDA
def test_recon_figure(trained, runs):
    """recon_figure must rebuild the run's model from config.yaml + model.pt and draw 2 x 4 SR jets."""
    p = recon.recon_figure(runs / "a")
    assert p == runs / "a" / "figures" / "reconstructions.png" and is_png(p)


def test_eda_main(env, capsys):
    """eda.main must write the overview figure and per-split summary for every available test split."""
    assert eda.main() == 0
    out = env / "results" / "eda"
    assert is_png(out / "overview.png")
    s = load_json(out / "summary.json")["summary"]
    d = Data(N_MAX)
    assert set(s) == {"bkg_test", "sig_test", "sig3_test"}
    assert all(s[k]["n_events"] == len(d.events(k)) for k in s)
    assert s["sig_test"]["frac_in_SR"] > s["bkg_test"]["frac_in_SR"]
    assert "bkg_test" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------- CLI
def test_set_dotted_keys_and_yaml_typing():
    """cli._set must create nested dicts, keep siblings and type values with YAML (int/float/bool/list/null)."""
    cfg = {"model": "set_ae", "train": {"epochs": 2, "lr": 0.1}}
    for s in (
        "train.epochs=5",
        "train.lr=1.0e-3",
        "train.device=cpu",
        "model_args.hidden=[8, 16]",
        "a.b.c=true",
        "data.train_mjj=null",
        "data.tag=x=y",
    ):
        cli._set(cfg, s)
    assert cfg == {
        "model": "set_ae",
        "train": {"epochs": 5, "lr": 1e-3, "device": "cpu"},
        "model_args": {"hidden": [8, 16]},
        "a": {"b": {"c": True}},
        "data": {"train_mjj": None, "tag": "x=y"},
    }
    assert isinstance(cfg["train"]["epochs"], int) and isinstance(cfg["train"]["lr"], float)
    with pytest.raises(SystemExit, match="key=value"):
        cli._set(cfg, "train.epochs")


def test_set_scientific_notation_is_a_float():
    """--set bug: YAML 1.1 loaded 1e-3, 1.0e3, -.5 as strings (AdamW TypeError); they must be floats."""
    floats = {"1e-3": 1e-3, "1E5": 1e5, "-2e+4": -2e4, "5e-1": 0.5, "1.0e3": 1e3, ".5e3": 500.0, "-.5": -0.5}
    nested = {"[1e-3, 2, x]": [1e-3, 2, "x"], "{a: [1e-4, '1e-4'], b: +.5}": {"a": [1e-4, "1e-4"], "b": 0.5}}
    for raw, want in {**floats, **nested}.items():
        cfg: dict = {}
        cli._set(cfg, f"train.v={raw}")
        assert cfg["train"]["v"] == want and type(cfg["train"]["v"]) is type(want), raw
    # everything else, quoted values included, is typed exactly as by yaml.safe_load, which is itself untouched
    same = "5 -3 0x10 1_000 2.5 .5 1.0e-3 .inf true off null 09 1e e5 1.2.3 1e-3x cpu '1e-3' \"5\" 2001-12-14"
    for raw in [*same.split(), "", "[1, 2.0, x]"]:
        assert repr(cli._yaml(raw)) == repr(yaml.safe_load(raw)), raw
    assert yaml.safe_load("lr: 1e-3") == {"lr": "1e-3"}  # config files keep PyYAML's YAML 1.1 typing


def test_set_without_equals_exits_in_train_dev_and_sweep(tmp_path, monkeypatch):
    """--set bug: `sweep --set key` silently set key: None; train, dev and sweep must exit with one message."""
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(yaml.safe_dump(CFG))
    for target in ("gaehep.train.run.run", "gaehep.train.dev.dev_run", "gaehep.train.sweep.sweep"):
        monkeypatch.setattr(target, lambda *a, _t=target, **k: pytest.fail(f"{_t} was called"))
    for cmd in ("train", "dev", "sweep"):
        for bad in ("data.n_const", "=30"):
            with pytest.raises(SystemExit, match=re.escape(f"--set expects key=value, got {bad!r}")):
                cli.main([cmd, str(cfg), "--set", "train.seed=1", "--set", bad])


def test_sweep_set_and_grid_values_typed_like_train(tmp_path, monkeypatch):
    """--set bug: sweep --set / --grid must type 1e-3 as a float and `key=` as None, like train's --set."""
    calls: list = []
    monkeypatch.setattr("gaehep.train.sweep.sweep", lambda *a, **k: calls.append(k) or {})
    sets = ["--set", "train.lr=1e-3", "--set", "data.tag="]
    assert cli.main(["sweep", "c.yaml", "--grid", "{beta: [0.0, 1e-3]}", *sets]) == 0
    assert calls[0]["grid"] == {"beta": [0.0, 1e-3]}
    fixed = {"train.device": "auto", "train.threads": 1}
    assert calls[0]["overrides"] == {**fixed, "train.lr": 1e-3, "data.tag": None}


def test_cli_train_end_to_end(cli_run, runs):
    """`gaehep train --name --set` must apply the overrides and run train, score and evaluate."""
    rc, out = cli_run
    assert rc == 0 and "[cli] status: done" in out
    cfg = yaml.safe_load((runs / "cli" / "config.yaml").read_text())
    assert cfg["train"]["seed"] == 2 and cfg["model_args"]["latent_dim"] == 3
    assert isinstance(cfg["train"]["lr"], float) and cfg["train"]["lr"] == 1e-3
    assert cfg["train"]["epochs"] == 1 and cfg["model_args"]["hidden"] == [8, 8]  # untouched keys kept
    assert load_npz(runs / "cli" / "latents.npz")["bkg_train"].shape == (50, 2, 3)
    assert set(load_npz(runs / "cli" / "scores.npz")) == set(SCORE_SPLITS)
    assert (runs / "cli" / "metrics.json").exists() and is_png(runs / "cli" / "figures" / "history.png")


def _cases(cfg: str):
    """(argv, patched target, expected positional args, expected keyword args) for every subcommand."""
    tr, dv, rp = "gaehep.train.run", "gaehep.train.dev", "gaehep.eval.report"
    scores = Path(cfg).with_name(
        "scores.csv"
    )  # a real file: eval-scores loads it before the (patched) evaluation
    scores.write_text("score,label,mjj\n0.1,0,3000\n0.9,1,3500\n")
    return [
        (["download", "--out", "x"], "gaehep.data.download.main", (["--out", "x"],), {}),
        (["preprocess", "--sample", "rnd"], "gaehep.data.preprocess.main", (["--sample", "rnd"],), {}),
        (["tables", "--sample", "rnd"], "gaehep.data.tables.main", (["--sample", "rnd"],), {}),
        (["splits", "--seed", "3"], "gaehep.data.splits.main", (["--seed", "3"],), {}),
        (
            ["train", cfg, "--name", "n", "--stages", "train,score"],
            f"{tr}.run",
            (None, "n"),
            {"stages": ("train", "score")},
        ),
        (["dev", cfg, "--bkg-events", "7"], f"{dv}.dev_run", (None, "tiny"), {"bkg_events": 7}),
        (["dev-report"], f"{dv}.dev_report", (), {}),
        (["dev-rescore"], f"{dv}.dev_rescore", (), {}),
        (["dev-eval", "r"], f"{dv}.dev_eval_run", ("r",), {}),
        (["dev-baselines"], f"{dv}.dev_baselines", (), {}),
        (["promote", "d", "n"], f"{dv}.promote", ("d", "n"), {}),
        (["recon", "some/dir"], "gaehep.eval.recon.recon_figure", (Path("some/dir"),), {}),
        (
            ["dev-kme", "k", "-k", "5", "--bank-mjj", "3300,3700"],
            f"{dv}.dev_kme",
            ("k",),
            {"bank_jets": 100_000, "n_pca": 64, "k": 5, "dim_per_scale": 128, "bank_mjj": (3300.0, 3700.0)},
        ),
        (["dev-profile", "r", "--components", "3"], f"{dv}.dev_profile", ("r", None), {"n_components": 3}),
        (["dev-ensemble", "a", "b", "--out", "o"], f"{dv}.dev_ensemble", (["a", "b"], "o"), {}),
        (["dev-flow", "r", "--kinds", "flow,cflow"], f"{dv}.dev_flow", ("r",), {"kinds": ("flow", "cflow")}),
        (["evaluate", "r"], f"{tr}.evaluate_run", ("r",), {}),
        (["flow", "r", "--conditional"], f"{tr}.latent_flow_variant", ("r",), {"conditional": True}),
        (["qn", "r"], f"{tr}.quantile_normalised_variant", ("r",), {}),
        (
            ["sweep", cfg, "--grid", "{beta: [0.0]}", "--set", "data.n_const=30", "--threads", "2"],
            "gaehep.train.sweep.sweep",
            (cfg,),
            {
                "prefix": "sw",
                "grid": {"beta": [0.0]},
                "overrides": {"train.device": "auto", "train.threads": 2, "data.n_const": 30},
                "budget_hours": 8.0,
                "top_k": 3,
                "n_seeds": 3,
                "final_seeds": 5,
                "score": "rec",
            },
        ),
        (
            ["collapse", "--dev-runs", "x", "y"],
            "gaehep.eval.collapse.collapse_study",
            (),
            {"dev_runs": ("x", "y")},
        ),
        (
            ["weak", "r", "--n-inj", "150,300", "--reps", "2"],
            "gaehep.train.weak_run.weak_run",
            ("r",),
            {"n_models": 5, "n_inj": (150, 300), "reps": (0, 1)},
        ),
        (
            ["bumphunt", "r", "--agg", "prod", "--mode", "qr"],
            f"{tr}.bumphunt_run",
            ("r",),
            {"agg": "prod", "mode": "qr"},
        ),
        (["ensemble", "a", "b", "--out", "o"], f"{tr}.ensemble_variant", (["a", "b"], "o"), {}),
        (["baselines"], f"{tr}.baselines", (), {}),
        (["obs-baselines"], "gaehep.baselines.run_obs_baselines", (), {}),
        (
            ["hl-density", "--seed", "3", "--epochs", "5", "--threads", "2"],
            "gaehep.baselines.run_hl_density",
            (3,),
            {"epochs": 5, "threads": 2},
        ),
        (
            ["efp-knn", "-k", "5", "--n-ref", "1000", "--threads", "2"],
            "gaehep.baselines.run_efp_knn",
            (),
            {"k": 5, "n_ref": 1000, "threads": 2},
        ),
        (["probes", "r"], "gaehep.eval.probes.probe", ("r",), {}),
        (
            ["posthoc", "--prefix", "study", "--curves", "stct"],
            "gaehep.eval.posthoc.posthoc",
            ("study", "stct"),
            {},
        ),
        (
            ["bb-diagnose", "a", "b", "--sample", "bb3"],
            "gaehep.eval.bbdiag.diagnose",
            (["a", "b"], "bb3"),
            {},
        ),
        (["tau", "--sample", "bb3", "--workers", "4"], "gaehep.data.nsubjettiness.main", (None,), {}),
        (["irc-probe", "a", "b", "-k", "6"], "gaehep.eval.irc_probe.main", (["a", "b"],), {"k": 6}),
        (
            ["study-table", "study", "--n-boot", "20"],
            "gaehep.eval.study.study_table",
            ("study",),
            {"n_boot": 20},
        ),
        (
            ["score-sample", "r", "--sample", "bb3", "--subsample", "--chunk", "1000", "--n-boot", "0"],
            "gaehep.eval.blackbox.score_sample",
            ("r", "bb3"),
            {"subsample": True, "chunk": 1000, "n_boot": 0, "rescore": False},
        ),
        (
            ["compare", "a", "b:prod", "--out", "d"],
            f"{rp}.compare",
            (["a", "b:prod"], Path("d")),
            {"agg": "max", "region": "SR"},
        ),
        (["report"], f"{rp}.main", (), {}),
        (
            ["eval-scores", str(scores), "--out", "o", "--sr", "3000,4000", "--n-boot", "5"],
            "gaehep.eval.score_file.evaluate_score_arrays",
            (None, None, None),  # the loaded score, label and mjj arrays
            {"sr": (3000.0, 4000.0), "n_boot": 5, "name": "scores", "out_dir": Path("o")},
        ),
    ]


def _subcommands() -> set[str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as e:
        cli.main(["-h"])
    assert e.value.code == 0
    return set(re.search(r"\{([a-z,-]+)\}", out.getvalue()).group(1).split(","))


def test_every_subcommand_parses_and_dispatches(tmp_path, monkeypatch):
    """Every subcommand must parse its options and call its entry point with correctly converted arguments."""
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(yaml.safe_dump(CFG))
    cases = _cases(str(cfg))
    assert {c[0][0] for c in cases} == _subcommands()  # a new subcommand needs a case here
    for argv, target, args, kwargs in cases:
        calls: list = []
        ret = 0 if target.startswith("gaehep.data.") else {}  # data mains return an exit code
        monkeypatch.setattr(target, lambda *a, _c=calls, _r=ret, **k: _c.append((a, k)) or _r)
        with contextlib.redirect_stdout(io.StringIO()):
            assert cli.main(argv) == 0, argv
        assert len(calls) == 1, argv
        a, k = calls[0]
        assert len(a) == len(args) and all(x is None or y == x for y, x in zip(a, args, strict=True)), (
            argv,
            a,
        )
        assert {n: k[n] for n in kwargs if n in k} == kwargs, (argv, k)
        if "log" in k:
            assert k["log"] is cli._log
        if argv[0] in ("train", "dev"):
            assert a[0]["model"] == "set_ae"  # the YAML config was read (and, for dev, DEV_TRAIN merged in)
            if argv[0] == "dev":
                assert a[0]["train"]["epochs"] == 5 and a[0]["train"]["batch_size"] == 64
        if argv[0] == "train":
            assert a[0]["train"] == CFG["train"]


def test_subcommand_help_and_bad_choice():
    """Each subcommand's parser must build and answer -h; invalid choices must exit with argparse's code 2."""
    for cmd in sorted(_subcommands()):
        with contextlib.redirect_stdout(io.StringIO()), pytest.raises(SystemExit) as e:
            cli.main([cmd, "-h"])
        assert e.value.code == 0, cmd
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit) as e:
        cli.main(["bumphunt", "r", "--agg", "median"])
    assert e.value.code == 2
