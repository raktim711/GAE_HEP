"""LHC Olympics black boxes as untouched test data / training data: download registry, master keys,
preprocessing (with ``--delete-raw``), tables, test-only splits, the unchanged R&D split hash, the
``data.train_sample`` option, and ``gaehep score-sample`` (outputs, resumption, baseline dispatch, BB2 null test).

Everything runs on small synthetic files in the black-box format (``tests/bb_synth.py``); no network."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from gaehep import cli
from gaehep.config import BB2_WINDOWS, BLACKBOX_WINDOWS, data_dir, load_json, run_metadata
from gaehep.data import download, splits
from gaehep.data.fake import make_fake
from gaehep.data.preprocess import preprocess, read_master_key, verify_processed
from gaehep.data.tables import build_tables
from gaehep.eval import blackbox
from gaehep.eval.evaluate import AGGREGATIONS, aggregate, region_mask
from gaehep.eval.metrics import auc
from gaehep.train import loop
from gaehep.train import run as R
from gaehep.train.dev import dev_dir, dev_run, promote
from gaehep.train.run import SPLIT_SAMPLE, Data, _run_dir
from tests.bb_synth import K_CONST, write_box

# resolved at import, before the fixtures point GAEHEP_DATA at fake data
REAL_PROCESSED = data_dir() / "processed"
N_MAX = 30
N_SUB = 150  # background events of the _test_sub splits (the real value is 200 000)
BOXES = {  # sample -> write_box arguments
    "bb1": {"n_bkg": 500, "signals": {1: 12}, "n_one_jet": 6, "n_empty": 3, "seed": 1},
    "bb2": {"n_bkg": 300, "n_one_jet": 4, "seed": 2},
    "bb3": {"n_bkg": 400, "signals": {1: 12, 2: 20}, "mass": 4200.0, "n_one_jet": 5, "seed": 3},
}
CFG = {  # a tiny set AE, as in tests/test_orchestration.py
    "model": "set_ae",
    "model_args": {"hidden": [8, 8], "k": 4, "latent_dim": 2, "n_out": N_MAX, "dec_hidden": [16]},
    "data": {"n_const": N_MAX, "latent_train_events": 50},
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


def quiet(*a: object) -> None:
    pass


def protocol_hash(path: Path) -> str:
    """The snippet of docs/STUDY_PROTOCOL.md, section 5."""
    f = np.load(path)
    h = hashlib.sha256()
    for k in sorted(f.files):
        a = f[k]
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------- environment
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Fake R&D data + black boxes bb1-bb3 made by the real pipeline (preprocess --delete-raw, tables, splits)."""
    root = tmp_path_factory.mktemp("bb")
    data = root / "data"
    make_fake(data, 600, 150, N_MAX)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GAEHEP_DATA", str(data))
        mp.setenv("GAEHEP_RESULTS", str(root / "results"))
        mp.setattr(blackbox, "BB3_MODE_COUNTS", {"dijet": 12, "trijet": 20})
        rnd_hash = protocol_hash(data / "processed" / "splits.npz")
        keys = {}
        for sample, kw in BOXES.items():
            keys[sample] = write_box(data / "raw", sample, **kw)
            preprocess(sample, chunk=100, workers=1, delete_raw=True)
            build_tables(sample)
        assert splits.main(["--sample", "bb1", "bb2", "bb3", "--n-bkg-sub", str(N_SUB)]) == 0
        yield {"root": root, "data": data, "keys": keys, "rnd_hash": rnd_hash}


@pytest.fixture(scope="module")
def box_tables(env):
    return {s: dict(np.load(env["data"] / "processed" / s / "events.npz")) for s in BOXES}


# ------------------------------------------------------------------------------------------- download
def test_blackbox_md5s_are_the_protocols():
    assert download.BLACKBOX_FILES == {
        "events_LHCO2020_BlackBox1.h5": "847be306afd923a4ba30543070113627",
        "events_LHCO2020_BlackBox1.masterkey": "c5d9b1e75ce26fd820c1c99dfc874e92",
        "events_LHCO2020_BlackBox2.h5": "2e5dce598238789938cb31e9cd3e7a46",
        "events_LHCO2020_BlackBox3.h5": "ffc53e0c0b42e0f2c752ce48db74ce27",
        "events_LHCO2020_BlackBox3.masterkey": "51e275f122eb48b10ccc2c858dda358e",
    }
    assert download.BLACKBOX_RECORD == "4536624"


def test_download_selection(monkeypatch, tmp_path):
    got: list[str] = []
    monkeypatch.setattr(download, "download_file", lambda name, out, retries=5: got.append(name))
    monkeypatch.setenv("GAEHEP_DATA", str(tmp_path))  # no processed black boxes here
    assert download.main(["--out", str(tmp_path)]) == 0
    assert sorted(got) == sorted(download.FILES)  # plain `gaehep download` stays R&D-only
    assert not set(got) & set(download.BLACKBOX_FILES)
    got.clear()
    download.main(["--out", str(tmp_path), "--sample", "bb1"])
    assert got == ["events_LHCO2020_BlackBox1.masterkey", "events_LHCO2020_BlackBox1.h5"]  # key first
    got.clear()
    download.main(["--out", str(tmp_path), "--sample", "bb2"])
    assert got == ["events_LHCO2020_BlackBox2.h5"]  # BB2 has no master key
    got.clear()
    download.main(["--out", str(tmp_path), "--sample", "bb3", "bb1"])
    assert sorted(got) == sorted(n for n in download.BLACKBOX_FILES if "BlackBox2" not in n)
    with pytest.raises(SystemExit):
        download.main(["--out", str(tmp_path), "--sample", "bb4"])


def test_download_uses_the_blackbox_record_and_verifies_md5(monkeypatch, tmp_path):
    name = "events_LHCO2020_BlackBox2.h5"
    content = b"toy events"
    monkeypatch.setitem(download.ALL_FILES, name, hashlib.md5(content).hexdigest())
    urls: list[str] = []

    def stream(url, dest, log_every=30.0):
        urls.append(url)
        dest.write_bytes(content if len(urls) == 1 else b"corrupt")

    monkeypatch.setattr(download, "_stream", stream)
    monkeypatch.setattr(download.time, "sleep", lambda s: None)
    path = download.download_file(name, tmp_path)
    assert path.read_bytes() == content and not (tmp_path / (name + ".part")).exists()
    assert "/records/4536624/files/" in urls[0] and name in urls[0]
    assert download.download_file(name, tmp_path) == path and len(urls) == 1  # verified file: no new transfer
    path.unlink()
    monkeypatch.setitem(download.ALL_FILES, name, hashlib.md5(b"other").hexdigest())
    with pytest.raises(RuntimeError, match="could not download"):
        download.download_file(name, tmp_path, retries=1)  # a md5 mismatch is never accepted
    assert not path.exists()
    assert download.record_of("events_anomalydetection_v2.h5") == download.ZENODO_RECORD


def test_download_skips_a_processed_box_whose_raw_file_was_deleted(env, monkeypatch, tmp_path):
    got: list[str] = []
    monkeypatch.setattr(download, "download_file", lambda name, out, retries=5: got.append(name))
    download.main(["--out", str(tmp_path), "--sample", "bb1"])
    assert got == []  # the fixture preprocessed bb1 with --delete-raw
    download.main(["--out", str(tmp_path), "--sample", "bb1", "--force"])
    assert "events_LHCO2020_BlackBox1.h5" in got


# ------------------------------------------------------------------------------------ preprocessing
def test_master_key_parsing(tmp_path):
    f = tmp_path / "k.masterkey"
    f.write_text("0.0\n1.0\n2.0\n0.0\n")
    lab = read_master_key(f)
    assert lab.dtype == np.int8 and lab.tolist() == [0, 1, 2, 0]
    f.write_text("0.0\n3.0\n")
    with pytest.raises(ValueError, match="0, 1 or 2"):
        read_master_key(f)
    f.write_text("0.0\n0.5\n")
    with pytest.raises(ValueError):
        read_master_key(f)


def test_master_key_alignment(env, box_tables):
    """labels.npy is the key in file order, and every label sits on a row whose event looks like that label
    (the marker is the constituent multiplicity), across chunk boundaries (chunks of 100 events)."""
    for sample, tab in box_tables.items():
        key = env["keys"][sample]
        assert tab["label"].dtype == np.int8 and np.array_equal(tab["label"], key)
        assert np.array_equal(np.load(env["data"] / "processed" / sample / "labels.npy"), key)
        has_jet = tab["n_jets"] >= 1
        k_leading = tab["n_const"][:, 0]
        for lab, k in K_CONST.items():
            sel = has_jet & (key == lab)
            assert sel.sum() == (key == lab).sum() - (key[~has_jet] == lab).sum()
            assert np.all(k_leading[sel] == k), (sample, lab)
    assert (box_tables["bb2"]["label"] != 0).sum() == 0  # BB2: all background, no key
    assert np.bincount(env["keys"]["bb3"], minlength=3)[1:].tolist() == [12, 20]


def test_raw_file_deleted_after_verified_run(env):
    raw = env["data"] / "raw"
    for sample in BOXES:
        assert not (raw / download.SAMPLE_FILES[sample][-1]).exists()  # the events file is gone ...
    assert (raw / "events_LHCO2020_BlackBox1.masterkey").exists()  # ... the key is kept
    meta = load_json(env["data"] / "processed" / "bb1" / "meta.json")
    assert meta["raw_deleted"] is True and meta["n_events"] == len(env["keys"]["bb1"])
    assert meta["master_key"].endswith("BlackBox1.masterkey")
    before = (env["data"] / "processed" / "bb1" / "labels.npy").stat().st_mtime_ns
    preprocess("bb1", workers=1, delete_raw=True)  # rerun: nothing to read, nothing to do
    assert (env["data"] / "processed" / "bb1" / "labels.npy").stat().st_mtime_ns == before


def test_delete_raw_is_refused_for_partial_runs_and_bad_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("GAEHEP_DATA", str(tmp_path))
    raw = tmp_path / "raw"
    write_box(raw, "bb1", 30, {1: 3}, seed=5)
    src = raw / "events_LHCO2020_BlackBox1.h5"
    with pytest.raises(ValueError, match="--limit"):
        preprocess("bb1", limit=10, delete_raw=True)
    key = raw / "events_LHCO2020_BlackBox1.masterkey"
    key.write_text("".join(key.read_text().splitlines(keepends=True)[:-1]))  # one label missing
    with pytest.raises(ValueError, match="labels for"):
        preprocess("bb1", workers=1, delete_raw=True)
    assert src.exists()


def test_verify_processed_reports_problems(env, tmp_path):
    import shutil

    d = tmp_path / "bb1"
    shutil.copytree(env["data"] / "processed" / "bb1", d)
    n = len(env["keys"]["bb1"])
    assert verify_processed(d, n, 100, env["keys"]["bb1"], 100) == []
    (d / "chunks" / "00000100_00000200.ok").unlink()
    assert any("not marked done" in p for p in verify_processed(d, n, 100, env["keys"]["bb1"], 100))
    (d / "chunks" / "00000100_00000200.ok").touch()
    wrong = env["keys"]["bb1"].copy()
    wrong[0] ^= 1
    assert any("master key" in p for p in verify_processed(d, n, 100, wrong, 100))
    p4 = np.load(d / "jets_p4.npy", mmap_mode="r+")
    p4[np.flatnonzero(np.load(d / "n_jets.npy") >= 2)[0]] = np.nan
    p4.flush()
    assert any("non-finite" in p for p in verify_processed(d, n, 100, None, 100))


def test_tables_of_a_box_without_feature_files(env, box_tables):
    t = box_tables["bb1"]
    assert np.isnan(t["tau1"]).all() and not t["matched"].any()
    two = t["n_jets"] >= 2
    assert np.isfinite(t["mjj"][two]).all() and np.isnan(t["mjj"][~two]).all()
    assert (t["n_jets"] < 2).sum() == 6 + 3  # the single-jet and empty events are in the table
    v = load_json(env["data"] / "processed" / "bb3" / "validation.json")
    assert v["label_counts"] == {"0": 400 + 5, "1": 12, "2": 20} and v["feature_file"] is None
    # signal events sit at the box's resonance mass
    sig = two & (t["label"] == 1)
    assert np.all(np.abs(t["mjj"][sig] - 3800.0) < 250.0)


# ----------------------------------------------------------------------------------------- splits
def test_blackbox_splits_are_test_only(env, box_tables):
    sp = splits.load_splits(env["data"] / "processed")
    for sample, tab in box_tables.items():
        two = tab["n_jets"] >= 2
        assert np.array_equal(sp[f"{sample}_test"], np.flatnonzero(two))  # every event with >= 2 jets
        sub = sp[f"{sample}_test_sub"]
        assert np.array_equal(sub, np.sort(sub)) and len(np.unique(sub)) == len(sub)
        assert set(sub) <= set(sp[f"{sample}_test"])
        bkg = np.flatnonzero(two & (tab["label"] == 0))
        sig = np.flatnonzero(two & (tab["label"] > 0))
        assert set(sig) <= set(sub)  # every signal event is kept
        assert len(sub) == N_SUB + len(sig)  # N_SUB background events
        assert np.array_equal(
            np.setdiff1d(sub, sig), np.sort(np.random.default_rng(20201).choice(bkg, N_SUB, replace=False))
        )
    assert not any(
        k.startswith("bb") and k.endswith(("train", "val")) for k in sp
    )  # no black-box training split
    assert SPLIT_SAMPLE["bb1_test"] == "bb1" and SPLIT_SAMPLE["bb3_test_sub"] == "bb3"


def test_subsample_keeps_everything_when_the_box_is_small():
    n_jets = np.array([2, 2, 1, 2, 2, 2])
    labels = np.array([0, 1, 0, 0, 2, 0], dtype=np.int8)
    s = splits.make_blackbox_splits("bb3", n_jets, labels, n_bkg_sub=100)
    assert s["bb3_test"].tolist() == [0, 1, 3, 4, 5] and s["bb3_test_sub"].tolist() == [0, 1, 3, 4, 5]
    s = splits.make_blackbox_splits("bb3", n_jets, labels, n_bkg_sub=2)
    assert len(s["bb3_test_sub"]) == 4 and {1, 4} <= set(s["bb3_test_sub"])
    assert s["bb3_test_sub"].dtype == np.int64


def test_rnd_splits_are_untouched(env):
    """The black-box splits live in their own file: splits.npz and its hash do not change, and make_splits
    still returns exactly what it returned before this change."""
    proc = env["data"] / "processed"
    assert splits.content_hash(proc) == protocol_hash(proc / "splits.npz") == env["rnd_hash"]
    assert set(np.load(proc / "splits.npz").files) == {
        "bkg_train", "bkg_val", "bkg_test", "sig_train", "sig_val", "sig_test", "sig3_test",
    }  # fmt: skip
    assert (proc / "splits_blackbox.npz").exists()
    rng = np.random.default_rng(0)
    labels = (rng.random(5000) < 0.2).astype(np.int8)
    n_jets = np.where(rng.random(5000) < 0.02, 1, 2)
    sp = splits.make_splits(n_jets, labels, np.where(rng.random(700) < 0.02, 1, 2), seed=20201)
    h = hashlib.sha256()
    for k in sorted(sp):
        h.update(k.encode())
        h.update(str(sp[k].dtype).encode())
        h.update(sp[k].tobytes())
    assert (
        h.hexdigest() == "e812081980cdd411afac5d47d6080564bed015189c1cc3afeb854a80e781398c"
    )  # value before P2


def test_content_hash_cli_and_metadata(env, monkeypatch, capsys):
    proc = env["data"] / "processed"
    assert splits.content_hash() == splits.content_hash(proc / "splits.npz") == env["rnd_hash"]
    assert run_metadata()["splits_sha256"] == env["rnd_hash"]
    assert splits.main(["--check", "--proc-root", str(proc)]) == 1  # the fake splits are not the real ones
    assert "DIFFERS" in capsys.readouterr().out
    monkeypatch.setattr(splits, "PROTOCOL_SPLITS_SHA256", env["rnd_hash"])
    assert splits.main(["--check", "--proc-root", str(proc)]) == 0
    monkeypatch.setenv("GAEHEP_DATA", str(proc / "nowhere"))
    assert run_metadata()["splits_sha256"] is None
    assert splits.main(["--check", "--proc-root", str(proc / "nowhere")]) == 1


@pytest.mark.data
def test_real_splits_hash_is_the_protocols():
    if not (REAL_PROCESSED / "splits.npz").exists():
        pytest.skip("the real R&D splits are not present")
    assert splits.content_hash(REAL_PROCESSED) == splits.PROTOCOL_SPLITS_SHA256


# ------------------------------------------------------------------------------------ train_sample
class _NoLabels(dict):
    """An event table that refuses to give out the labels."""

    def __getitem__(self, key):
        if key == "label":
            raise AssertionError("the labels were read")
        return super().__getitem__(key)


def test_train_sample_splits_and_scaler(env):
    d = Data(N_MAX, train_sample="bb1")
    all_ev = d.splits["bb1_test"]
    tr, va = d.events("bkg_train"), d.events("bkg_val")
    assert len(tr) + len(va) == len(all_ev) and len(va) == len(all_ev) - round(0.9 * len(all_ev))
    assert np.array_equal(np.sort(np.r_[tr, va]), all_ev) and not set(tr) & set(va)
    assert np.array_equal(tr, splits.train_val_split(all_ev)[0])  # default_rng(20201), 90/10
    idx = all_ev.copy()
    np.random.default_rng(20201).shuffle(idx)
    assert np.array_equal(va, np.sort(idx[round(0.9 * len(idx)) :]))
    assert d.sample_of("bkg_train") == d.sample_of("bkg_val") == "bb1" and d.sample_of("bkg_test") == "rnd"
    assert np.array_equal(d.events("bkg_test"), Data(N_MAX).events("bkg_test"))  # test splits unchanged
    assert not np.array_equal(Data(N_MAX).events("bkg_val"), va)
    sc = d.scaler()
    assert (env["data"] / "processed" / f"scaler_bb1_n{N_MAX}.json").exists()
    assert not (env["data"] / "processed" / f"scaler_n{N_MAX}.json").exists() or sc is not None
    with pytest.raises(ValueError, match="train_sample"):
        Data(N_MAX, train_sample="qqq")
    assert Data.from_config({"n_const": 20, "train_sample": "bb1"}).train_sample == "bb1"
    assert Data.from_config(None).train_sample == "rnd" and Data.from_config({}).n_const == 50


def test_train_sample_never_reads_the_labels(env):
    d = Data(N_MAX, train_sample="bb1")
    store = d.store("bb1")
    labelled = dict(store.table)
    store.table = _NoLabels(labelled)
    try:
        for split in ("bkg_train", "bkg_val"):
            pool = d.pool(split)
            assert len(pool) == 2 * len(d.events(split)) and not pool.label.any()
            assert "label" not in d.table(split)
        d.scaler()
        with pytest.raises(AssertionError, match="labels were read"):
            d.pool("bb1_test_sub")  # the guard is live: a test split does read them
    finally:
        store.table = labelled
    assert d.store("bb1").pool(d.events("bkg_val"), N_MAX).label.any()  # the signal is in the validation part
    assert (labelled["label"][d.events("bkg_train")] > 0).any()  # ... and in the training part: not removed


@pytest.fixture(scope="module")
def runs(env):
    """A run trained on the R&D background, and one trained on BB1 (gaehep dev -> gaehep promote)."""
    R.run(copy.deepcopy(CFG), "rnd", log=quiet)
    cfg = copy.deepcopy(CFG)
    cfg["data"]["train_sample"] = "bb1"
    dev = dev_run(cfg, "dev_bb1", log=quiet)
    promoted = promote("dev_bb1", "bb1data", log=quiet)
    return {"dev": dev, "promoted": promoted, "cfg": cfg}


def test_dev_and_promote_with_train_sample(env, runs):
    d = Data(N_MAX, train_sample="bb1")
    n_val = len(d.events("bkg_val"))
    assert runs["dev"]["epochs"] == 1 and runs["dev"]["splits_sha256"] == env["rnd_hash"]
    with np.load(dev_dir("dev_bb1") / "dev_scores.npz") as f:
        assert f["bkg_val"].shape == (n_val, 2)  # scored on BB1's validation part
    rd = _run_dir("bb1data")
    assert yaml.safe_load((rd / "config.yaml").read_text())["data"]["train_sample"] == "bb1"
    sc = {k: v for k, v in np.load(rd / "scores.npz").items()}
    assert sc["bkg_val"].shape == (n_val, 2)  # F_B comes from BB1's validation part
    assert sc["bkg_test"].shape == (
        len(Data(N_MAX).events("bkg_test")),
        2,
    )  # the R&D test splits are scored too
    assert np.isfinite(sc["bkg_val"]).all()
    lat = np.load(rd / "latents.npz")
    assert lat["bkg_train"].shape[0] == CFG["data"]["latent_train_events"]
    m = load_json(rd / "metrics.json")
    assert m["splits_sha256"] == env["rnd_hash"] and "headline" in m
    assert load_json(dev_dir("dev_bb1") / "dev_metrics.json")["splits_sha256"] == env["rnd_hash"]
    # the trained-on-BB1 model is not the R&D-trained one
    assert (rd / "model.pt").read_bytes() != (_run_dir("rnd") / "model.pt").read_bytes()
    assert load_json(_run_dir("rnd") / "metrics.json")["splits_sha256"] == env["rnd_hash"]


# -------------------------------------------------------------------------------------- metrics
def test_region_mask_window():
    mjj = np.array([3500.0, 3600.0, 3800.0, 4000.0, 4001.0])
    assert region_mask(mjj, "window", (3600.0, 4000.0)).tolist() == [False, True, True, True, False]
    assert region_mask(mjj, "full").all() and region_mask(mjj, "SR").tolist()[0] is True
    with pytest.raises(ValueError, match="window"):
        region_mask(mjj, "window")
    with pytest.raises(ValueError):
        region_mask(mjj, "nowhere")


def test_bb3_modes_are_identified_by_their_counts(monkeypatch):
    monkeypatch.setattr(blackbox, "BB3_MODE_COUNTS", {"dijet": 1200, "trijet": 2000})
    lab = np.r_[np.zeros(50, np.int8), np.ones(1200, np.int8), np.full(2000, 2, np.int8)]
    assert blackbox.mode_labels(lab) == {"dijet": 1, "trijet": 2}
    swapped = np.where(lab == 1, 2, np.where(lab == 2, 1, lab)).astype(np.int8)
    assert blackbox.mode_labels(swapped) == {"dijet": 2, "trijet": 1}  # by counts, not by the label value
    assert set(blackbox.signal_masks("bb3", swapped[swapped != 0], swapped)) == {
        "bb3",
        "bb3_dijet",
        "bb3_trijet",
    }
    with pytest.raises(ValueError, match="counts"):
        blackbox.mode_labels(lab[:-1])
    assert blackbox.signal_masks("bb2", lab, lab) == {} and set(blackbox.signal_masks("bb1", lab, lab)) == {
        "bb1"
    }


def test_evaluate_signal_box_numbers():
    """Perfectly separated toy scores: AUC 1 in the window and over the full range; counts are per region."""
    rng = np.random.default_rng(3)
    n = 4000
    mjj = rng.uniform(2800.0, 4800.0, n).astype(np.float32)
    label = np.zeros(n, np.int8)
    label[rng.choice(n, 60, replace=False)] = 1
    jets = rng.normal(0.0, 1.0, (n, 2)).astype(np.float32) + 5.0 * (label[:, None] > 0)
    ref = np.sort(rng.normal(0.0, 1.0, 5000))
    res = blackbox.evaluate_signal_box("bb1", jets, mjj, label, ref, n_boot=5)
    win = BLACKBOX_WINDOWS["bb1"]
    in_win = (mjj >= win[0]) & (mjj <= win[1])
    e = res["event"]["prod"]
    assert set(e) == {"bb1/window", "bb1/full"} and set(res["event"]) == set(AGGREGATIONS)
    assert e["bb1/window"]["n_sig"] == int((in_win & (label == 1)).sum())
    assert e["bb1/window"]["n_bkg"] == int((in_win & (label == 0)).sum())
    assert e["bb1/full"]["n_sig"] == 60 and e["bb1/full"]["n_bkg"] == n - 60
    assert e["bb1/full"]["auc"] > 0.999 and e["bb1/window"]["auc"] > 0.999 and "bootstrap" in e["bb1/full"]
    assert "bootstrap" not in res["event"]["sum"]["bb1/full"]
    prod = aggregate(jets, "prod", ref)
    assert np.isclose(e["bb1/full"]["auc"], auc(prod[label == 1], prod[label == 0]))
    assert set(res["jet"]) == {"bb1"} and res["jet"]["bb1"]["n_sig"] == 120
    assert "jsd@0.01" in res["sculpting"]["prod"]["mjj"] and set(res) == {"event", "jet", "sculpting"}
    # a window without signal gives NaNs and the counts, not an error
    res2 = blackbox.evaluate_signal_box(
        "bb1", jets, mjj, np.where(in_win, 0, label).astype(np.int8), ref, n_boot=0
    )
    w = res2["event"]["max"]["bb1/window"]
    assert w["n_sig"] == 0 and np.isnan(w["auc"]) and w["n_bkg"] > 0


def test_evaluate_null_box():
    rng = np.random.default_rng(4)
    n = 20_000
    mjj = (2700.0 + rng.exponential(500.0, n)).clip(max=4990.0).astype(np.float32)
    jets = rng.normal(0.0, 1.0, (n, 2)).astype(np.float32)
    ref = np.sort(rng.normal(0.0, 1.0, 10_000))
    same = blackbox.evaluate_null_box(jets, mjj, ref, jets)
    assert set(same["ks"]) == {*AGGREGATIONS, "jet"} and all(v == 0.0 for v in same["ks"].values())
    shifted = blackbox.evaluate_null_box(jets, mjj, ref, jets + 0.5)
    assert all(v > 0.1 for v in shifted["ks"].values())
    rows = same["bumphunt"]
    assert len(rows) == len(BB2_WINDOWS) * len(blackbox.BUMPHUNT_EFFS)
    assert {tuple(r["window"]) for r in rows} == set(BB2_WINDOWS) and all(r["n_inj"] == 0 for r in rows)
    assert {r["eff"] for r in rows} == set(blackbox.BUMPHUNT_EFFS)
    uncut = [r for r in rows if r["eff"] == 1.0]
    assert all(r["fit_ok"] and abs(r["z"]) < 5 for r in uncut)  # smooth spectrum: no bump without a cut


# --------------------------------------------------------------------------------- score-sample
def _check_files(run: str, sample: str, subsample: bool, env):
    rd = _run_dir(run)
    d = Data.from_config(yaml.safe_load((rd / "config.yaml").read_text()).get("data"))
    split = f"{sample}_test_sub" if subsample else f"{sample}_test"
    ev = d.events(split)
    with np.load(rd / f"{sample}_scores.npz") as f:
        assert set(f.files) == {"jet_scores", "event_index", "mjj", "label", "subsample"}
        assert f["jet_scores"].shape == (len(ev), 2) and f["jet_scores"].dtype == np.float32
        assert np.isfinite(f["jet_scores"]).all()
        assert f["event_index"].dtype == np.int64 and np.array_equal(f["event_index"], ev)
        assert f["mjj"].dtype == np.float32 and f["label"].dtype == np.int8
        assert (
            f["subsample"].dtype == np.bool_
            and f["subsample"].shape == ()
            and bool(f["subsample"]) is subsample
        )
        tab = np.load(env["data"] / "processed" / sample / "events.npz")
        assert np.array_equal(f["mjj"], tab["mjj"][ev]) and np.array_equal(
            f["label"], env["keys"][sample][ev]
        )
        out = {k: f[k] for k in f.files}
    meta = load_json(rd / f"{sample}_metrics.json")
    assert list(meta) == ["name", "sample", "subsample", "splits_sha256", "git", "time", "metrics"]
    assert meta["name"] == run and meta["sample"] == sample and meta["subsample"] is subsample
    assert meta["splits_sha256"] == env["rnd_hash"]
    return out, meta["metrics"]


def test_score_sample_bb1_trained_on_bb1(env, runs):
    out, m = blackbox.score_sample("bb1data", "bb1", subsample=True, chunk=60, n_boot=5, log=quiet), None
    scores, m = _check_files("bb1data", "bb1", True, env)
    assert set(m) == {"event", "jet", "sculpting"} and set(m["event"]) == set(AGGREGATIONS)
    assert set(m["event"]["prod"]) == {"bb1/window", "bb1/full"}
    win = BLACKBOX_WINDOWS["bb1"]
    lab, mjj = scores["label"], scores["mjj"]
    inw = (mjj >= win[0]) & (mjj <= win[1])
    e = m["event"]["prod"]
    assert e["bb1/full"]["n_sig"] == 12 and e["bb1/full"]["n_bkg"] == N_SUB
    assert e["bb1/window"]["n_sig"] == int((inw & (lab > 0)).sum()) and e["bb1/window"]["n_bkg"] == int(
        (inw & (lab == 0)).sum()
    )
    # F_B is BB1's validation part: recompute the prod AUC from the files
    sc = np.load(_run_dir("bb1data") / "scores.npz")
    assert sc["bkg_val"].shape[0] == len(Data(N_MAX, train_sample="bb1").events("bkg_val"))
    ref = np.sort(sc["bkg_val"].astype(np.float64).ravel())
    cdf = (np.searchsorted(ref, scores["jet_scores"], side="right") + 0.5) / (len(ref) + 1.0)
    prod = np.log(cdf).sum(1)
    assert np.isclose(e["bb1/full"]["auc"], auc(prod[lab > 0], prod[lab == 0]))
    assert out["name"] == "bb1data"
    assert not list(_run_dir("bb1data").glob("_partial*"))  # the resumption cache is gone


def test_score_sample_bb3_modes_and_full_split(env, runs):
    blackbox.score_sample("rnd", "bb3", subsample=False, chunk=128, n_boot=0, log=quiet)
    scores, m = _check_files("rnd", "bb3", False, env)
    e = m["event"]["prod"]
    assert set(e) == {f"{s}/{r}" for s in ("bb3", "bb3_dijet", "bb3_trijet") for r in ("window", "full")}
    key = env["keys"]["bb3"][scores["event_index"]]
    assert (
        e["bb3_dijet/full"]["n_sig"] == 12
        and e["bb3_trijet/full"]["n_sig"] == 20
        and e["bb3/full"]["n_sig"] == 32
    )
    assert e["bb3_dijet/full"]["n_bkg"] == e["bb3/full"]["n_bkg"] == int((key == 0).sum()) == 400
    win = BLACKBOX_WINDOWS["bb3"]
    assert e["bb3_dijet/window"]["n_sig"] == 12  # the dijet mode sits at 4.2 TeV
    assert e["bb3_trijet/window"]["n_sig"] == 0 and np.isnan(
        e["bb3_trijet/window"]["auc"]
    )  # the trijet mode does not
    assert e["bb3/window"]["n_sig"] == 12
    inw = (scores["mjj"] >= win[0]) & (scores["mjj"] <= win[1])
    assert e["bb3/window"]["n_bkg"] == int((inw & (key == 0)).sum())
    assert set(m["jet"]) == {"bb3", "bb3_dijet", "bb3_trijet"}
    assert "bootstrap" not in e["bb3/full"]  # --n-boot 0
    # the combined AUC lies between the two modes' (all share the background)
    a = [e[f"{s}/full"]["auc"] for s in ("bb3_dijet", "bb3_trijet")]
    assert min(a) - 1e-9 <= e["bb3/full"]["auc"] <= max(a) + 1e-9


def test_score_sample_bb2_null_test(env, runs):
    blackbox.score_sample("rnd", "bb2", subsample=True, chunk=100, log=quiet)
    _, m = _check_files("rnd", "bb2", True, env)
    assert set(m) == {"ks", "bumphunt"} and set(m["ks"]) == {*AGGREGATIONS, "jet"}
    assert all(0.0 <= v <= 1.0 for v in m["ks"].values())
    assert len(m["bumphunt"]) == 9 and {tuple(r["window"]) for r in m["bumphunt"]} == set(BB2_WINDOWS)
    assert all(r["n_inj"] == 0 for r in m["bumphunt"])


def test_score_sample_resumes_per_chunk(env, runs, monkeypatch):
    rd = _run_dir("rnd")
    blackbox.score_sample("rnd", "bb1", subsample=True, chunk=30, n_boot=0, log=quiet)
    straight = np.load(rd / "bb1_scores.npz")["jet_scores"]
    (rd / "bb1_scores.npz").unlink()
    real, calls = loop.score_pool, []

    def flaky(*a, **kw):
        if len(calls) == 3:
            raise RuntimeError("job killed")
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(loop, "score_pool", flaky)
    with pytest.raises(RuntimeError, match="killed"):
        blackbox.score_sample("rnd", "bb1", subsample=True, chunk=30, n_boot=0, log=quiet)
    assert (
        len(list((rd / "_partial_bb1_sub").glob("chunk_*.npy"))) == 3 and not (rd / "bb1_scores.npz").exists()
    )
    calls.clear()
    monkeypatch.setattr(loop, "score_pool", lambda *a, **kw: (calls.append(1), real(*a, **kw))[1])
    blackbox.score_sample("rnd", "bb1", subsample=True, chunk=30, n_boot=0, log=quiet)
    n_chunks = -(-len(Data(N_MAX).events("bb1_test_sub")) // 30)
    assert len(calls) == n_chunks - 3  # only the missing chunks were scored
    assert np.array_equal(np.load(rd / "bb1_scores.npz")["jet_scores"], straight)
    # an up-to-date result is reused (metrics only); --rescore and a new model both score again
    calls.clear()
    blackbox.score_sample("rnd", "bb1", subsample=True, chunk=30, n_boot=0, log=quiet)
    assert calls == []
    blackbox.score_sample("rnd", "bb1", subsample=True, chunk=30, n_boot=0, rescore=True, log=quiet)
    assert len(calls) == n_chunks
    calls.clear()
    (rd / "model.pt").touch()
    blackbox.score_sample("rnd", "bb1", subsample=True, chunk=30, n_boot=0, log=quiet)
    assert len(calls) == n_chunks


def test_score_sample_discards_chunks_of_another_model(env, runs, monkeypatch):
    rd = _run_dir("rnd")
    other = _run_dir("bb1data")
    blackbox.score_sample("rnd", "bb2", subsample=True, chunk=100, n_boot=0, log=quiet)
    (rd / "bb2_scores.npz").unlink()
    part = rd / "_partial_bb2_sub"
    part.mkdir()
    (part / "signature.txt").write_text("model deadbeef n=0 chunk=100 first=-1 last=-1")
    (part / "chunk_00000.npy").write_bytes(b"stale")
    blackbox.score_sample("rnd", "bb2", subsample=True, chunk=100, n_boot=0, log=quiet)
    assert np.isfinite(np.load(rd / "bb2_scores.npz")["jet_scores"]).all()
    assert other.exists()


def test_baseline_runs_dispatch_to_scorers(env, runs, monkeypatch):
    """A run whose config `model` is in gaehep.baselines.SCORERS is scored by that function (stub: P3 adds
    the real module in parallel)."""
    d = Data(N_MAX)
    rd = _run_dir("obs_stub")
    rd.mkdir(parents=True)
    (rd / "config.yaml").write_text(yaml.safe_dump({"model": "observable", "data": {"n_const": N_MAX}}))
    # F_B: a smooth reference of jet multiplicities
    ref = np.linspace(0.0, 40.0, 400)[:, None] * np.ones((1, 2))
    np.savez(rd / "scores.npz", bkg_val=ref, bkg_test=ref)
    calls = []

    def scorer(run_dir, data, split):
        calls.append((run_dir, type(data), split))
        return data.table(split)["n_const"].astype(np.float64)  # larger multiplicity = more anomalous

    stub = types.ModuleType("gaehep.baselines")
    stub.SCORERS = {"observable": scorer}
    monkeypatch.setitem(sys.modules, "gaehep.baselines", stub)
    assert blackbox.baseline_scorers() == {"observable": scorer}
    blackbox.score_sample("obs_stub", "bb1", subsample=True, n_boot=0, log=quiet)
    assert calls == [(rd, Data, "bb1_test_sub")]
    scores, m = _check_files("obs_stub", "bb1", True, env)
    assert scores["jet_scores"].dtype == np.float32
    assert np.array_equal(scores["jet_scores"], d.store("bb1")["n_const"][scores["event_index"]])
    assert (
        m["event"]["prod"]["bb1/full"]["auc"] > 0.95
    )  # the signal has 12 constituents per jet, background 5
    calls.clear()
    blackbox.score_sample("obs_stub", "bb1", subsample=True, n_boot=0, log=quiet)
    assert calls == []  # up to date
    monkeypatch.setitem(stub.SCORERS, "observable", lambda *a: np.zeros((3, 2)))
    with pytest.raises(ValueError, match="shape"):
        blackbox.score_sample("obs_stub", "bb1", subsample=True, rescore=True, log=quiet)


def test_real_baselines_score_the_black_boxes(env):
    """The P3 baselines (gaehep.baselines.SCORERS) through `gaehep score-sample`, on the fake boxes.  The boxes
    have no tau values (NaN): tau-based observables are constant there and hl_density imputes the missing taus."""
    from gaehep import baselines

    assert set(blackbox.baseline_scorers()) == set(baselines.SCORERS) == set(blackbox.BASELINE_MODELS)
    data = Data(N_MAX)
    baselines.run_obs_baselines(data=data, n_boot=0, log=quiet)
    baselines.run_hl_density(1, data=data, epochs=1, n_boot=0, log=quiet)
    baselines.run_efp_knn(data=data, k=3, n_ref=100, n_boot=0, log=quiet)
    for run in ("study_obs_nconst", "study_obs_tau21", "study_hldensity_s1", "study_efpknn"):
        for sample, sub in (("bb1", True), ("bb3", False), ("bb2", True)):
            blackbox.score_sample(run, sample, subsample=sub, n_boot=0, log=quiet)
            _, m = _check_files(run, sample, sub, env)
            if sample != "bb2":
                assert m["event"]["prod"][f"{sample}/full"]["n_sig"] > 0
    # the 12-constituent signal jets of bb1 are far from the 5-constituent background in n_const (judged on the
    # raw 'max' score: the fake boxes lie entirely below the fake R&D reference, so the CDF behind 'prod'
    # saturates; the direction of the EFP k-NN score on these toy jets means nothing and is not asserted)
    nconst = load_json(_run_dir("study_obs_nconst") / "bb1_metrics.json")["metrics"]["event"]["max"]
    assert nconst["bb1/full"]["auc"] > 0.95
    # a baseline's scores are those of its scorer on the whole split
    split_scores = baselines.SCORERS["efp_knn"](_run_dir("study_efpknn"), data, "bb1_test_sub")
    with np.load(_run_dir("study_efpknn") / "bb1_scores.npz") as f:
        assert np.array_equal(f["jet_scores"], split_scores.astype(np.float32))


def test_baseline_without_the_module_is_an_error(env, runs, monkeypatch):
    rd = _run_dir("hl_stub")
    rd.mkdir(parents=True)
    (rd / "config.yaml").write_text(yaml.safe_dump({"model": "hl_density", "data": {"n_const": N_MAX}}))
    monkeypatch.setattr(blackbox, "baseline_scorers", lambda: {})
    with pytest.raises(RuntimeError, match="hl_density"):
        blackbox.score_sample("hl_stub", "bb1", subsample=True, log=quiet)
    if importlib.util.find_spec("gaehep.baselines") is None:  # P3 not merged: the guarded import gives {}
        monkeypatch.undo()
        assert blackbox.baseline_scorers() == {}


def test_score_sample_errors(env, runs):
    with pytest.raises(ValueError, match="sample"):
        blackbox.score_sample("rnd", "rnd", log=quiet)
    rd = _run_dir("no_fb")
    rd.mkdir(parents=True)
    (rd / "config.yaml").write_text((_run_dir("rnd") / "config.yaml").read_text())
    (rd / "model.pt").write_bytes((_run_dir("rnd") / "model.pt").read_bytes())
    np.savez(rd / "scores.npz", bkg_test=np.zeros((4, 2), np.float32))  # no bkg_val: F_B is undefined
    with pytest.raises(KeyError, match="bkg_val"):
        blackbox.score_sample("no_fb", "bb1", subsample=True, log=quiet)


def test_score_sample_cli(env, runs, capsys):
    rc = cli.main(["score-sample", "rnd", "--sample", "bb1", "--subsample", "--n-boot", "0", "--rescore"])
    assert rc == 0
    assert "bb1_test_sub" in capsys.readouterr().out
    assert json.loads((_run_dir("rnd") / "bb1_metrics.json").read_text())["subsample"] is True
    with pytest.raises(SystemExit):
        cli.main(["score-sample", "rnd", "--sample", "rnd"])
