"""``gaehep posthoc``: paired and Holm-corrected contrasts, the low-fit sensitivity, the verdict rules and the
training-length (H4) signs, on synthetic metrics files (no model, no data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from gaehep.eval import posthoc as P
from gaehep.eval.study import collect

ENCODERS = ("deepsets", "edgeconv", "transformer", "gcn_complete", "efn", "ewmp", "ew_attention")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def write_run(root: Path, name: str, enc: str, auc3: float, bb1: float, bb3: float, jsd: float, loss: float):
    d = root / "runs" / name

    def e(auc):
        return {"auc": auc, "max_sic": 2 * auc, "rej@0.3": 10 * auc}

    write_json(
        d / "metrics.json",
        {
            "metrics": {
                "event": {"prod": {"sig_test/SR": e(0.85), "sig3_test/SR": e(auc3)}},
                "sculpting": {"prod": {"mjj": {"jsd@0.01": jsd}}},
            }
        },
    )
    write_json(d / "bb1_metrics.json", {"metrics": {"event": {"prod": {"bb1/window": e(bb1)}}}})
    write_json(d / "bb3_metrics.json", {"metrics": {"event": {"prod": {"bb3/window": e(bb3)}}}})
    write_json(d / "history.json", {"status": "done", "best_val_loss": loss})
    (d / "config.yaml").write_text(yaml.safe_dump({"model": "ef_ae", "model_args": {"encoder": enc}}))


def build(root: Path, curves: bool = False) -> None:
    """Interaction encoders better on the 3-prong signal by 0.02, equal elsewhere; deepsets has two low-fit seeds
    with a much lower 3-prong AUC; IRC-safe encoders have lower JSD."""
    rng = np.random.default_rng(0)
    base3 = {
        "deepsets": 0.78,
        "edgeconv": 0.80,
        "transformer": 0.80,
        "gcn_complete": 0.70,
        "efn": 0.80,
        "ewmp": 0.82,
        "ew_attention": 0.82,
    }
    for enc in ENCODERS:
        for s in range(1, 6):
            low = enc == "deepsets" and s in (2, 4)
            auc3 = base3[enc] - (0.07 if low else 0.0) + rng.normal(0, 0.002)
            jsd = (0.008 if enc in ("efn", "ewmp", "ew_attention") else 0.012) + rng.normal(0, 0.0003)
            write_run(
                root,
                f"study_{enc}_s{s}",
                enc,
                auc3,
                0.91 + rng.normal(0, 0.002),
                0.47 + rng.normal(0, 0.002),
                jsd,
                0.40 if low else 0.25 + rng.normal(0, 0.005),
            )
        if curves:
            for ep in (1, 3, 5):
                for s in (1, 2, 3):
                    flip = ep == 1 and enc == "edgeconv"  # edgeconv falls below deepsets at 1 epoch only
                    auc3 = (0.70 if flip else base3[enc]) + rng.normal(0, 0.002)
                    write_run(root, f"stct_{enc}_e{ep}_s{s}", enc, auc3, 0.91, 0.47, 0.01, 0.25)
    contrasts = {
        f"{a} - {b}": {
            k: {"auc": {"bootstrap": {"excludes_zero": True}}} for k in ("bb1/window", "bb3/window")
        }
        for a in ENCODERS
        for b in ENCODERS
        if a != b
    }
    write_json(root / "study" / "study_contrasts.json", {"contrasts": contrasts})


def test_holm_steps_down_and_stops():
    assert P.holm([0.01, 0.04, 0.03, None]) == [True, False, False, None]  # 0.01 <= 0.05/3, 0.03 > 0.05/2
    assert P.holm([0.001, 0.02, 0.04]) == [True, True, True]


def test_paired_uses_common_seeds_and_differences():
    out = P.paired({1: 1.0, 2: 2.0, 3: 3.0}, {1: 0.5, 2: 1.5, 4: 9.0})
    assert out["seeds"] == [1, 2] and out["delta"] == pytest.approx(0.5) and out["se"] == pytest.approx(0.0)
    assert out["sig2se"] is True


def test_welch_two_standard_errors():
    w = P.welch([1.0, 1.1, 0.9], [0.0, 0.1, -0.1])
    assert w["delta"] == pytest.approx(1.0) and w["sig2se"] and w["p"] < 0.01


def test_grid_low_fit_sensitivity_and_verdicts(tmp_path):
    build(tmp_path)
    configs = [c for c in collect("study", tmp_path) if len(c.runs) > 1]
    rows = P.contrast_grid(
        configs, json.loads((tmp_path / "study" / "study_contrasts.json").read_text())["contrasts"]
    )
    r = next(x for x in rows if (x["a"], x["b"], x["signal_key"]) == ("edgeconv", "deepsets", "rd3"))
    assert r["low_fit_removed"] == ["study_deepsets_s2", "study_deepsets_s4"]
    assert r["delta"] == pytest.approx(0.02 + 0.07 * 2 / 5, abs=0.004)
    assert r["fit_delta"] == pytest.approx(0.02, abs=0.004)
    v = P.verdicts(rows, "protocol")
    assert v["H1"] == "inconclusive"  # positive on the 3-prong signal only
    assert (
        v["H2b"] == "supported"
    )  # two of three IRC-safe encoders have significantly lower JSD... all three here
    assert v["H3"] == "supported"  # gcn_complete is never significantly better


def test_h4_signs_and_pending(tmp_path):
    build(tmp_path, curves=True)
    configs = [c for c in collect("study", tmp_path) if len(c.runs) > 1]
    rows = P.contrast_grid(configs, None)
    out = P.h4(rows, [c for c in collect("stct", tmp_path) if c.runs])
    row = next(h for h in out if h["contrast"] == "edgeconv - deepsets" and h["signal"] == "R&D 3-prong")
    assert row["signs"] == {1: -1, 3: 1, 5: 1} and row["status"] == "flips"
    assert P.h4(rows, []) and all(h["status"] == "pending" for h in P.h4(rows, []))


def test_posthoc_writes_report_json_and_figure(tmp_path):
    build(tmp_path, curves=True)
    paths = P.posthoc("study", "stct", root=tmp_path, log=lambda *a: None)
    md = paths["md"].read_text()
    assert "| H1 |" in md and "Low-fit runs removed" in md
    data = json.loads(paths["json"].read_text())
    assert set(data["verdicts"]) == set(P.VARIANTS) and data["h4"]
    assert paths["png"].stat().st_size > 1000


def test_posthoc_without_runs_raises(tmp_path):
    with pytest.raises(ValueError):
        P.posthoc("study", "", root=tmp_path, log=lambda *a: None)


def test_fit_adjusted_removes_a_difference_that_is_only_fit_quality(tmp_path):
    rng = np.random.default_rng(3)
    loss = {e: 0.2 + 0.05 * i for i, e in enumerate(ENCODERS)}
    for enc in ENCODERS:
        for s in range(1, 6):
            lo = loss[enc] + rng.normal(0, 0.01)
            a = 0.95 - 0.5 * lo + rng.normal(0, 0.001)  # the AUC depends on the fit only
            write_run(tmp_path, f"study_{enc}_s{s}", enc, a, a, a, 0.01, lo)
    configs = [c for c in collect("study", tmp_path) if len(c.runs) > 1]
    raw = {(r["a"], r["b"], r["signal_key"]): r for r in P.contrast_grid(configs, None)}
    adj = P.fit_adjusted(configs)
    row = next(c for c in adj["rd3"]["contrasts"] if (c["a"], c["b"]) == ("edgeconv", "deepsets"))
    assert abs(raw[("edgeconv", "deepsets", "rd3")]["delta"]) > 0.02
    assert abs(row["delta"]) < 0.003 and not row["sig2se"]
