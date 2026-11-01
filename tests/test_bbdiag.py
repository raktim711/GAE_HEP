"""``gaehep bb-diagnose``: observable AUCs per signal set and region, score shift and mass correlation."""

from __future__ import annotations

import numpy as np
import pytest

from gaehep.eval import bbdiag as D


def toy(n_bkg=4000, n_sig=400, seed=0):
    rng = np.random.default_rng(seed)
    n = n_bkg + n_sig
    label = np.r_[np.zeros(n_bkg, int), np.ones(n_sig, int)]
    m = np.where(label[:, None] == 1, rng.normal(80, 10, (n, 2)), rng.normal(150, 30, (n, 2)))  # light signal
    mjj = np.where(label == 1, rng.normal(4200, 50, n), rng.uniform(3000, 5000, n))
    scores = m / 100 + rng.normal(0, 0.1, (n, 2))  # a mass-like score
    return label, m, mjj, scores


def test_observables_report_aucs_and_directions():
    label, m, mjj, _ = toy()
    rep = D.observables_report({"m": m}, mjj, {"sig": label == 1}, label == 0, (4000.0, 4400.0))
    w = rep["sig/window"]
    assert w["n_sig_events"] > 300 and 0 < w["n_bkg_events"] < 1000
    assert w["auc_m"] < 0.1 and w["auc_-m"] == pytest.approx(1 - w["auc_m"])
    assert w["mean_m_sig"] < w["mean_m_bkg"]


def test_run_report_detects_a_shift_and_the_mass_correlation():
    label, m, mjj, s = toy()
    rnd = s[label == 0][:2000] - 0.5  # the R&D background scores lower: the box background looks shifted
    rep = D.run_report(
        s, mjj, {"sig": label == 1}, label == 0, (4000.0, 4400.0), m, rnd, m[label == 0][:2000]
    )
    assert rep["shift"]["ks_jet"] > 0.5 and rep["shift"]["frac_above_rnd_q90"] > 0.5
    assert rep["mass"]["spearman_score_mJ_box_bkg"] > 0.9
    assert rep["score"]["sig/window"] < 0.2  # a mass-like score ranks the light signal as background


def test_run_report_without_reference_scores():
    label, m, mjj, s = toy()
    rep = D.run_report(s, mjj, {"sig": label == 1}, label == 0, None, m, None, None)
    assert "shift" not in rep and set(rep["score"]) == {"sig/full"}


def test_render_lists_runs():
    label, m, mjj, s = toy()
    res = {
        "sample": "bb3",
        "observables": D.observables_report({"m": m}, mjj, {"sig": label == 1}, label == 0, None),
        "runs": {
            "r": D.run_report(s, mjj, {"sig": label == 1}, label == 0, None, m, s[label == 0], m[label == 0])
        },
    }
    md = D.render(res)
    assert "| r |" in md and "sig/full" in md
