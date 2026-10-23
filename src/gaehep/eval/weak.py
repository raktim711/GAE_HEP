"""Idealised weakly supervised anomaly detection (IAD) on top of fixed jet representations.

Resonant anomaly detection in the CATHODE / "Back To The Roots" convention: in the signal region
(SR, 3.3 < m_jj < 3.7 TeV) a classifier separates "data" (background + an injected amount of signal) from a
perfectly modelled background template (independent background events).  If the event representation
carries the information that distinguishes the signal, the classifier output is a monotone function of
p_data / p_bkg = 1 + (S/B) p_sig / p_bkg, i.e. of the optimal anomaly score.  No event-level truth label is
used; the quality at each injection is measured on independent test events (AUC, max SIC).  The *idealised*
template (more background from the same simulation) is an upper bound for data-driven templates (CATHODE).

Event representations are symmetric in the two jets (jets ordered by mass, lighter first):

* ``hl4``    (m_J1, m_J2 - m_J1, τ21_J1, τ21_J2): the standard benchmark features;
* ``score+m`` the GNN's per-jet anomaly scores and the jet masses (s_J1, s_J2, m_J1, m_J2): HL4 with the
  learned score in place of τ21;
* ``latent`` the permutation-invariant GNN's per-jet latent vectors (z_J1, z_J2);
* ``latent+m`` latents plus the two jet masses.

The classifier is a small ensemble of histogram gradient-boosted trees (the BDT choice of "Back To The
Roots", which is robust for this task); a fully supervised classifier on the same features (signal vs
background labels) is reported as the per-representation upper bound.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from gaehep.config import SR_MJJ
from gaehep.eval.metrics import auc, max_sic


def mass_order(m: np.ndarray) -> np.ndarray:
    """(n, 2) column order putting the lighter jet first."""
    return np.argsort(m, axis=1, kind="stable")


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b > 0, a / np.where(b > 0, b, 1.0), 1.0)
    return np.nan_to_num(r, nan=1.0)


def hl4(tab: dict[str, np.ndarray]) -> np.ndarray:
    o = mass_order(tab["m"])
    m = np.take_along_axis(tab["m"], o, 1)
    t21 = np.take_along_axis(_ratio(tab["tau2"], tab["tau1"]), o, 1)
    return np.c_[m[:, 0], m[:, 1] - m[:, 0], t21[:, 0], t21[:, 1]].astype(np.float32)


def latent_features(tab: dict[str, np.ndarray], lat: np.ndarray, with_mass: bool = False) -> np.ndarray:
    """lat: (n, 2, d) per-jet latents in the table's (pT) jet order."""
    o = mass_order(tab["m"])
    z = np.take_along_axis(lat, o[..., None], 1).reshape(len(lat), -1)
    if with_mass:
        z = np.c_[z, np.take_along_axis(tab["m"], o, 1)]
    return z.astype(np.float32)


def _clf(seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.1,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=seed,
    )


def classifier_scores(
    x_pos: np.ndarray, x_neg: np.ndarray, x_test: np.ndarray, n_models: int = 5, seed: int = 0
) -> np.ndarray:
    """Ensemble-averaged P(pos | x) on ``x_test`` from classifiers trained on pos-vs-neg (class balanced)."""
    x = np.r_[x_pos, x_neg]
    y = np.r_[np.ones(len(x_pos)), np.zeros(len(x_neg))]
    w = np.where(y == 1, 0.5 * len(y) / len(x_pos), 0.5 * len(y) / len(x_neg))
    out = np.zeros(len(x_test))
    rng = np.random.default_rng(seed)
    for i in range(n_models):
        sel = rng.random(len(x)) < 0.8  # bagging: each model sees a random 80% of the training events
        clf = _clf(seed * 100 + i).fit(x[sel], y[sel], sample_weight=w[sel])
        out += clf.predict_proba(x_test)[:, 1]
    return out / n_models


def run_iad(
    feats: dict[str, dict[str, np.ndarray]],
    n_inj: Sequence[int] = (150, 300, 600, 1200, 3000),
    reps: Sequence[int] = (0, 1, 2),
    n_models: int = 5,
    log=print,
) -> dict[str, Any]:
    """IAD study for several representations.

    ``feats[rep]`` holds SR-only event features: ``bkg_train`` (split in two halves: data background and
    template), ``sig_train`` (injection pool), ``bkg_test`` and ``sig_test`` (evaluation).
    Returns {rep: {"iad": [{n_inj, s_over_b, rep, auc, max_sic, eff_s}], "supervised": {...}}}.
    """
    res: dict[str, Any] = {}
    for rep, f in feats.items():
        n_b = len(f["bkg_train"])
        perm = np.random.default_rng(1234).permutation(n_b)
        data_b, templ = f["bkg_train"][perm[: n_b // 2]], f["bkg_train"][perm[n_b // 2 :]]
        x_test = np.r_[f["sig_test"], f["bkg_test"]]
        n_s = len(f["sig_test"])
        rows = []
        for n in [n for n in n_inj if n <= len(f["sig_train"])]:  # no injection larger than the signal pool
            for r in reps:
                pick = np.random.default_rng([r, n]).choice(len(f["sig_train"]), n, replace=False)
                s = classifier_scores(np.r_[data_b, f["sig_train"][pick]], templ, x_test, n_models, seed=r)
                sic, eff = max_sic(s[:n_s], s[n_s:])
                rows.append(
                    {
                        "n_inj": n,
                        "s_over_b": n / len(data_b),
                        "rep": r,
                        "auc": auc(s[:n_s], s[n_s:]),
                        "max_sic": sic,
                        "eff_s": eff,
                    }
                )
            last = [x for x in rows if x["n_inj"] == n]
            log(
                f"[iad:{rep}] n_inj {n:5d} (S/B {n / len(data_b):.2%}): AUC "
                f"{np.mean([x['auc'] for x in last]):.3f}  max SIC {np.mean([x['max_sic'] for x in last]):.2f}"
            )
        s = classifier_scores(f["sig_train"], f["bkg_train"], x_test, n_models, seed=0)
        sup = {"auc": auc(s[:n_s], s[n_s:]), "max_sic": max_sic(s[:n_s], s[n_s:])[0]}
        log(f"[iad:{rep}] supervised: AUC {sup['auc']:.3f}  max SIC {sup['max_sic']:.2f}")
        res[rep] = {"iad": rows, "supervised": sup, "n_bkg_data": len(data_b), "n_template": len(templ)}
    return res


def sr_mask(mjj: np.ndarray) -> np.ndarray:
    return (mjj >= SR_MJJ[0]) & (mjj <= SR_MJJ[1])
