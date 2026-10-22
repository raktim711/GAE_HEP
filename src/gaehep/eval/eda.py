"""Exploratory plots of the processed LHCO sample: m_jj, jet masses, tau21, multiplicity (bkg vs signals).

python -m gaehep.eval.eda            # -> results/eda/*.png + results/eda/summary.json
"""

from __future__ import annotations

import sys

import numpy as np

from gaehep.config import SR_MJJ, results_dir, run_metadata, save_json
from gaehep.eval import style

style.apply()
import matplotlib.pyplot as plt  # noqa: E402

from gaehep.train.run import Data  # noqa: E402

LABELS = {"bkg_test": "QCD background", "sig_test": "2-prong W'", "sig3_test": "3-prong W'"}


def _hist(ax, data: dict[str, np.ndarray], bins, xlabel: str, logy: bool = False) -> None:
    for k, v in data.items():
        v = np.asarray(v, dtype=np.float64)
        v = v[np.isfinite(v)]
        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.6, label=LABELS.get(k, k))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    if logy:
        ax.set_yscale("log")
    ax.legend()


def main() -> int:
    data = Data(n_const=50)
    tabs = {k: data.table(k) for k in LABELS if data.has(k)}
    out = results_dir() / "eda"
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    _hist(
        axes[0, 0],
        {k: t["mjj"] for k, t in tabs.items()},
        np.linspace(1500, 6000, 90),
        r"$m_{jj}$ [GeV]",
        True,
    )
    for x in SR_MJJ:
        axes[0, 0].axvline(x, color="grey", ls=":", lw=1)
    _hist(
        axes[0, 1],
        {k: np.nanmax(t["m"], 1) for k, t in tabs.items()},
        np.linspace(0, 900, 90),
        r"heavier jet mass [GeV]",
    )
    _hist(
        axes[0, 2],
        {k: np.nanmin(t["m"], 1) for k, t in tabs.items()},
        np.linspace(0, 600, 90),
        r"lighter jet mass [GeV]",
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        _hist(
            axes[1, 0],
            {k: (t["tau2"] / t["tau1"]).ravel() for k, t in tabs.items()},
            np.linspace(0, 1, 60),
            r"$\tau_{21}$ (both jets)",
        )
        _hist(
            axes[1, 1],
            {k: (t["tau3"] / t["tau2"]).ravel() for k, t in tabs.items()},
            np.linspace(0, 1, 60),
            r"$\tau_{32}$ (both jets)",
        )
    _hist(
        axes[1, 2],
        {k: t["n_const"].ravel() for k, t in tabs.items()},
        np.arange(0, 250, 4),
        "constituents per jet",
    )
    fig.tight_layout()
    fig.savefig(out / "overview.png")
    plt.close(fig)

    summary = {}
    for k, t in tabs.items():
        sr = (t["mjj"] >= SR_MJJ[0]) & (t["mjj"] <= SR_MJJ[1])
        nc = t["n_const"].ravel()
        summary[k] = {
            "n_events": len(t["mjj"]),
            "frac_in_SR": float(sr.mean()),
            "mjj_median": float(np.median(t["mjj"])),
            "n_const_median": float(np.median(nc)),
            "frac_jets_n_const_gt_50": float(np.mean(nc > 50)),
            "frac_jets_n_const_gt_100": float(np.mean(nc > 100)),
        }
    save_json({"summary": summary, **run_metadata()}, out / "summary.json")
    for k, v in summary.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
