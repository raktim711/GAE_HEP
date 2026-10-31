"""E3(a): bump hunt on the BB1 m_jj spectrum with the score of a run trained on BB1 (docs/STUDY_PROTOCOL.md,
section 5, "Train-on-data scenario").

For every run given, read ``bb1_scores.npz`` (all BB1 events, written by ``gaehep score-sample <run> --sample
bb1`` without ``--subsample``), form the ``prod`` event score with F_B from the run's ``scores.npz['bkg_val']``
(for a BB1-trained run: BB1's validation part), and run the existing ``gaehep.eval.bumphunt.search`` in the
BB1 window (3.6-4.0 TeV) with no cut and with sideband-defined cuts that keep 10 % and 2 % of the sideband
events.  The significance is the local one of ``search``.  The labels enter only the count of true signal events
kept in the window (a check, not part of the search).

Writes ``<run>/bb1_bumphunt.json`` and, with ``--out``, a markdown table.

    python scripts/e3_bumphunt.py study_bb1data_edgeconv_s1 ... --out results/study/e3_bb1data_bumphunt.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gaehep.config import BLACKBOX_WINDOWS, run_metadata, save_json
from gaehep.eval.bumphunt import search
from gaehep.eval.evaluate import aggregate, jet_reference
from gaehep.train.run import _run_dir

EFFS = (1.0, 0.1, 0.02)  # no cut, and cuts keeping 10 % and 2 % of the sideband events


def bumphunt_run(run: str, sample: str = "bb1") -> dict:
    run_dir = _run_dir(run)
    window = BLACKBOX_WINDOWS[sample]
    with np.load(run_dir / "scores.npz") as f:
        ref = jet_reference({"bkg_val": f["bkg_val"]})
    with np.load(run_dir / f"{sample}_scores.npz") as f:
        jet_scores, mjj, label, sub = (
            f["jet_scores"],
            f["mjj"].astype(np.float64),
            f["label"],
            bool(f["subsample"]),
        )
    score = aggregate(jet_scores, "prod", ref)
    in_win = (mjj >= window[0]) & (mjj < window[1])
    rows = []
    for eff in EFFS:
        r = search(mjj, score, eff, sr=window)
        sel = np.ones(len(mjj), bool) if eff >= 1 else score >= r["threshold"]
        r["n_sig_window_kept"] = int(np.sum(sel & in_win & (label > 0)))
        r["n_sig_window"] = int(np.sum(in_win & (label > 0)))
        rows.append(r)
    out = {
        "name": run,
        "sample": sample,
        "subsample": sub,
        "window": list(window),
        "n_events": len(mjj),
        "agg": "prod",
        "rows": rows,
        **{k: v for k, v in run_metadata().items() if k in ("git", "time", "splits_sha256")},
    }
    save_json(out, run_dir / f"{sample}_bumphunt.json")
    return out


def table(results: list[dict]) -> str:
    lines = [
        "| run | cut keeps (sideband) | n_SR | b_SR (fit) | Z_local | true signal kept in window | fit ok |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for res in results:
        for r in res["rows"]:
            keep = "no cut" if r["eff"] >= 1 else f"{100 * r['eff']:g} %"
            lines.append(
                f"| {res['name']} | {keep} | {r['n_sr']:.0f} | {r['b_sr']:.1f} | {r['z']:+.2f} "
                f"| {r['n_sig_window_kept']}/{r['n_sig_window']} | {r['fit_ok']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", type=Path, default=None, help="write a markdown table here")
    a = p.parse_args()
    results = [bumphunt_run(r) for r in a.runs]
    md = table(results)
    print(md)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(md)


if __name__ == "__main__":
    main()
