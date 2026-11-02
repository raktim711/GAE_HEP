"""E3(a) check: BB1 metrics of the BB1-trained runs on all BB1 events and on the validation part alone
(docs/STUDY_PROTOCOL.md, section 5, "Train-on-data scenario").

The validation part is the 10 % of the BB1 events with at least two jets that ``gaehep.data.splits.
train_val_split`` (seed 20201) sets aside; its jets are the ones F_B is built from, and the model never trains on
them.  Labels are used only here, for evaluation.  Needs ``bb1_scores.npz`` of the full box (``gaehep
score-sample <run> --sample bb1`` without ``--subsample``).

    python scripts/e3_valpart.py study_bb1data_edgeconv_s1 ... --out results/study/e3_bb1data_valpart.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gaehep.config import BLACKBOX_WINDOWS, save_json
from gaehep.data.splits import load_splits, train_val_split
from gaehep.eval.evaluate import aggregate, jet_reference, region_mask
from gaehep.eval.metrics import summarize
from gaehep.train.run import _run_dir

KEYS = ("auc", "max_sic", "rej@0.3")


def valpart_run(run: str) -> dict:
    run_dir = _run_dir(run)
    window = BLACKBOX_WINDOWS["bb1"]
    with np.load(run_dir / "scores.npz") as f:
        ref = jet_reference({"bkg_val": f["bkg_val"]})
    with np.load(run_dir / "bb1_scores.npz") as f:
        jet_scores, mjj, label, events, sub = (
            f["jet_scores"],
            f["mjj"],
            f["label"],
            f["event_index"],
            bool(f["subsample"]),
        )
    if sub:
        raise ValueError(f"{run}: bb1_scores.npz is the subsample; score the full box")
    _, val = train_val_split(load_splits()["bb1_test"])
    in_val = np.isin(events, val)
    ev = aggregate(jet_scores, "prod", ref)
    out = {"name": run, "n_val_events": int(in_val.sum())}
    for part, pm in (("all", np.ones(len(ev), bool)), ("val", in_val)):
        for region in ("window", "full"):
            m = pm & region_mask(mjj, region, window)
            s = summarize(ev[m & (label > 0)], ev[m & (label == 0)])
            out[f"{part}/{region}"] = {k: s[k] for k in ("n_sig", "n_bkg", *KEYS)}
    save_json(out, run_dir / "bb1_valpart.json")
    return out


def table(results: list[dict]) -> str:
    lines = [
        "| run | part | region | n_sig | n_bkg | AUC | max SIC | 1/eps_B@0.3 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for res in results:
        for part in ("all", "val"):
            for region in ("window", "full"):
                e = res[f"{part}/{region}"]
                lines.append(
                    f"| {res['name']} | {part} | {region} | {e['n_sig']} | {e['n_bkg']} "
                    f"| {e['auc']:.4f} | {e['max_sic']:.2f} | {e['rej@0.3']:.1f} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    results = [valpart_run(r) for r in a.runs]
    md = table(results)
    print(md)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(md)


if __name__ == "__main__":
    main()
