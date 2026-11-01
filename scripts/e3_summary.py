"""E3 summary tables that `gaehep study-table` does not cover (markdown on stdout, optionally to a file):

* the bump hunt on BB1 for the BB1-trained runs (``bb1_bumphunt.json``, written by ``e3_bumphunt.py``): the local
  significance without a cut and with the cuts keeping 10 % and 2 % of the sideband events, per run and as the mean
  over seeds, and the BB1 metrics on the validation part alone (``bb1_valpart.json``, ``e3_valpart.py``);
* the BB2 null test of the 100-constituent runs (``bb2_metrics.json``): the bump hunt in the three BB2 windows
  and the KS distance to the R&D ``bkg_test`` scores;
* training wall time and flags (``history.json``: ``train_minutes``, ``best_val_loss``).

    python scripts/e3_summary.py --out results/study/e3_summary.md
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from gaehep.config import load_json, results_dir

ENCS_BB1 = ("edgeconv", "ewmp", "efn")
ENCS_N100 = ("edgeconv", "ewmp", "transformer")
SEEDS = (1, 2, 3)
EFFS = (1.0, 0.1, 0.02)
WINDOWS = ((3300.0, 3700.0), (3600.0, 4000.0), (4000.0, 4400.0))


def _load(name: str, fname: str):
    p = results_dir() / "runs" / name / fname
    return load_json(p) if p.exists() else None


def _eff_label(eff: float) -> str:
    return "no cut" if eff >= 1 else f"{100 * eff:g} %"


def bb1_bumphunt() -> list[str]:
    out = [
        "### BB1 bump hunt (BB1-trained runs; window 3.6-4.0 TeV; local Z)",
        "",
        "| encoder | seed | Z no cut | Z 10 % cut | Z 2 % cut | true signal kept (10 % / 2 %) of 512 |",
        "|---|--:|--:|--:|--:|---|",
    ]
    means = {}
    for enc in ENCS_BB1:
        zs = {e: [] for e in EFFS}
        for s in SEEDS:
            res = _load(f"study_bb1data_{enc}_s{s}", "bb1_bumphunt.json")
            if not res:
                continue
            rows = {r["eff"]: r for r in res["rows"]}
            for e in EFFS:
                zs[e].append(rows[e]["z"])
            out.append(
                f"| {enc} | {s} | {rows[1.0]['z']:+.2f} | {rows[0.1]['z']:+.2f} | {rows[0.02]['z']:+.2f} "
                f"| {rows[0.1]['n_sig_window_kept']} / {rows[0.02]['n_sig_window_kept']} |"
            )
        means[enc] = zs
    out += [
        "",
        "Mean +- standard deviation of Z over seeds:",
        "",
        "| encoder | seeds | no cut | 10 % cut | 2 % cut |",
        "|---|--:|--:|--:|--:|",
    ]
    for enc, zs in means.items():
        if not zs[1.0]:
            continue
        cells = [
            f"{st.mean(zs[e]):+.2f} +- {st.stdev(zs[e]) if len(zs[e]) > 1 else float('nan'):.2f}"
            for e in EFFS
        ]
        out.append(f"| {enc} | {len(zs[1.0])} | " + " | ".join(cells) + " |")
    return [*out, ""]


def bb1_valpart() -> list[str]:
    out = [
        "### BB1 metrics on all events and on the validation part alone (event `prod` AUC, mean +- sd over seeds)",
        "",
        "| encoder | seeds | window, all | window, validation part | full, all | full, validation part |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for enc in ENCS_BB1:
        v = {k: [] for k in ("all/window", "val/window", "all/full", "val/full")}
        for s in SEEDS:
            res = _load(f"study_bb1data_{enc}_s{s}", "bb1_valpart.json")
            if res:
                for k in v:
                    v[k].append(res[k]["auc"])
        if v["all/window"]:
            cells = [
                f"{st.mean(x):.4f} +- {st.stdev(x) if len(x) > 1 else float('nan'):.4f}" for x in v.values()
            ]
            out.append(f"| {enc} | {len(v['all/window'])} | " + " | ".join(cells) + " |")
    return [*out, ""]


def bb2_null() -> list[str]:
    out = [
        "### BB2 null test of the R&D-trained 100-constituent runs (BB2 subsample: 200 000 events; local Z)",
        "",
        "| encoder | seed | KS (`prod`) | "
        + " | ".join(f"Z {int(a)}-{int(b)} GeV: no cut / 10 % / 2 %" for a, b in WINDOWS)
        + " |",
        "|---|--:|--:|" + "---|" * len(WINDOWS),
    ]
    worst = 0.0
    for enc in ENCS_N100:
        for s in SEEDS:
            res = _load(f"study_n100_{enc}_s{s}", "bb2_metrics.json")
            if not res:
                continue
            bh = res["metrics"]["bumphunt"]
            cells = []
            for w in WINDOWS:
                zs = [next(r["z"] for r in bh if r["window"] == list(w) and r["eff"] == e) for e in EFFS]
                worst = max(worst, *(abs(z) for z in zs))
                cells.append(" / ".join(f"{z:+.2f}" for z in zs))
            out.append(f"| {enc} | {s} | {res['metrics']['ks']['prod']:.3f} | " + " | ".join(cells) + " |")
    return [*out, "", f"Largest |Z| over all runs, windows and cuts: {worst:.2f}", ""]


def bb2_null_e1() -> list[str]:
    """BB2 bump hunt of the E1 main-grid runs (50 constituents, R&D-trained): largest |Z| per encoder."""
    import re

    runs = sorted(
        d.name
        for d in (results_dir() / "runs").glob("study_*_s[0-9]")
        if (d / "bb2_metrics.json").exists() and not re.match(r"study_(n100|bb1data)_", d.name)
    )
    by_enc: dict[str, list[str]] = {}
    for r in runs:
        by_enc.setdefault(r.rsplit("_s", 1)[0].removeprefix("study_"), []).append(r)
    out = [
        "### BB2 null test of the E1 main-grid runs (50 constituents, R&D-trained; BB2 subsample; local Z)",
        "",
        "| encoder | runs | max abs Z, no cut | max abs Z, 10 % cut | max abs Z, 2 % cut | most extreme (run, window, cut) |",
        "|---|--:|--:|--:|--:|---|",
    ]
    worst_all = (0.0, "")
    for enc, rs in by_enc.items():
        mx = {e: 0.0 for e in EFFS}
        worst = (0.0, "")
        for r in rs:
            for row in load_json(results_dir() / "runs" / r / "bb2_metrics.json")["metrics"]["bumphunt"]:
                z = row["z"]
                mx[row["eff"]] = max(mx[row["eff"]], abs(z))
                if abs(z) > abs(worst[0]):
                    worst = (
                        z,
                        f"{r}, {int(row['window'][0])}-{int(row['window'][1])} GeV, {_eff_label(row['eff'])}",
                    )
        if abs(worst[0]) > abs(worst_all[0]):
            worst_all = worst
        out.append(
            f"| {enc} | {len(rs)} | {mx[1.0]:.2f} | {mx[0.1]:.2f} | {mx[0.02]:.2f} | {worst[0]:+.2f} ({worst[1]}) |"
        )
    return [*out, "", f"Most extreme excess over all E1 runs: {worst_all[0]:+.2f} ({worst_all[1]})", ""]


def walltime() -> list[str]:
    out = [
        "### Training wall time and flags",
        "",
        "| group | encoder | seeds | training minutes (sum) | best_val_loss per seed | flagged |",
        "|---|---|--:|--:|---|---|",
    ]
    for group, encs in (("bb1data", ENCS_BB1), ("n100", ENCS_N100)):
        for enc in encs:
            hs = [_load(f"study_{group}_{enc}_s{s}", "history.json") for s in SEEDS]
            hs = [h for h in hs if h]
            if not hs:
                continue
            losses = [h["best_val_loss"] for h in hs]
            med = st.median(losses)
            flagged = [f"s{s}" for s, h in zip(SEEDS, hs, strict=False) if h["best_val_loss"] > 1.5 * med]
            tot = sum(h.get("train_minutes", 0.0) for h in hs)
            out.append(
                f"| {group} | {enc} | {len(hs)} | {tot:.0f} | {', '.join(f'{x:.4f}' for x in losses)} "
                f"| {', '.join(flagged) or 'none'} |"
            )
    return [*out, ""]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    md = "\n".join(
        ["## E3 summary", "", *bb1_bumphunt(), *bb1_valpart(), *bb2_null(), *bb2_null_e1(), *walltime()]
    )
    print(md)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(md)


if __name__ == "__main__":
    main()
