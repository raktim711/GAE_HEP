"""Download the LHC Olympics 2020 datasets from Zenodo with resume support and md5 verification.

Record 6466204 (R&D dataset v5) contains the v2 raw events file that matches the v2 high-level features file.
A plain ``gaehep download`` fetches only this R&D record.  The three black boxes (record 4536624) are fetched
per box with ``gaehep download --sample bb1`` (``bb2``, ``bb3``): the events file and, for ``bb1`` and ``bb3``,
the master key (one label per line, in file order).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from gaehep.config import raw_dir

ZENODO_RECORD = "6466204"
BLACKBOX_RECORD = "4536624"

# filename -> md5 (from the Zenodo record page)
FILES: dict[str, str] = {
    "events_anomalydetection_v2.h5": "629789d55813be3860781b084ae7f1de",
    "events_anomalydetection_v2.features.h5": "271cf5e71fc756b2a8d2b32730689bdb",
    "events_anomalydetection_Z_XY_qqq.h5": "54e123a86143b668f9cb76905152a124",
    "events_anomalydetection_Z_XY_qqq.features.h5": "1e729f7dff225451182c28afaa4bb411",
    "delphes_card_RnD.dat": "cb11b729ec10c04ae5250d057fd088b2",
    "pythia_RnD_qcd.cmnd": "19555e76f8a787184ec43fd5ff295465",
    "pythia_RnD_Z_XY_qq.cmnd": "1e9b731c2bf90f4ba549b85996cd7424",
    "pythia_RnD_Z_XY_qqq.cmnd": "21472daafd7d54cd10e7548869a41d03",
}

# Black-box files (record 4536624), md5 from the record page (docs/STUDY_PROTOCOL.md, section 5)
BLACKBOX_FILES: dict[str, str] = {
    "events_LHCO2020_BlackBox1.h5": "847be306afd923a4ba30543070113627",
    "events_LHCO2020_BlackBox1.masterkey": "c5d9b1e75ce26fd820c1c99dfc874e92",
    "events_LHCO2020_BlackBox2.h5": "2e5dce598238789938cb31e9cd3e7a46",
    "events_LHCO2020_BlackBox3.h5": "ffc53e0c0b42e0f2c752ce48db74ce27",
    "events_LHCO2020_BlackBox3.masterkey": "51e275f122eb48b10ccc2c858dda358e",
}

# --sample -> the record files of that sample (BB2 has no master key: it is background only)
SAMPLE_FILES: dict[str, tuple[str, ...]] = {
    "rnd": tuple(FILES),
    "bb1": ("events_LHCO2020_BlackBox1.masterkey", "events_LHCO2020_BlackBox1.h5"),
    "bb2": ("events_LHCO2020_BlackBox2.h5",),
    "bb3": ("events_LHCO2020_BlackBox3.masterkey", "events_LHCO2020_BlackBox3.h5"),
}

ALL_FILES: dict[str, str] = {**FILES, **BLACKBOX_FILES}


def record_of(name: str) -> str:
    return BLACKBOX_RECORD if name in BLACKBOX_FILES else ZENODO_RECORD


URL_TEMPLATES = (
    "https://zenodo.org/records/{record}/files/{name}?download=1",
    "https://zenodo.org/api/records/{record}/files/{name}/content",
)

CHUNK = 8 << 20  # 8 MiB


def md5sum(path: Path, chunk: int = CHUNK) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _stream(url: str, dest: Path, log_every: float = 30.0) -> None:
    """Stream ``url`` into ``dest``; resume from a partial ``dest`` via an HTTP Range request."""
    have = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "gaehep/0.1"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=120) as resp:
        status = resp.status
        total = resp.headers.get("Content-Length")
        mode = "ab" if (have and status == 206) else "wb"
        if mode == "wb":
            have = 0
        total_bytes = int(total) + have if total is not None else None
        t0 = last = time.time()
        done = have
        with open(dest, mode) as fh:
            while block := resp.read(CHUNK):
                fh.write(block)
                done += len(block)
                now = time.time()
                if now - last > log_every:
                    rate = (done - have) / max(now - t0, 1e-9) / 1e6
                    frac = f"{100 * done / total_bytes:5.1f}%" if total_bytes else "?"
                    print(f"  {dest.name}: {done / 1e6:8.1f} MB ({frac}) at {rate:.1f} MB/s", flush=True)
                    last = now


def download_file(name: str, out_dir: Path, retries: int = 5) -> Path:
    """Download one record file into ``out_dir`` and verify its md5. Returns the final path."""
    expected = ALL_FILES[name]
    dest = out_dir / name
    if dest.exists() and md5sum(dest) == expected:
        print(f"[ok] {name} already present (md5 verified)")
        return dest
    part = out_dir / (name + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        for tmpl in URL_TEMPLATES:
            url = tmpl.format(record=record_of(name), name=name)
            try:
                print(f"[get] {name} (attempt {attempt}) from {url}", flush=True)
                _stream(url, part)
                got = md5sum(part)
                if got != expected:
                    part.unlink(missing_ok=True)
                    raise OSError(f"md5 mismatch for {name}: got {got}, expected {expected}")
                part.replace(dest)
                print(f"[ok] {name} ({dest.stat().st_size / 1e6:.1f} MB, md5 verified)", flush=True)
                return dest
            except (urllib.error.URLError, OSError, TimeoutError) as err:  # retry on network trouble
                last_err = err
                print(f"  ! {type(err).__name__}: {err}", flush=True)
        time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"could not download {name}: {last_err}")


def raw_deleted(sample: str) -> bool:
    """True if ``sample`` was preprocessed and its raw events file was deleted after the verified run."""
    from gaehep.config import load_json, processed_dir

    meta = processed_dir() / sample / "meta.json"
    return meta.exists() and bool(load_json(meta).get("raw_deleted", False))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=raw_dir())
    p.add_argument("--files", nargs="*", default=None, help="subset of record files to fetch")
    p.add_argument(
        "--sample",
        nargs="+",
        choices=list(SAMPLE_FILES),
        default=None,
        help="fetch the files of these samples (rnd, bb1, bb2, bb3); default: the R&D record only",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="fetch an events file even if its raw copy was deleted after processing",
    )
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for s_ in args.sample or []:
        if s_ != "rnd" and not args.force and raw_deleted(s_):
            print(f"[skip] {s_}: already processed and the raw file deleted (use --force to fetch it again)")
            continue
        names += [n for n in SAMPLE_FILES[s_] if n not in names]
    if args.files is not None:
        names += [n for n in args.files if n not in names]
    if args.files is None and args.sample is None:
        names = list(FILES)
    unknown = [f for f in names if f not in ALL_FILES]
    if unknown:
        p.error(f"unknown file(s): {unknown}")
    # small files first so that problems surface early
    big = {"events_anomalydetection_v2.h5", *(n for n in BLACKBOX_FILES if n.endswith(".h5"))}
    for name in sorted(names, key=lambda n: (n in big, n)):
        download_file(name, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
