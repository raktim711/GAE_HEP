"""Download the LHC Olympics 2020 R&D dataset from Zenodo with resume support and md5 verification.

Record 6466204 (R&D dataset v5) contains the v2 raw events file that matches the v2 high-level features file.
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
    expected = FILES[name]
    dest = out_dir / name
    if dest.exists() and md5sum(dest) == expected:
        print(f"[ok] {name} already present (md5 verified)")
        return dest
    part = out_dir / (name + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        for tmpl in URL_TEMPLATES:
            url = tmpl.format(record=ZENODO_RECORD, name=name)
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=raw_dir())
    p.add_argument("--files", nargs="*", default=list(FILES), help="subset of record files to fetch")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    unknown = [f for f in args.files if f not in FILES]
    if unknown:
        p.error(f"unknown file(s): {unknown}")
    # small files first so that problems surface early
    for name in sorted(args.files, key=lambda n: ("events_anomalydetection_v2.h5" == n, n)):
        download_file(name, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
