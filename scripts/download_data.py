# Copyright 2026 Aryan Kumar Singh and co-authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Download the five paper datasets into ``data/raw/`` and verify SHA-256.

Also places copies where existing loaders expect them:

* ``data/household_power_consumption.txt`` (UCI)
* ``data/external/{ETTh1,ETTh2,OPSD_DE_household,OPSD_DE_total}.csv``

The Apache license covers this repository's code only. Each dataset remains
under its upstream terms — see ``DATA.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "data" / "raw"
EXTERNAL_DIR = ROOT / "data" / "external"
UCI_PATH = ROOT / "data" / "household_power_consumption.txt"

# SHA-256 of the exact files used to produce results/ (recorded 2026-09).
CHECKSUMS: dict[str, dict[str, str]] = {
    "UCI": {
        "url": "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
        "sha256": "4259c9d7ece5dbee9ab8d53682baac68d791c864f0f64a52b4043cb3b90894b7",
        "raw_name": "household_power_consumption.txt",
        "kind": "uci_zip",
    },
    "ETTh1": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "sha256": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
        "raw_name": "ETTh1.csv",
        "kind": "file",
        "dest_external": "ETTh1.csv",
    },
    "ETTh2": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        "sha256": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
        "raw_name": "ETTh2.csv",
        "kind": "file",
        "dest_external": "ETTh2.csv",
    },
    "OPSD_DE_household": {
        "url": "https://data.open-power-system-data.org/household_data/2020-04-15/household_data_60min_singleindex.csv",
        "sha256": "15956440b27465686eae2abf5895cbc3dab80f934a891bcb2f4f944428e97d60",
        "raw_name": "OPSD_DE_household.csv",
        "kind": "file",
        "dest_external": "OPSD_DE_household.csv",
    },
    "OPSD_DE_total": {
        "url": "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv",
        "sha256": "6a7f2bc571314cbf9c321cc03437691cd4be95c3a6f075e60ff99e8035c704c8",
        "raw_name": "OPSD_DE_total.csv",
        "kind": "file",
        "dest_external": "OPSD_DE_total.csv",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "cse-benchmark/1.0"})
    with urlopen(req, timeout=600) as resp:
        return resp.read()


def _place(src: Path, *dests: Path) -> None:
    for dest in dests:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.resolve() != src.resolve():
            shutil.copy2(src, dest)


def _legacy_path(name: str, meta: dict) -> Path | None:
    if name == "UCI" and UCI_PATH.exists():
        return UCI_PATH
    if meta.get("dest_external"):
        p = EXTERNAL_DIR / meta["dest_external"]
        if p.exists():
            return p
    return None


def ensure_dataset(name: str, *, force: bool = False) -> Path:
    meta = CHECKSUMS[name]
    raw_path = RAW_DIR / meta["raw_name"]
    expected = meta["sha256"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if raw_path.exists() and not force:
        digest = sha256_file(raw_path)
        if digest != expected:
            raise SystemExit(
                f"{name}: checksum mismatch for existing {raw_path}\n"
                f"  expected {expected}\n  got      {digest}"
            )
        print(f"[ok] {name}: {raw_path} (checksum match)")
    else:
        legacy = _legacy_path(name, meta)
        if legacy is not None and not force:
            digest = sha256_file(legacy)
            if digest == expected:
                _place(legacy, raw_path)
                print(f"[ok] {name}: seeded {raw_path} from {legacy}")
            else:
                legacy = None
        if not raw_path.exists() or force:
            print(f"[download] {name} from {meta['url']}")
            payload = _download(meta["url"])
            if meta["kind"] == "uci_zip":
                with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                    member = "household_power_consumption.txt"
                    with zf.open(member) as src, raw_path.open("wb") as out:
                        shutil.copyfileobj(src, out)
            else:
                raw_path.write_bytes(payload)
            digest = sha256_file(raw_path)
            if digest != expected:
                raise SystemExit(
                    f"{name}: downloaded checksum mismatch\n"
                    f"  expected {expected}\n  got      {digest}\n"
                    "If the upstream file changed, update CHECKSUMS in this script "
                    "and DATA.md after verifying the paper artefacts still apply."
                )
            print(f"[ok] {name}: wrote {raw_path}")

    # Mirror into loader paths.
    if name == "UCI":
        _place(raw_path, UCI_PATH)
    else:
        _place(raw_path, EXTERNAL_DIR / meta["dest_external"])
    return raw_path


def verify_all() -> int:
    missing = 0
    for name, meta in CHECKSUMS.items():
        path = RAW_DIR / meta["raw_name"]
        alt = UCI_PATH if name == "UCI" else EXTERNAL_DIR / meta.get("dest_external", "")
        candidate = path if path.exists() else alt
        if not candidate.exists():
            print(f"[missing] {name}: {path}")
            missing += 1
            continue
        digest = sha256_file(candidate)
        status = "ok" if digest == meta["sha256"] else "MISMATCH"
        print(f"[{status}] {name}: {candidate}")
        if status != "ok":
            missing += 1
    return missing


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="*", default=list(CHECKSUMS),
                   help="Subset of datasets to download (default: all five).")
    p.add_argument("--force", action="store_true", help="Re-download even if present.")
    p.add_argument("--verify-only", action="store_true",
                   help="Only verify checksums of files already on disk.")
    args = p.parse_args()
    if args.verify_only:
        raise SystemExit(verify_all())
    for name in args.datasets:
        if name not in CHECKSUMS:
            raise SystemExit(f"Unknown dataset {name!r}. Choose from {list(CHECKSUMS)}")
        ensure_dataset(name, force=args.force)
    print("Done. See DATA.md for licences and cleaning notes.")


if __name__ == "__main__":
    main()
