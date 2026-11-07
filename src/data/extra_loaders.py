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

"""Loaders for the four external benchmark CSVs (datasets 2–5 of the project).

The assignment dataset is UCI ``household_power_consumption.txt`` — loaded
by :func:`src.data.load_uci_household_power`, not this module.

Project scope (five datasets total)::

    UCI                      → data/household_power_consumption.txt
    ETTh1, ETTh2             → data/external/*.csv  (auto-download)
    OPSD_DE_household        → data/external/*.csv  (auto-download)
    OPSD_DE_total            → data/external/*.csv  (auto-download)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import pandas as pd

from .. import config

logger = logging.getLogger(__name__)

EXTERNAL_DIR: Path = config.DATA_DIR / "external"
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

# Only the four external benchmarks in project scope.
_KNOWN_DATASETS: Mapping[str, dict] = {
    "ETTh1": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "freq": "h", "target": "OT", "datetime_col": "date",
        "season": 24, "source": "Informer benchmark (China, hourly)",
    },
    "ETTh2": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        "freq": "h", "target": "OT", "datetime_col": "date",
        "season": 24, "source": "Informer benchmark (China, hourly)",
    },
    "OPSD_DE_household": {
        "url": "https://data.open-power-system-data.org/household_data/2020-04-15/household_data_60min_singleindex.csv",
        "freq": "h", "target": "DE_household_consumption", "datetime_col": "utc_timestamp",
        "season": 24, "loader": "opsd",
        "source": "OPSD German residential (6 households, hourly)",
    },
    "OPSD_DE_total": {
        "url": "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv",
        "freq": "h", "target": "DE_load_actual_entsoe_transparency",
        "datetime_col": "utc_timestamp",
        "season": 24, "loader": "opsd_total",
        "source": "OPSD German national load (hourly)",
    },
}


def download_instructions() -> str:
    """Return wget-style commands for every external benchmark."""
    lines = [
        "# Download external benchmarks (project datasets 2–5):",
        f"mkdir -p {EXTERNAL_DIR}",
        "",
    ]
    for name, meta in _KNOWN_DATASETS.items():
        lines.append(f"# {name} — {meta.get('source', '')}")
        lines.append(f"wget -O {EXTERNAL_DIR / (name + '.csv')} {meta['url']}")
        lines.append("")
    return "\n".join(lines)


def list_external_datasets() -> list[str]:
    """Names of external benchmarks that already exist on disk."""
    return [
        name for name in config.EXTERNAL_DATASETS
        if (EXTERNAL_DIR / f"{name}.csv").exists()
    ]


def load_external(name: str) -> tuple[pd.DataFrame, str, dict]:
    """Load one external benchmark; returns ``(df, target, meta)``."""
    if name not in _KNOWN_DATASETS:
        raise KeyError(
            f"Unknown dataset {name!r}. "
            f"Project scope: {list(config.EXTERNAL_DATASETS)}"
        )
    meta = _KNOWN_DATASETS[name]
    path = EXTERNAL_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run:\n"
            f"  python -m src.data.extra_loaders --download {name}\n"
            f"or: python -m src.data.extra_loaders --download {name}"
        )

    loader_kind = meta.get("loader", "default")
    if loader_kind == "opsd":
        return _load_opsd_household(path, meta)
    if loader_kind == "opsd_total":
        return _load_opsd_total(path, meta)

    df = pd.read_csv(path)
    dt_col = meta["datetime_col"]
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.set_index(dt_col).sort_index()
    return df, meta["target"], meta


# Largest hourly import one household can draw through a standard German
# 63 A three-phase connection: 3 x 230 V x 63 A = 43.5 kW, i.e. 43.5 kWh in
# one hour. Larger hourly steps in a cumulative meter are meter errors.
OPSD_HOUSEHOLD_MAX_KWH_PER_HOUR: float = 43.5


def opsd_household_series(raw: pd.DataFrame, max_kwh: float = OPSD_HOUSEHOLD_MAX_KWH_PER_HOUR
                          ) -> tuple[pd.Series, pd.DataFrame]:
    """Mean hourly grid import across households from cumulative meters.

    Each household's cumulative reading is differenced on its own first, so a
    household that starts or stops reporting does not create a jump in the
    average. Negative steps (meter resets) and steps above ``max_kwh`` are set
    to missing, and the households reporting in each hour are averaged.

    Returns the series and a table of the readings that were set to missing.
    """
    inc = raw.diff()
    neg = inc < 0
    big = inc > max_kwh
    removed = []
    for col in raw.columns:
        for t in inc.index[neg[col].to_numpy() | big[col].to_numpy()]:
            removed.append({"timestamp": t, "household": col,
                            "hourly_step_kwh": float(inc.at[t, col]),
                            "reason": "negative step" if inc.at[t, col] < 0 else f"above {max_kwh} kWh"})
    inc = inc.mask(neg | big)
    return inc.mean(axis=1, skipna=True), pd.DataFrame(removed)


def _load_opsd_household(path: Path, meta: dict) -> tuple[pd.DataFrame, str, dict]:
    """OPSD household file → mean residential hourly grid import (kWh/h).

    Since v1.2 each household is differenced before averaging and meter
    errors are removed (see :func:`opsd_household_series`). Earlier releases
    averaged the cumulative readings first and differenced the average, which
    turned a change in the set of reporting households into a spike.
    """
    df = pd.read_csv(path, low_memory=False)
    dt_col = meta["datetime_col"]
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True).dt.tz_convert(None)
    df = df.set_index(dt_col).sort_index()

    grid_cols = [c for c in df.columns
                 if c.startswith("DE_KN_residential") and c.endswith("_grid_import")]
    if not grid_cols:
        grid_cols = [c for c in df.columns
                     if c.startswith("DE_KN_residential") and c.endswith("_consumption")]
    if not grid_cols:
        grid_cols = [c for c in df.columns
                     if "residential" in c.lower() and "grid_import" in c.lower()]
    if not grid_cols:
        raise ValueError(
            f"No DE residential columns in {path}. "
            f"Columns: {', '.join(df.columns[:8])}…"
        )

    series, removed = opsd_household_series(df[grid_cols])
    if len(removed):
        big = removed[removed.reason != "negative step"]
        logger.info("OPSD household: %d hourly readings set to missing (%d negative steps, %d above %.1f kWh)",
                    len(removed), len(removed) - len(big), len(big), OPSD_HOUSEHOLD_MAX_KWH_PER_HOUR)
        for _, r in big.iterrows():
            logger.info("OPSD household: removed %s at %s: %.1f kWh in one hour",
                        r.household, r.timestamp, r.hourly_step_kwh)
    out = pd.DataFrame({meta["target"]: series})
    return out.dropna(), meta["target"], {**meta, "n_readings_removed": int(len(removed)),
                                          "removed_readings": removed}


def _load_opsd_total(path: Path, meta: dict) -> tuple[pd.DataFrame, str, dict]:
    """OPSD national time-series → German actual load column."""
    df = pd.read_csv(path, low_memory=False)
    dt_col = meta["datetime_col"]
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True).dt.tz_convert(None)
    df = df.set_index(dt_col).sort_index()
    target = meta["target"]
    if target not in df.columns:
        cands = [c for c in df.columns if c.startswith("DE_load_actual")]
        if not cands:
            raise ValueError("No DE_load_actual_* column in OPSD time-series file.")
        target = cands[0]
    return df[[target]].dropna(), target, {**meta, "target": target}


def maybe_download(name: str) -> Path | None:
    """Download one external CSV if missing (needs network)."""
    if name not in _KNOWN_DATASETS:
        raise KeyError(f"{name!r} not in project scope: {list(_KNOWN_DATASETS)}")
    target_path = EXTERNAL_DIR / f"{name}.csv"
    if target_path.exists():
        return target_path
    try:
        import urllib.request
        url = _KNOWN_DATASETS[name]["url"]
        logger.info("Downloading %s → %s", url, target_path)
        urllib.request.urlretrieve(url, target_path)
        return target_path
    except Exception as exc:  # pragma: no cover
        logger.warning("Download failed for %s: %s", name, exc)
        return None


def download_all_external() -> list[str]:
    """Download every external benchmark; return names that succeeded."""
    ok = []
    for name in config.EXTERNAL_DATASETS:
        if maybe_download(name):
            ok.append(name)
    return ok


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download", nargs="*", const="all",
                   help="Download external CSVs (default: all four).")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list:
        print("Project external datasets:", list(config.EXTERNAL_DATASETS))
        print("On disk:", list_external_datasets())
    if args.download is not None:
        targets = list(config.EXTERNAL_DATASETS) if (
            not args.download or "all" in args.download
        ) else args.download
        for n in targets:
            print(f"Downloading {n} …")
            print("  ->", maybe_download(n))
        print("On disk:", list_external_datasets())
    elif not args.list:
        print(download_instructions())
