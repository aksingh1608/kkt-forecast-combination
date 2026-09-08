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

"""Dataset loaders.

Two entry points:

* :func:`load_uci_household_power` — strict loader for the bundled UCI
  ``household_power_consumption.txt`` file with parquet caching.
* :func:`load_generic_csv` — flexible loader for arbitrary time-series CSVs
  uploaded through the Streamlit frontend.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from .. import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UCI household power consumption
# ---------------------------------------------------------------------------
def load_uci_household_power(
    path: str | Path | None = None,
    *,
    use_cache: bool = True,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load the UCI ``household_power_consumption.txt`` dataset.

    The raw file is semicolon-separated, uses ``?`` as the missing-value token
    and has separate ``Date``/``Time`` columns. After parsing we return a
    DataFrame with a single tz-naive :class:`~pandas.DatetimeIndex`.

    Parameters
    ----------
    path
        Optional override of the raw file location.
    use_cache
        Read/write a Parquet cache for ~10x faster subsequent loads.
    nrows
        Optional row cap (useful for unit tests / quick experiments).
    """
    raw_path = Path(path) if path is not None else config.RAW_DATASET_PATH
    cache_path = config.CACHE_DIR / "household_power_raw.parquet"

    if use_cache and cache_path.exists() and nrows is None:
        logger.info("Loading cached parquet: %s", cache_path)
        return pd.read_parquet(cache_path)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw UCI dataset not found at {raw_path}. "
            "Expected household_power_consumption.txt in the data/ folder."
        )

    logger.info("Parsing raw UCI dataset: %s", raw_path)
    df = pd.read_csv(
        raw_path,
        sep=";",
        na_values=[config.MISSING_TOKEN],
        low_memory=False,
        nrows=nrows,
    )

    # Combine Date + Time into a single DatetimeIndex.
    df[config.DATETIME_COLUMN] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.drop(columns=["Date", "Time"])

    # Coerce numeric columns; the file mixes strings and floats.
    for col in config.RAW_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=[config.DATETIME_COLUMN])
        .set_index(config.DATETIME_COLUMN)
        .sort_index()
    )

    if use_cache and nrows is None:
        try:
            df.to_parquet(cache_path)
            logger.info("Wrote parquet cache: %s", cache_path)
        except Exception as exc:  # pragma: no cover - cache best-effort
            logger.warning("Could not write parquet cache: %s", exc)

    return df


# ---------------------------------------------------------------------------
# Generic uploader-friendly loader
# ---------------------------------------------------------------------------
def _guess_datetime_column(df: pd.DataFrame) -> str | None:
    """Pick the most likely datetime column from a generic CSV."""
    candidates: Iterable[str] = (
        "datetime",
        "timestamp",
        "date_time",
        "date",
        "time",
        "ds",
    )
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    # Fallback: any column that parses as datetime for the first 100 rows.
    for col in df.columns:
        sample = df[col].dropna().head(100)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() > 0.9:
            return col
    return None


def load_generic_csv(
    source,
    *,
    datetime_column: str | None = None,
    target_column: str | None = None,
    sep: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load an arbitrary time-series CSV.

    Designed for the Streamlit upload path. Heuristically locates the
    datetime column and returns a sorted DataFrame indexed by it, plus the
    name of the chosen target column.

    Parameters
    ----------
    source
        Anything :func:`pandas.read_csv` accepts (path, file-like, …).
    datetime_column, target_column
        Optional explicit overrides.
    sep
        Optional CSV separator; ``None`` auto-detects (`,` / `;` / `\\t`).
    """
    if sep is None:
        # ``engine='python'`` allows separator sniffing via sep=None.
        df = pd.read_csv(source, sep=None, engine="python")
    else:
        df = pd.read_csv(source, sep=sep)

    dt_col = datetime_column or _guess_datetime_column(df)
    if dt_col is None:
        raise ValueError(
            "Could not infer a datetime column. "
            "Please specify `datetime_column=` explicitly."
        )

    # If we got separate Date+Time columns (UCI-style), merge them.
    if dt_col.lower() == "date" and "Time" in df.columns:
        df[dt_col] = pd.to_datetime(
            df[dt_col].astype(str) + " " + df["Time"].astype(str),
            errors="coerce",
        )
        df = df.drop(columns=["Time"])
    else:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

    df = df.dropna(subset=[dt_col]).set_index(dt_col).sort_index()
    df.index.name = config.DATETIME_COLUMN

    # Coerce numeric where possible.
    for col in df.columns:
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().mean() > 0.5:
                df[col] = coerced

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns available to forecast.")

    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(
                f"target_column={target_column!r} not in columns {list(df.columns)}"
            )
        target = target_column
    elif config.TARGET_COLUMN in df.columns:
        target = config.TARGET_COLUMN
    else:
        # Pick the numeric column with highest variance (most informative).
        target = df[numeric_cols].var().idxmax()
        logger.info("Auto-selected target column: %s", target)

    return df, target
