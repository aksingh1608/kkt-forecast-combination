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

"""Cleaning, resampling and chronological splitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .. import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSchema:
    """Lightweight description of a parsed time-series dataset."""

    freq: str
    target: str
    n_rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    missing_pct: float
    columns: tuple[str, ...]


def infer_schema(df: pd.DataFrame, target: str) -> DatasetSchema:
    """Build a :class:`DatasetSchema` from a cleaned, indexed DataFrame."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex.")
    freq = pd.infer_freq(df.index[:200]) or "irregular"
    return DatasetSchema(
        freq=freq,
        target=target,
        n_rows=len(df),
        start=df.index.min(),
        end=df.index.max(),
        missing_pct=float(df.isna().mean().mean() * 100),
        columns=tuple(df.columns),
    )


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_dataset(
    df: pd.DataFrame,
    *,
    drop_duplicate_index: bool = True,
    interpolate_method: str = "time",
    short_gap_limit: int = 60,
) -> pd.DataFrame:
    """Robustly clean a raw indexed time-series DataFrame.

    Strategy:

    1. Drop exact duplicate timestamps (keep the first observation).
    2. Reindex onto the inferred regular frequency so that genuine gaps
       become NaNs (rather than being silently skipped).
    3. Short gaps are filled by **time-aware linear interpolation**, which
       respects irregular spacing — this is the dominant pattern in the
       UCI dataset (the missingness comes in short bursts of a few minutes).
    4. Longer gaps (default: > 60 timesteps) are filled forward with the
       seasonal-naive value, falling back to forward-fill.
    """
    if df.empty:
        return df

    df = df.copy()
    if drop_duplicate_index:
        before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        if len(df) != before:
            logger.info("Dropped %d duplicate timestamps", before - len(df))

    # Reindex onto a regular frequency so gaps surface as NaNs.
    inferred = pd.infer_freq(df.index[:1000])
    if inferred is None:
        # Best-effort fallback: assume the median delta.
        median_delta = df.index.to_series().diff().median()
        inferred = pd.tseries.frequencies.to_offset(median_delta)
    if inferred is not None:
        df = df.asfreq(inferred)

    # Time-aware interpolation handles the bulk of short gaps cleanly.
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].interpolate(
        method=interpolate_method,
        limit=short_gap_limit,
        limit_direction="both",
    )

    # For long gaps, use seasonal-naive (24h back) if possible.
    if isinstance(df.index, pd.DatetimeIndex):
        season = _detect_seasonal_period(df.index)
        if season is not None and len(df) > season:
            mask = df[numeric_cols].isna()
            seasonal_fill = df[numeric_cols].shift(season)
            df[numeric_cols] = df[numeric_cols].where(~mask, seasonal_fill)

    # Final safety net: bidirectional fill.
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    return df


def _detect_seasonal_period(index: pd.DatetimeIndex) -> int | None:
    """Heuristic for the dominant seasonal period given an index."""
    freq = pd.infer_freq(index[:200])
    if freq is None:
        return None
    freq_root = "".join(c for c in freq if c.isalpha()).upper()
    return config.FREQ_TO_SEASON.get(freq_root) or config.FREQ_TO_SEASON.get(
        freq_root[:1]
    )


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
# Sub-metering columns are watt-hours consumed *within* the interval — they
# must be SUMMED on aggregation. All other measurements are instantaneous
# readings and should be averaged.
_SUM_COLUMNS: tuple[str, ...] = (
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
)


def resample_dataset(
    df: pd.DataFrame,
    freq: str = config.DEFAULT_FREQ,
    *,
    sum_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Resample a cleaned dataset to a coarser frequency.

    Aggregation rule per column:
      * Listed in ``sum_columns`` → sum (defaults to UCI sub-meterings).
      * Otherwise numeric          → mean.
      * Non-numeric                → first.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Need a DatetimeIndex to resample.")
    sum_set = set(sum_columns) if sum_columns is not None else set(_SUM_COLUMNS)

    agg_map: dict[str, str] = {}
    for col in df.columns:
        if col in sum_set:
            agg_map[col] = "sum"
        elif pd.api.types.is_numeric_dtype(df[col]):
            agg_map[col] = "mean"
        else:
            agg_map[col] = "first"

    resampled = df.resample(freq).agg(agg_map)
    # Resampling can re-introduce NaNs at the edges; tidy them up.
    numeric_cols = resampled.select_dtypes(include="number").columns
    resampled[numeric_cols] = (
        resampled[numeric_cols].interpolate(method="time", limit_direction="both")
    )
    return resampled.dropna(how="all")


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def train_test_split_temporal(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    val_size: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Chronological train/val/test split.

    ``val_size`` carves the validation window from the *end* of the training
    set so the test set always lies in the future relative to both other
    splits — this matches realistic deployment.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    if not 0.0 <= val_size < 1.0:
        raise ValueError("val_size must be in [0, 1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1")

    n = len(df)
    n_test = max(int(round(n * test_size)), 1)
    n_val = int(round(n * val_size))
    test = df.iloc[n - n_test :]
    train_full = df.iloc[: n - n_test]
    if n_val > 0:
        val = train_full.iloc[-n_val:]
        train = train_full.iloc[:-n_val]
        return {"train": train, "val": val, "test": test}
    return {"train": train_full, "test": test}


def safe_log1p(x: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Numerically safe log1p (clips negatives to 0 to handle measurement noise)."""
    if isinstance(x, pd.Series):
        return np.log1p(x.clip(lower=0))
    return np.log1p(np.clip(x, 0, None))
