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

"""Feature engineering primitives.

Converts a time-indexed univariate (or multivariate) series into a
supervised-learning matrix suitable for any sklearn-compatible regressor.

Design notes
------------
* All transformations are **causal**: the value at row ``t`` depends only on
  observations strictly before ``t``. This is essential when a feature is
  used as input to a forecaster — leaking future information is the most
  common pitfall in time-series ML.
* Fourier terms encode known seasonalities (daily/weekly/yearly) at low
  parameter cost and are far more sample-efficient than huge one-hot
  encodings of the hour-of-day.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..config import FeatureConfig


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add categorical and cyclical calendar features.

    Cyclical encodings (sin/cos) avoid the artificial "discontinuity" at the
    23→0 hour boundary that plain integer hours would introduce.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Calendar features require a DatetimeIndex.")
    idx = df.index
    out = df.copy()
    out["hour"] = idx.hour
    out["day_of_week"] = idx.dayofweek
    out["day_of_month"] = idx.day
    out["day_of_year"] = idx.dayofyear
    out["week_of_year"] = idx.isocalendar().week.astype(int)
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    out["year"] = idx.year
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # Cyclical (sin/cos) encodings of the dominant cycles.
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    return out


# ---------------------------------------------------------------------------
# Lags & rolling windows
# ---------------------------------------------------------------------------
def add_lag_features(
    df: pd.DataFrame, target: str, lags: Iterable[int]
) -> pd.DataFrame:
    """Add lagged copies of ``target``: y(t-l) for each l in ``lags``."""
    out = df.copy()
    for lag in lags:
        out[f"{target}_lag_{lag}"] = out[target].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    target: str,
    windows: Iterable[int],
    *,
    shift: int = 1,
) -> pd.DataFrame:
    """Rolling mean/std/min/max over ``windows`` past observations.

    The ``shift=1`` ensures the window does NOT contain the value at time
    ``t`` itself — preserving causality.
    """
    out = df.copy()
    base = out[target].shift(shift)
    for w in windows:
        roll = base.rolling(window=w, min_periods=max(1, w // 2))
        out[f"{target}_roll_mean_{w}"] = roll.mean()
        out[f"{target}_roll_std_{w}"] = roll.std()
        out[f"{target}_roll_min_{w}"] = roll.min()
        out[f"{target}_roll_max_{w}"] = roll.max()
    return out


# ---------------------------------------------------------------------------
# Fourier features
# ---------------------------------------------------------------------------
def add_fourier_features(
    df: pd.DataFrame,
    fourier_orders: Iterable[tuple[int, int]],
) -> pd.DataFrame:
    """Append Fourier terms of period P and order K.

    Each ``(P, K)`` tuple adds ``2K`` columns:
      sin(2πk·t/P), cos(2πk·t/P)   for k = 1..K
    where ``t`` is the row index (in samples).
    """
    out = df.copy()
    t = np.arange(len(out))
    for period, order in fourier_orders:
        for k in range(1, order + 1):
            out[f"fourier_sin_{period}_{k}"] = np.sin(2 * np.pi * k * t / period)
            out[f"fourier_cos_{period}_{k}"] = np.cos(2 * np.pi * k * t / period)
    return out


# ---------------------------------------------------------------------------
# Composite supervised frame builder
# ---------------------------------------------------------------------------
def build_supervised_frame(
    df: pd.DataFrame,
    target: str,
    cfg: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build a supervised (X, y) frame from a time-indexed DataFrame.

    Returns
    -------
    X
        Feature DataFrame indexed by timestamp.
    y
        Target series aligned with ``X.index``.
    feature_names
        Convenience copy of ``X.columns.tolist()``.
    """
    cfg = cfg or FeatureConfig()
    if target not in df.columns:
        raise KeyError(
            f"target {target!r} missing from frame; have {list(df.columns)}"
        )

    work = df.copy()
    if cfg.add_calendar:
        work = add_calendar_features(work)
    work = add_lag_features(work, target, cfg.lags)
    work = add_rolling_features(work, target, cfg.rolling_windows)
    if cfg.fourier_orders:
        work = add_fourier_features(work, cfg.fourier_orders)

    if cfg.drop_na:
        work = work.dropna()

    y = work[target]
    # Drop the target *and* any other raw measurements at time t — those would
    # be unknown when actually forecasting future values.
    drop_cols = [c for c in df.columns if c in work.columns]
    X = work.drop(columns=drop_cols)
    return X, y, X.columns.tolist()
