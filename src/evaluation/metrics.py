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

"""Regression metrics for time-series forecasting.

We deliberately use vectorised, dependency-light implementations so the same
helpers work in scripts, notebooks and the Streamlit frontend.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


ArrayLike = np.ndarray | pd.Series


def _to_arrays(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt, yp = _to_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt, yp = _to_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-6) -> float:
    """Mean Absolute Percentage Error (%).

    Guards against division-by-zero by clamping the denominator at ``eps``.
    """
    yt, yp = _to_arrays(y_true, y_pred)
    denom = np.where(np.abs(yt) < eps, eps, np.abs(yt))
    return float(np.mean(np.abs((yt - yp) / denom)) * 100.0)


def smape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-6) -> float:
    """Symmetric MAPE (%). Always in [0, 200]."""
    yt, yp = _to_arrays(y_true, y_pred)
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    denom = np.where(denom < eps, eps, denom)
    return float(np.mean(np.abs(yt - yp) / denom) * 100.0)


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt, yp = _to_arrays(y_true, y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def mase(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_train: ArrayLike,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    Hyndman & Koehler (2006). The scale is the in-sample MAE of a
    seasonal-naive forecast on the training data — making MASE comparable
    across series of different magnitudes and the canonical metric for
    forecasting competitions (M4/M5).
    """
    yt, yp = _to_arrays(y_true, y_pred)
    yt_train = np.asarray(y_train, dtype=float).ravel()
    yt_train = yt_train[np.isfinite(yt_train)]
    if len(yt_train) <= seasonality:
        return float("nan")
    naive_errors = np.abs(yt_train[seasonality:] - yt_train[:-seasonality])
    scale = naive_errors.mean()
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)) / scale)


def compute_all_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    y_train: ArrayLike | None = None,
    seasonality: int = 1,
) -> Mapping[str, float]:
    """Compute MAE, RMSE, MAPE, sMAPE, R² and (if y_train given) MASE."""
    out: dict[str, float] = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
    if y_train is not None:
        out["MASE"] = mase(y_true, y_pred, y_train, seasonality=seasonality)
    return out
