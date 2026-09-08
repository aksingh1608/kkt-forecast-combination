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

"""Energy-forecasting metrics from Meyer et al. (2024).

These two metrics are *household-aware* analogues of MAE and a
peak-aware p-norm error. They were introduced for the residential
power-forecasting benchmark in:

    Meyer, B. et al. (2024) 'Benchmarking household-level
    short-term load forecasting with foundation models'.

* **MAEh** (Mean Absolute Error per Household, then averaged):
  MAEh(Y, Ŷ) = (1 / H) Σ_h MAE(Y_h, Ŷ_h)
  where ``Y`` is a {household × time} matrix.

* **APNEh** (Adjusted P-Norm Error per Household):
  APNEh_p(Y, Ŷ) = (1 / H) Σ_h (mean(|y_t − ŷ_t|^p · w(y_t))^(1/p))
  where ``w(y_t) = 1 + λ · max(0, y_t − τ)`` over-weights peaks
  beyond a per-household threshold ``τ`` (default = 95-th percentile
  of ``y``) by a configurable peak-importance factor ``λ`` (default 2).

Both metrics support a single-series fallback so they're directly
usable on the standard univariate UCI benchmark.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _as_matrix(arr: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Return a (n_households, n_timesteps) ndarray.

    A 1-D vector or single-column frame is treated as a single household.
    A 2-D array of shape (T, H) is transposed if the long dim is on axis 0.
    """
    if isinstance(arr, pd.DataFrame):
        arr = arr.values
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        # Heuristic: more rows than columns ⇒ time on axis 0; transpose.
        return arr.T if arr.shape[0] > arr.shape[1] else arr
    raise ValueError(f"Expected 1-D or 2-D array, got shape {arr.shape}")


def maeh(y_true: np.ndarray | pd.DataFrame,
         y_pred: np.ndarray | pd.DataFrame) -> float:
    """Mean Absolute Error averaged across households (Meyer et al. 2024)."""
    y_true_m = _as_matrix(y_true)
    y_pred_m = _as_matrix(y_pred)
    if y_true_m.shape != y_pred_m.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_m.shape} vs y_pred {y_pred_m.shape}"
        )
    per_household_mae = np.mean(np.abs(y_true_m - y_pred_m), axis=1)
    return float(np.mean(per_household_mae))


@dataclass
class APNEhResult:
    overall: float
    per_household: np.ndarray
    p: float
    peak_lambda: float
    peak_quantile: float


def apneh(
    y_true: np.ndarray | pd.DataFrame,
    y_pred: np.ndarray | pd.DataFrame,
    *,
    p: float = 2.0,
    peak_lambda: float = 2.0,
    peak_quantile: float = 0.95,
) -> APNEhResult:
    """Adjusted P-Norm Error per Household (peak-aware) — Meyer et al. (2024).

    Larger ``peak_lambda`` (≥ 1) places more weight on time-steps that
    exceed each household's ``peak_quantile`` ground-truth value. A
    classical p-norm error is recovered for ``peak_lambda = 0``.

    Returns
    -------
    APNEhResult
        ``overall`` is the across-household mean; ``per_household`` is
        the per-series APNE so the user can inspect tail risk.
    """
    if p <= 0:
        raise ValueError("p must be > 0")
    y_true_m = _as_matrix(y_true)
    y_pred_m = _as_matrix(y_pred)
    if y_true_m.shape != y_pred_m.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_m.shape} vs y_pred {y_pred_m.shape}"
        )

    abs_err = np.abs(y_true_m - y_pred_m)
    thresholds = np.quantile(y_true_m, peak_quantile, axis=1, keepdims=True)
    weights = 1.0 + peak_lambda * np.clip(y_true_m - thresholds, 0.0, None) \
        / (np.abs(thresholds) + 1e-12)

    # Weighted p-norm per household.
    weighted_err_p = (abs_err ** p) * weights
    per_h = np.mean(weighted_err_p, axis=1) ** (1.0 / p)
    return APNEhResult(
        overall=float(np.mean(per_h)),
        per_household=per_h,
        p=float(p),
        peak_lambda=float(peak_lambda),
        peak_quantile=float(peak_quantile),
    )


def both_metrics(
    y_true: np.ndarray | pd.DataFrame,
    y_pred: np.ndarray | pd.DataFrame,
    *,
    p: float = 2.0,
    peak_lambda: float = 2.0,
    peak_quantile: float = 0.95,
) -> dict[str, float]:
    """Convenience: return ``{'MAEh': …, 'APNEh': …}`` in one call."""
    apne = apneh(
        y_true, y_pred,
        p=p, peak_lambda=peak_lambda, peak_quantile=peak_quantile,
    )
    return {
        "MAEh": maeh(y_true, y_pred),
        "APNEh": apne.overall,
        "APNEh_p": apne.p,
        "APNEh_lambda": apne.peak_lambda,
        "APNEh_quantile": apne.peak_quantile,
    }


__all__ = ["maeh", "apneh", "both_metrics", "APNEhResult"]
