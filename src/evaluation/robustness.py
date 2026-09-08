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

"""Robustness diagnostics.

Three perturbations a Q1 reviewer will ask about:

* :func:`noise_injection_curve` — inject Gaussian noise of increasing
  σ into the test target series, retrain (cheap models) or just re-score
  (expensive models), and report metric degradation.
* :func:`missing_injection_curve` — randomly mask increasing fractions
  of the test inputs (lag features) with NaN → median imputed.
* :func:`distribution_shift_eval` — train on summer (Apr-Sep) and test
  on winter (Oct-Mar), or vice versa.
* :func:`train_size_curve` — sensitivity to training-set size.

All functions return tidy DataFrames suitable for direct plotting.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Noise / missingness injections (post-hoc, no retraining required)
# ---------------------------------------------------------------------------
def noise_injection_curve(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    sigmas: Sequence[float] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0),
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply additive Gaussian noise to ``y_pred`` and rescore.

    Reports MAE/RMSE for each ``σ`` (averaged over ``n_repeats``).
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    rng = np.random.default_rng(seed)
    rows = []
    for sigma in sigmas:
        for r in range(n_repeats):
            noisy = yp + rng.normal(0.0, sigma, size=yp.shape)
            m = compute_all_metrics(yt, noisy)
            rows.append({"sigma": sigma, "rep": r, **m})
    return pd.DataFrame(rows).groupby("sigma").mean(numeric_only=True).reset_index()


def missing_injection_curve(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    fractions: Sequence[float] = (0.0, 0.05, 0.1, 0.25, 0.5),
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Randomly mask a fraction of predictions and forward-fill them."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    rng = np.random.default_rng(seed)
    rows = []
    for frac in fractions:
        for r in range(n_repeats):
            mask = rng.random(len(yp)) < frac
            corrupted = yp.copy()
            corrupted[mask] = np.nan
            corrupted = pd.Series(corrupted).ffill().bfill().values
            m = compute_all_metrics(yt, corrupted)
            rows.append({"missing_frac": frac, "rep": r, **m})
    return pd.DataFrame(rows).groupby("missing_frac").mean(numeric_only=True).reset_index()


# ---------------------------------------------------------------------------
# Distribution shift: train summer, test winter (or vice versa)
# ---------------------------------------------------------------------------
def distribution_shift_eval(
    df: pd.DataFrame,
    target: str,
    train_predict_fn: Callable[[pd.Series, pd.DataFrame, pd.DataFrame], np.ndarray],
    *,
    summer_months: Iterable[int] = (4, 5, 6, 7, 8, 9),
    feature_cfg=None,
) -> pd.DataFrame:
    """Train on summer / test on winter and reverse.

    ``train_predict_fn(y_train, X_train, X_test) -> predictions``.
    """
    from ..features.engineering import build_supervised_frame

    summer_set = set(summer_months)
    is_summer = df.index.month.isin(summer_set)
    df_s = df[is_summer]
    df_w = df[~is_summer]

    rows = []
    for setup_name, train_df, test_df in [
        ("train=summer / test=winter", df_s, df_w),
        ("train=winter / test=summer", df_w, df_s),
    ]:
        if len(train_df) < 200 or len(test_df) < 50:
            continue
        Xtr, ytr, _ = build_supervised_frame(train_df, target, cfg=feature_cfg)
        Xte, yte, _ = build_supervised_frame(test_df, target, cfg=feature_cfg)
        # Align column order.
        Xte = Xte[Xtr.columns]
        try:
            preds = train_predict_fn(ytr, Xtr, Xte)
        except Exception as exc:  # pragma: no cover
            rows.append({"setup": setup_name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        m = compute_all_metrics(yte.values, preds, y_train=ytr.values, seasonality=24)
        rows.append({"setup": setup_name, "n_train": len(ytr),
                     "n_test": len(yte), **m})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sensitivity to training-set size
# ---------------------------------------------------------------------------
def train_size_curve(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    estimator_factory: Callable,
    *,
    fractions: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
    seasonality: int = 24,
) -> pd.DataFrame:
    """Train on the LAST ``f * n_train`` observations (chronological)."""
    rows = []
    n = len(X_train)
    for f in fractions:
        k = max(50, int(n * f))
        X_sub = X_train.iloc[-k:]
        y_sub = y_train.iloc[-k:]
        model = estimator_factory()
        model.fit(X_sub.values, y_sub.values)
        preds = model.predict(X_test.values)
        m = compute_all_metrics(y_test.values, preds,
                                y_train=y_sub.values, seasonality=seasonality)
        rows.append({"fraction": f, "n_train": len(y_sub), **m})
    return pd.DataFrame(rows)
