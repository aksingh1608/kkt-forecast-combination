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

"""Per-horizon error breakdown.

Given a multi-step (recursive) forecast on the test window, compute
metrics for each forecast step independently. This is the key figure
that quantifies **error compounding** with horizon and answers RQ4 of
the paper: how much of the tree-vs-classical gap is capacity vs.
protocol.

Two interpretations of "horizon-h" error are supported:

1. **Prefix mode** (``mode="prefix"``, default). Metrics on the first
   ``h`` forecast steps of a single rolled-out forecast — the semantics
   of the original implementation, kept for backward compatibility.
2. **Rolling-origin mode** (``mode="rolling"``). Metrics on the
   *cross-section* of horizon-``h`` errors gathered from many forecast
   origins — the semantics used in the load forecasting literature and the one
   that gives meaningful DM tests. Requires a list of (origin, prediction)
   tuples.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import compute_all_metrics


def per_horizon_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    horizons: Sequence[int] = (1, 6, 24, 168),
    y_train: np.ndarray | pd.Series | None = None,
    seasonality: int = 24,
) -> pd.DataFrame:
    """For each ``h`` in ``horizons`` evaluate metrics on y[:h].

    This treats the test set as a single rolled-out forecast: the
    "horizon-h" error is the mean error over the first h forecast steps.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    n = min(len(yt), len(yp))
    yt, yp = yt[:n], yp[:n]
    rows = []
    for h in horizons:
        h_eff = min(h, n)
        m = compute_all_metrics(
            yt[:h_eff], yp[:h_eff],
            y_train=y_train, seasonality=seasonality,
        )
        m = {"horizon": h, **m, "h_actual": h_eff}
        rows.append(m)
    return pd.DataFrame(rows)


def per_horizon_for_models(
    y_true: np.ndarray | pd.Series,
    predictions: dict[str, np.ndarray],
    *,
    horizons: Sequence[int] = (1, 6, 24, 168),
    y_train: np.ndarray | pd.Series | None = None,
    seasonality: int = 24,
    metric: str = "RMSE",
) -> pd.DataFrame:
    """Wide-format DataFrame: rows = models, cols = horizons, values = metric."""
    rows = []
    for name, preds in predictions.items():
        per_h = per_horizon_metrics(
            y_true, preds, horizons=horizons,
            y_train=y_train, seasonality=seasonality,
        )
        if metric not in per_h.columns:
            # MASE is only reported when y_train is provided. Emit NaNs
            # for that horizon so the leaderboard shape is consistent.
            values = {h: float("nan") for h in per_h["horizon"].tolist()}
        else:
            values = per_h.set_index("horizon")[metric].to_dict()
        rows.append(pd.Series(values, name=name))
    out = pd.DataFrame(rows)
    out.index.name = "model"
    out.columns = [f"h={h}" for h in out.columns]
    return out


# ---------------------------------------------------------------------------
# Rolling-origin per-horizon evaluation (paper Section VII-C)
# ---------------------------------------------------------------------------
def rolling_origin_horizon_metrics(
    y_true: np.ndarray | pd.Series,
    predictions_per_origin: Sequence[tuple[int, np.ndarray]],
    *,
    horizons: Sequence[int] = (1, 3, 6, 12, 24),
    y_train: np.ndarray | pd.Series | None = None,
    seasonality: int = 24,
) -> pd.DataFrame:
    """Per-horizon metrics gathered across many forecast origins.

    Parameters
    ----------
    y_true
        The full test-set actuals (1-D array).
    predictions_per_origin
        List of ``(origin_index, forecast_array)`` tuples. Each forecast
        array must have length ≥ max(horizons); the value at index
        ``h-1`` is the ``h``-step-ahead forecast issued at that origin.
    horizons
        The horizons at which to report metrics. The paper reports
        h ∈ {1, 3, 6, 12, 24}.
    """
    yt = np.asarray(y_true).ravel()
    n_test = len(yt)
    rows = []
    for h in horizons:
        yt_h: list[float] = []
        yp_h: list[float] = []
        for origin, forecast in predictions_per_origin:
            fc = np.asarray(forecast).ravel()
            target_idx = origin + h - 1
            if target_idx >= n_test or len(fc) < h:
                continue
            yt_h.append(float(yt[target_idx]))
            yp_h.append(float(fc[h - 1]))
        if not yt_h:
            continue
        m = compute_all_metrics(
            np.asarray(yt_h), np.asarray(yp_h),
            y_train=y_train, seasonality=seasonality,
        )
        rows.append({"horizon": h, "n_pairs": len(yt_h), **m})
    return pd.DataFrame(rows)


def per_horizon_leaderboard(
    y_true: np.ndarray | pd.Series,
    predictions_by_model: Mapping[str, np.ndarray],
    *,
    horizons: Sequence[int] = (1, 3, 6, 12, 24),
    y_train: np.ndarray | pd.Series | None = None,
    seasonality: int = 24,
    metric: str = "RMSE",
) -> pd.DataFrame:
    """(model × horizon) leaderboard using **prefix** semantics.

    Given each model's recursive multi-step forecast over the same test
    window, compute the chosen ``metric`` on the first ``h`` steps for
    every ``h`` in ``horizons``. Suitable for the paper's per-horizon
    table alongside the recursive-rollout results.
    """
    return per_horizon_for_models(
        y_true, dict(predictions_by_model),
        horizons=horizons, y_train=y_train,
        seasonality=seasonality, metric=metric,
    )


def choose_rolling_origins(
    n_test: int,
    *,
    max_horizon: int,
    n_origins: int = 48,
) -> list[int]:
    """Evenly spaced forecast origins that leave room for ``max_horizon`` steps."""
    last = n_test - max_horizon
    if last < 0:
        return []
    if n_origins <= 1:
        return [0]
    n_origins = min(n_origins, last + 1)
    if n_origins == 1:
        return [0]
    return sorted({int(round(i * last / (n_origins - 1))) for i in range(n_origins)})


def rolling_origin_leaderboard(
    y_true: np.ndarray | pd.Series,
    forecasts_by_model: Mapping[str, Sequence[tuple[int, np.ndarray]]],
    *,
    horizons: Sequence[int] = (1, 3, 6, 12, 24),
    y_train: np.ndarray | pd.Series | None = None,
    seasonality: int = 24,
    metric: str = "RMSE",
) -> pd.DataFrame:
    """Wide (model × horizon) table using rolling-origin semantics."""
    rows = []
    yt = np.asarray(y_true).ravel()
    for name, per_origin in forecasts_by_model.items():
        per_h = rolling_origin_horizon_metrics(
            yt, per_origin, horizons=horizons,
            y_train=y_train, seasonality=seasonality,
        )
        if per_h.empty or metric not in per_h.columns:
            values = {h: float("nan") for h in horizons}
        else:
            values = per_h.set_index("horizon")[metric].to_dict()
            for h in horizons:
                values.setdefault(h, float("nan"))
        rows.append(pd.Series(values, name=name))
    out = pd.DataFrame(rows)
    out.index.name = "model"
    out.columns = [f"h={h}" for h in out.columns]
    return out


__all__ = [
    "per_horizon_metrics",
    "per_horizon_for_models",
    "rolling_origin_horizon_metrics",
    "per_horizon_leaderboard",
    "choose_rolling_origins",
    "rolling_origin_leaderboard",
]
