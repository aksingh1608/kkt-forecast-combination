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

"""Bootstrap confidence intervals for forecasting metrics.

We use **moving-block bootstrap** (Künsch, 1989) which preserves
short-range autocorrelation in the residual series — the standard choice
for time-series uncertainty quantification.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .metrics import compute_all_metrics


def moving_block_bootstrap(
    n: int, *, block_size: int, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate `n_bootstrap` block-bootstrap index arrays of length n."""
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=(n_bootstrap, n_blocks))
    blocks = starts[:, :, None] + np.arange(block_size)[None, None, :]
    flat = blocks.reshape(n_bootstrap, -1)[:, :n]
    return flat


def bootstrap_metric_ci(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] | str = "RMSE",
    block_size: int = 24,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
    y_train: np.ndarray | pd.Series | None = None,
    seasonality: int = 24,
) -> dict[str, float]:
    """Return point estimate and bootstrap (1-α) CI for a metric.

    Parameters
    ----------
    metric
        Either a metric name (one of MAE/RMSE/MAPE/sMAPE/R2/MASE) or
        a custom callable ``f(y_true, y_pred) -> float``.
    block_size
        Length of each resampled block. Choose ≈ √n for short-range
        autocorrelation, larger for stronger persistence.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    n = len(yt)
    rng = np.random.default_rng(seed)
    indices = moving_block_bootstrap(
        n, block_size=block_size, n_bootstrap=n_bootstrap, rng=rng
    )

    if isinstance(metric, str):
        metric_name = metric

        def _eval(idx):
            return compute_all_metrics(
                yt[idx], yp[idx], y_train=y_train, seasonality=seasonality
            ).get(metric_name, float("nan"))
    else:
        metric_name = getattr(metric, "__name__", "metric")

        def _eval(idx):
            return float(metric(yt[idx], yp[idx]))

    samples = np.array([_eval(idx) for idx in indices])
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "metric": metric_name, "n_bootstrap": n_bootstrap}

    point = float(np.mean(samples))
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {
        "point": point, "lo": float(lo), "hi": float(hi),
        "metric": metric_name, "n_bootstrap": n_bootstrap,
    }
