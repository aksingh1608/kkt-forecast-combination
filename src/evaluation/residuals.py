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

"""Residual diagnostics: Ljung-Box autocorrelation test and helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LjungBoxResult:
    statistic: float
    p_value: float
    lags: int
    significant: bool   # True ⇒ residuals show structure (bad)


def ljung_box(
    residuals: np.ndarray | pd.Series,
    *,
    lags: int = 24,
    alpha: float = 0.05,
) -> LjungBoxResult:
    """Ljung-Box Q-statistic for residual whiteness.

    H₀: residuals are i.i.d. (no autocorrelation up to ``lags``).
    A *significant* p-value means the model leaves structure on the
    table — the residuals are *not* white noise.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    r = np.asarray(residuals, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if len(r) < lags + 5:
        # statsmodels needs more samples than lags; bump lags down.
        lags = max(2, len(r) // 4)
    res = acorr_ljungbox(r, lags=[lags], return_df=True)
    stat = float(res["lb_stat"].iloc[0])
    p = float(res["lb_pvalue"].iloc[0])
    return LjungBoxResult(
        statistic=stat, p_value=p, lags=lags,
        significant=p < alpha,
    )


def ljung_box_for_models(
    actual: np.ndarray | pd.Series,
    predictions: dict[str, np.ndarray],
    *,
    lags: int = 24,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run Ljung-Box on every model's residuals; return a tidy frame."""
    actual_a = np.asarray(actual, dtype=float).ravel()
    rows = []
    for name, preds in predictions.items():
        preds_a = np.asarray(preds, dtype=float).ravel()
        if len(preds_a) != len(actual_a):
            logger.warning("Skipping %s: length mismatch.", name)
            continue
        try:
            res = ljung_box(actual_a - preds_a, lags=lags, alpha=alpha)
        except Exception as exc:
            logger.warning("Ljung-Box failed for %s: %s", name, exc)
            continue
        rows.append({
            "Model": name,
            "Q_stat": res.statistic,
            "p_value": res.p_value,
            "lags": res.lags,
            "residuals_white": (not res.significant),
        })
    return pd.DataFrame(rows).sort_values("p_value", ascending=False) \
        if rows else pd.DataFrame(
            columns=["Model", "Q_stat", "p_value", "lags", "residuals_white"]
        )


__all__ = ["ljung_box", "ljung_box_for_models", "LjungBoxResult"]
