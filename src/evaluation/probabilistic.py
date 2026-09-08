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

"""Probabilistic-forecast evaluation metrics.

Modern Q1 forecasting papers report not only point-forecast accuracy but
the *quality of the predictive distribution*. The four canonical metrics:

* **Pinball loss** at each quantile (Koenker & Bassett, 1978) —
  the proper scoring rule for quantile forecasts.
* **CRPS** (Continuous Ranked Probability Score; Gneiting & Raftery,
  2007) — proper scoring rule for the full predictive distribution.
* **Coverage** (PICP) — empirical fraction of actuals inside the
  nominal prediction interval.
* **Sharpness** (MPIW) — mean width of the prediction interval.
  *Among well-calibrated forecasts, narrower is better.*
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def pinball_loss(
    y_true: np.ndarray, y_pred_q: np.ndarray, q: float
) -> float:
    """Pinball loss at quantile ``q ∈ (0, 1)``."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred_q).ravel()
    diff = yt - yp
    loss = np.maximum(q * diff, (q - 1) * diff)
    return float(np.mean(loss))


def average_pinball_loss(
    y_true: np.ndarray,
    y_pred_quantiles: Mapping[float, np.ndarray],
) -> float:
    """Mean pinball loss across the supplied quantile grid."""
    losses = [pinball_loss(y_true, p, q) for q, p in y_pred_quantiles.items()]
    return float(np.mean(losses))


def crps_from_quantiles(
    y_true: np.ndarray,
    y_pred_quantiles: Mapping[float, np.ndarray],
) -> float:
    """Approximate CRPS using the trapezoidal rule on a quantile grid.

    CRPS = ∫₀¹ 2 · pinball_q(y, F⁻¹(q)) dq.
    With a discrete quantile set this becomes a weighted sum of pinball
    losses (the "quantile-discretised CRPS"; see Bracher et al., 2021).
    """
    qs = sorted(y_pred_quantiles.keys())
    if len(qs) < 2:
        return float("nan")
    losses = np.array([pinball_loss(y_true, y_pred_quantiles[q], q) for q in qs])
    # Trapezoidal weights * 2 for the CRPS conversion.
    weights = np.diff(qs)
    crps = np.sum(weights * (losses[:-1] + losses[1:])) * 2.0 / 2.0
    return float(crps)


def coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Empirical PICP — fraction of points inside [lower, upper]."""
    yt = np.asarray(y_true).ravel()
    lo = np.asarray(lower).ravel()
    up = np.asarray(upper).ravel()
    return float(np.mean((yt >= lo) & (yt <= up)))


def sharpness(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean prediction-interval width (MPIW)."""
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """Winkler score (interval score) at miscoverage rate ``alpha``."""
    yt = np.asarray(y_true).ravel()
    lo = np.asarray(lower).ravel()
    up = np.asarray(upper).ravel()
    width = up - lo
    below = (lo - yt) * (yt < lo)
    above = (yt - up) * (yt > up)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


@dataclass
class ProbabilisticReport:
    quantiles: Sequence[float]
    pinball_per_q: Mapping[float, float]
    mean_pinball: float
    crps: float
    coverage_per_alpha: Mapping[float, float]   # nominal → empirical
    width_per_alpha: Mapping[float, float]
    winkler_per_alpha: Mapping[float, float]


def evaluate_probabilistic(
    y_true: np.ndarray,
    y_pred_quantiles: Mapping[float, np.ndarray],
    *,
    intervals: Sequence[tuple[float, float]] = ((0.1, 0.9), (0.05, 0.95)),
) -> ProbabilisticReport:
    """End-to-end probabilistic evaluation."""
    pin_per_q = {q: pinball_loss(y_true, p, q) for q, p in y_pred_quantiles.items()}
    cov_per_alpha: dict[float, float] = {}
    width_per_alpha: dict[float, float] = {}
    winkler_per_alpha: dict[float, float] = {}
    for lo_q, hi_q in intervals:
        if lo_q in y_pred_quantiles and hi_q in y_pred_quantiles:
            lo = y_pred_quantiles[lo_q]
            hi = y_pred_quantiles[hi_q]
            nominal = hi_q - lo_q
            cov_per_alpha[nominal] = coverage(y_true, lo, hi)
            width_per_alpha[nominal] = sharpness(lo, hi)
            winkler_per_alpha[nominal] = winkler_score(y_true, lo, hi, 1 - nominal)
    return ProbabilisticReport(
        quantiles=list(y_pred_quantiles.keys()),
        pinball_per_q=pin_per_q,
        mean_pinball=float(np.mean(list(pin_per_q.values()))),
        crps=crps_from_quantiles(y_true, y_pred_quantiles),
        coverage_per_alpha=cov_per_alpha,
        width_per_alpha=width_per_alpha,
        winkler_per_alpha=winkler_per_alpha,
    )
