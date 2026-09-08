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

"""Conformal prediction intervals.

Two flavours are exposed:

* :func:`split_conformal_intervals` — the canonical split (inductive)
  conformal procedure (Vovk, Gammerman & Shafer, 2005). Distribution-
  free coverage guarantee under exchangeability.
* :func:`adaptive_conformal_intervals` — Adaptive Conformal Inference
  (Gibbs & Candès, 2021). Online correction that maintains coverage
  even under distribution shift.

Both work with any base point-forecaster: you supply the residuals on a
held-out calibration window plus the test predictions, and they return
the lower/upper bounds at any miscoverage rate ``alpha``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def split_conformal_intervals(
    cal_residuals: np.ndarray,
    test_pred: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Distribution-free symmetric prediction intervals.

    Returns
    -------
    lower, upper, q_hat
        The lower/upper bounds and the conformal radius used.
    """
    cal_residuals = np.abs(np.asarray(cal_residuals).ravel())
    cal_residuals = cal_residuals[np.isfinite(cal_residuals)]
    n = len(cal_residuals)
    if n == 0:
        raise ValueError("Empty calibration set.")
    # Finite-sample correction: ⌈(n+1)(1-α)⌉ / n quantile.
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q_hat = float(np.quantile(cal_residuals, q_level, method="higher"))
    test_pred = np.asarray(test_pred).ravel()
    return test_pred - q_hat, test_pred + q_hat, q_hat


@dataclass
class ACIState:
    alpha_t: float
    history_alpha: list[float]
    history_q: list[float]
    history_covered: list[int]


def adaptive_conformal_intervals(
    cal_residuals: np.ndarray,
    test_pred: np.ndarray,
    test_actual: np.ndarray | None = None,
    *,
    alpha: float = 0.1,
    gamma: float = 0.005,
) -> tuple[np.ndarray, np.ndarray, ACIState]:
    """ACI (Gibbs & Candès, 2021) — online coverage correction.

    The "step size" ``gamma`` controls how aggressively the effective
    miscoverage level is updated after each observation. The default
    (0.005) is the value used in the original paper.

    If ``test_actual`` is None we fall back to a static split-conformal
    interval (no online updates possible without ground truth).
    """
    cal_residuals = np.abs(np.asarray(cal_residuals).ravel())
    cal_residuals = cal_residuals[np.isfinite(cal_residuals)]
    test_pred = np.asarray(test_pred).ravel()
    n_test = len(test_pred)

    state = ACIState(alpha_t=alpha, history_alpha=[alpha],
                     history_q=[], history_covered=[])

    if test_actual is None:
        lo, up, _q = split_conformal_intervals(cal_residuals, test_pred, alpha=alpha)
        return lo, up, state

    test_actual = np.asarray(test_actual).ravel()
    lower = np.empty(n_test, dtype=float)
    upper = np.empty(n_test, dtype=float)
    cal_pool = list(cal_residuals)

    for t in range(n_test):
        a_t = float(np.clip(state.alpha_t, 1e-4, 0.5))
        q_level = min(1.0, np.ceil((len(cal_pool) + 1) * (1 - a_t)) / len(cal_pool))
        q_hat = float(np.quantile(cal_pool, q_level, method="higher"))
        lower[t] = test_pred[t] - q_hat
        upper[t] = test_pred[t] + q_hat
        covered = int(lower[t] <= test_actual[t] <= upper[t])
        state.history_q.append(q_hat)
        state.history_covered.append(covered)
        # Online update of alpha_t (Gibbs & Candès, 2021, Eq. 7).
        state.alpha_t = a_t + gamma * (alpha - (1 - covered))
        state.history_alpha.append(state.alpha_t)
        # Append the new residual so future intervals improve.
        cal_pool.append(abs(test_actual[t] - test_pred[t]))

    return lower, upper, state
