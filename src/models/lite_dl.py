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

"""Lightweight 'deep-learning-style' baselines — no torch required.

* :class:`NLinearForecaster` and :class:`DLinearForecaster` are
  PyTorch-free re-implementations of the simple-but-strong baselines from
  Zeng et al. (2023, *Are Transformers Effective for Time Series
  Forecasting?*, AAAI). They use a fixed look-back window and predict a
  single step ahead with linear regression on either the raw lags
  (NLinear) or on the trend / seasonal components after STL
  (DLinear).

* :class:`ThetaForecaster` — Assimakopoulos & Nikolopoulos (2000), the
  M3-winning method. We use the standard θ=0 / θ=2 decomposition with
  SES on the latter and a linear trend on the former.

* :class:`NaiveDriftForecaster` — y(t+h) = y(T) + h · (y(T) - y(1)) / (T-1).
  The "drift" baseline of the M-competitions, surprisingly competitive
  on data with mild trend.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import ForecastModel


def _prediction_bounds(
    history: np.ndarray,
    *,
    upper_quantile: float = 0.995,
) -> tuple[float, float]:
    """Robust prediction bounds inferred from the training history.

    The previous bound used the raw maximum, which is unsafe for datasets
    such as OPSD household where a single bad spike (~200) appears in an
    otherwise sub-2 kW residential series. Quantile/MAD bounds prevent one
    outlier from making recursive linear forecasts explode, while still
    allowing negative targets such as the ETT benchmark.
    """
    h = np.asarray(history, dtype=float)
    h = h[np.isfinite(h)]
    if h.size == 0:
        return -1.0, 1.0

    upper_q = float(np.clip(upper_quantile, 0.90, 0.999))
    lower_q = max(0.001, 1.0 - upper_q)
    q_lo, q25, q50, q75, q_hi = np.quantile(h, [lower_q, 0.25, 0.50, 0.75, upper_q])
    iqr = max(float(q75 - q25), 1e-8)
    mad = max(float(np.median(np.abs(h - q50))), 1e-8)

    lo = min(float(q_lo - 2.0 * iqr), float(q50 - 8.0 * mad))
    hi = max(float(q_hi + 2.0 * iqr), float(q50 + 8.0 * mad))

    # Preserve non-negativity for physical load series, but do not force it
    # on ETT-like signals whose target legitimately goes below zero.
    if float(np.quantile(h, 0.01)) >= 0:
        lo = max(0.0, lo)
    if hi <= lo:
        hi = lo + max(1.0, float(np.std(h)) * 2.0)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Naive Drift
# ---------------------------------------------------------------------------
@dataclass
class NaiveDriftForecaster(ForecastModel):
    name: str = "NaiveDrift"
    family: str = "classical"

    last_value_: float = field(default=float("nan"), init=False)
    slope_: float = field(default=float("nan"), init=False)

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "NaiveDriftForecaster":
        y = y_train.astype(float).values
        self.last_value_ = float(y[-1])
        self.slope_ = float((y[-1] - y[0]) / max(1, len(y) - 1))
        self.fitted_ = True
        return self

    def predict(self, n_periods: int, X_test: pd.DataFrame | None = None) -> np.ndarray:
        h = np.arange(1, n_periods + 1, dtype=float)
        return self.last_value_ + self.slope_ * h


# ---------------------------------------------------------------------------
# Theta
# ---------------------------------------------------------------------------
@dataclass
class ThetaForecaster(ForecastModel):
    """Standard theta method (θ=0, θ=2 lines, then averaged)."""

    name: str = "Theta"
    family: str = "classical"
    seasonal_periods: int = 24
    deseasonalize: bool = True

    results_: object | None = field(default=None, init=False)

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "ThetaForecaster":
        from statsmodels.tsa.forecasting.theta import ThetaModel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.results_ = ThetaModel(
                y_train.astype(float),
                period=self.seasonal_periods,
                deseasonalize=self.deseasonalize,
            ).fit()
        self.fitted_ = True
        return self

    def predict(self, n_periods: int, X_test: pd.DataFrame | None = None) -> np.ndarray:
        if self.results_ is None:
            raise RuntimeError("Theta not fitted.")
        return np.asarray(self.results_.forecast(steps=n_periods)).ravel()


# ---------------------------------------------------------------------------
# NLinear / DLinear (lookback-based, no torch)
# ---------------------------------------------------------------------------
@dataclass
class NLinearForecaster(ForecastModel):
    """Single-step linear model on the last ``lookback`` observations.

    ŷ(t+1) = w · [y(t-L+1), …, y(t)] + b — after subtracting the
    last value (the "Normalize" trick of Zeng et al. 2023).
    Multi-step prediction is recursive.
    """

    name: str = "NLinear"
    family: str = "lite-dl"
    lookback: int = 168
    ridge: float = 1e-2
    clip_quantile: float = 0.995
    coef_: np.ndarray = field(default_factory=lambda: np.empty(0), init=False)
    intercept_: float = field(default=0.0, init=False)
    history_: np.ndarray = field(default_factory=lambda: np.empty(0), init=False)

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "NLinearForecaster":
        y = y_train.astype(float).values
        L = self.lookback
        if len(y) <= L + 10:
            raise ValueError(f"Need at least {L + 10} obs for NLinear(lookback={L}).")
        # Build sliding windows.
        windows = np.lib.stride_tricks.sliding_window_view(y, L + 1)
        X = windows[:, :L]
        Y = windows[:, L]
        # Normalisation: subtract the last value of the window.
        last_vals = X[:, -1:].copy()
        Xn = X - last_vals
        Yn = Y - last_vals.ravel()
        # Closed-form linear regression with tunable ridge regularisation.
        XtX = Xn.T @ Xn + self.ridge * np.eye(L)
        XtY = Xn.T @ Yn
        try:
            self.coef_ = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(XtX, XtY, rcond=None)[0]
        self.intercept_ = 0.0
        self.history_ = y.copy()
        self.fitted_ = True
        return self

    def predict(self, n_periods: int, X_test: pd.DataFrame | None = None) -> np.ndarray:
        L = self.lookback
        history = list(self.history_)
        preds = []
        lo, hi = _prediction_bounds(self.history_, upper_quantile=self.clip_quantile)
        for _ in range(n_periods):
            window = np.array(history[-L:])
            last = window[-1]
            xn = window - last
            yhat = float(self.coef_ @ xn + self.intercept_) + last
            yhat = float(np.clip(yhat, lo, hi))
            preds.append(yhat)
            history.append(yhat)
        return np.array(preds)


@dataclass
class DLinearForecaster(ForecastModel):
    """Decomposition-Linear with reversible instance normalisation.

    Faithful, torch-free re-implementation of Zeng et al. (2023) DLinear
    with **RevIN** (Kim et al., 2022) — the per-window instance-normalisation
    trick that makes the model robust on non-stationary series such as
    residential load. Without RevIN, plain DLinear diverges catastrophically
    on right-skewed loads (RMSE ≫ scale of the target) because the
    trend/seasonal coefficients are estimated on the raw amplitude and
    are then applied out-of-sample where the level has shifted.

    The design changes vs. the previous implementation are:

    1. **Per-window RevIN.** Each look-back window is centred by its mean
       and scaled by its std before the trend/seasonal decomposition; the
       inverse transform is applied to the model output.
    2. **Larger ridge on the seasonal branch** — the seasonal component is
       zero-mean but heavy-tailed on load data.
    3. **Physical clipping** to the empirical support of the training
       series (mirrors what NLinear already does).

    RevIN restores DLinear's status as a strong-but-simple DL baseline
    and removes the divergence noted in the original paper draft.
    """

    name: str = "DLinear"
    family: str = "lite-dl"
    lookback: int = 168
    kernel_size: int = 25          # moving-average window for trend extraction
    ridge_trend: float = 1e-2
    ridge_seasonal: float = 1.0    # heavier ridge on the seasonal branch
    use_revin: bool = True
    revin_eps: float = 1e-5
    clip_quantile: float = 0.995

    coef_t_: np.ndarray = field(default_factory=lambda: np.empty(0), init=False)
    coef_s_: np.ndarray = field(default_factory=lambda: np.empty(0), init=False)
    history_: np.ndarray = field(default_factory=lambda: np.empty(0), init=False)

    @staticmethod
    def _moving_avg(x: np.ndarray, k: int) -> np.ndarray:
        """Centered moving average using cumulative sums."""
        if k <= 1:
            return x.copy()
        pad = k // 2
        x_pad = np.pad(x, pad, mode="edge")
        kernel = np.ones(k) / k
        return np.convolve(x_pad, kernel, mode="valid")[: len(x)]

    def _revin_window(self, w: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Return the RevIN-normalised window and its (mean, std)."""
        if not self.use_revin:
            return w, 0.0, 1.0
        m = float(w.mean())
        s = float(w.std())
        s = s if s > self.revin_eps else self.revin_eps
        return (w - m) / s, m, s

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "DLinearForecaster":
        y = y_train.astype(float).values
        L = self.lookback
        if len(y) <= L + 10:
            raise ValueError(f"Need at least {L + 10} obs for DLinear(lookback={L}).")

        windows_y = np.lib.stride_tricks.sliding_window_view(y, L + 1)
        Xy = windows_y[:, :L].astype(float)
        Yy = windows_y[:, L].astype(float)
        if self.use_revin:
            means = Xy.mean(axis=1, keepdims=True)
            stds = Xy.std(axis=1, keepdims=True)
            stds = np.where(stds > self.revin_eps, stds, self.revin_eps)
            Xy_n = (Xy - means) / stds
            Yy_n = (Yy - means.ravel()) / stds.ravel()
        else:
            Xy_n, Yy_n = Xy, Yy

        trend = np.apply_along_axis(
            self._moving_avg, 1, Xy_n, self.kernel_size
        )
        seas = Xy_n - trend

        # Faithful DLinear (Zeng et al., 2023): two linear heads whose outputs
        # are ADDED. The parameters must therefore be estimated JOINTLY —
        # fitting each branch to the target independently would double-count
        # the level and blow up out-of-sample (the bug the paper flagged
        # under the original implementation). We concatenate [trend, seas]
        # into a single 2L feature matrix and solve one ridge regression
        # with separate penalties per branch.
        Z = np.concatenate([trend, seas], axis=1)                       # (N, 2L)
        penalty = np.diag(
            np.concatenate([np.full(L, self.ridge_trend),
                            np.full(L, self.ridge_seasonal)])
        )
        try:
            w = np.linalg.solve(Z.T @ Z + penalty, Z.T @ Yy_n)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(Z.T @ Z + penalty, Z.T @ Yy_n, rcond=None)[0]
        self.coef_t_ = w[:L]
        self.coef_s_ = w[L:]
        self.history_ = y.copy()
        self.fitted_ = True
        return self

    def predict(self, n_periods: int, X_test: pd.DataFrame | None = None) -> np.ndarray:
        L = self.lookback
        hist = list(self.history_)
        preds = []
        lo, hi = _prediction_bounds(self.history_, upper_quantile=self.clip_quantile)
        for _ in range(n_periods):
            window = np.asarray(hist[-L:], dtype=float)
            xw_n, m, s = self._revin_window(window)
            trend = self._moving_avg(xw_n, self.kernel_size)
            seas = xw_n - trend
            yhat_n = float(self.coef_t_ @ trend + self.coef_s_ @ seas)
            yhat = yhat_n * s + m
            if not np.isfinite(yhat):
                yhat = float(hist[-1])
            yhat = float(np.clip(yhat, lo, hi))
            preds.append(yhat)
            hist.append(yhat)
        return np.array(preds)
