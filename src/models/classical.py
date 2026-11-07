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

"""Classical statistical forecasters.

* :class:`NaiveForecaster`          — y(t+1) = y(t)
* :class:`SeasonalNaiveForecaster`  — y(t+1) = y(t+1-m)
* :class:`ARIMAForecaster`          — ARIMA(p, d, q)
* :class:`SARIMAForecaster`         — SARIMAX(p,d,q)(P,D,Q,m)
* :class:`HoltWintersForecaster`    — Triple exponential smoothing

These are the workhorses of the classical-statistical literature and serve
as the lower-bound baselines that any modern method must beat to be
publishable.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import ForecastModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persistence baselines
# ---------------------------------------------------------------------------
@dataclass
class NaiveForecaster(ForecastModel):
    name: str = "Naive"
    family: str = "classical"
    last_value_: float = field(default=float("nan"), init=False)

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "NaiveForecaster":
        self.last_value_ = float(y_train.iloc[-1])
        self.fitted_ = True
        return self

    def predict(self, n_periods: int, X_test: pd.DataFrame | None = None) -> np.ndarray:
        return np.full(n_periods, self.last_value_, dtype=float)


@dataclass
class SeasonalNaiveForecaster(ForecastModel):
    name: str = "SeasonalNaive"
    family: str = "classical"
    season_length: int = 24
    history_: np.ndarray = field(default_factory=lambda: np.empty(0), init=False)

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "SeasonalNaiveForecaster":
        if len(y_train) < self.season_length:
            raise ValueError(
                f"Need at least {self.season_length} observations for "
                f"SeasonalNaive (got {len(y_train)})."
            )
        self.history_ = y_train.values[-self.season_length :].astype(float)
        self.fitted_ = True
        return self

    def predict(self, n_periods: int, X_test: pd.DataFrame | None = None) -> np.ndarray:
        reps = int(np.ceil(n_periods / self.season_length))
        return np.tile(self.history_, reps)[:n_periods]


# ---------------------------------------------------------------------------
# ARIMA family (statsmodels)
# ---------------------------------------------------------------------------
@dataclass
class ARIMAForecaster(ForecastModel):
    name: str = "ARIMA"
    family: str = "classical"
    order: tuple[int, int, int] = (2, 1, 2)
    results_: object | None = field(default=None, init=False)

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "ARIMAForecaster":
        from statsmodels.tsa.arima.model import ARIMA

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(y_train.astype(float), order=self.order)
            self.results_ = model.fit(method_kwargs={"warn_convergence": False},
                                      low_memory=True)
        self.fitted_ = True
        return self

    def predict(self, n_periods: int, X_test: pd.DataFrame | None = None) -> np.ndarray:
        if self.results_ is None:
            raise RuntimeError("ARIMA not fitted.")
        forecast = self.results_.forecast(steps=n_periods)
        return np.asarray(forecast).ravel()


@dataclass
class SARIMAForecaster(ForecastModel):
    name: str = "SARIMA"
    family: str = "classical"
    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24)
    results_: object | None = field(default=None, init=False)

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "SARIMAForecaster":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y_train.astype(float),
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            # low_memory=True: identical parameters and forecasts, a small
            # fraction of the memory (the full Kalman output for a seasonal
            # model on tens of thousands of hours runs to several GB).
            self.results_ = model.fit(disp=False, maxiter=100, low_memory=True)
        self.fitted_ = True
        return self

    def predict(self, n_periods: int, X_test: pd.DataFrame | None = None) -> np.ndarray:
        if self.results_ is None:
            raise RuntimeError("SARIMA not fitted.")
        return np.asarray(self.results_.forecast(steps=n_periods)).ravel()


# ---------------------------------------------------------------------------
# Holt-Winters (ETS)
# ---------------------------------------------------------------------------
@dataclass
class HoltWintersForecaster(ForecastModel):
    name: str = "Holt-Winters"
    family: str = "classical"
    seasonal_periods: int = 24
    trend: str | None = "add"
    seasonal: str | None = "add"
    results_: object | None = field(default=None, init=False)

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "HoltWintersForecaster":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        y = y_train.astype(float)
        if self.seasonal == "mul" and (y <= 0).any():
            logger.info("Falling back to additive seasonality (non-positive values present).")
            seasonal = "add"
        else:
            seasonal = self.seasonal

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=seasonal,
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            )
            self.results_ = model.fit(optimized=True, use_brute=False)
        self.fitted_ = True
        return self

    def predict(self, n_periods: int, X_test: pd.DataFrame | None = None) -> np.ndarray:
        if self.results_ is None:
            raise RuntimeError("Holt-Winters not fitted.")
        return np.asarray(self.results_.forecast(steps=n_periods)).ravel()
