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

"""Quantile-regression forecasters.

* :class:`QuantileLightGBMForecaster` — fits one LightGBM per quantile
  using the native ``objective='quantile'`` loss. The fastest, most
  accurate way to get a full quantile grid from a tree-based model
  (Meinshausen, 2006; March-2017 LightGBM release notes).
* :class:`QuantileGBRForecaster`     — sklearn fallback that uses
  ``GradientBoostingRegressor`` with ``loss='quantile'``. Slower but
  always available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .base import ForecastModel

logger = logging.getLogger(__name__)


@dataclass
class QuantileLightGBMForecaster(ForecastModel):
    """One LightGBM per quantile."""

    name: str = "QuantileLightGBM"
    family: str = "probabilistic"
    requires_features: bool = True
    quantiles: tuple[float, ...] = (0.05, 0.1, 0.5, 0.9, 0.95)
    learning_rate: float = 0.05
    n_estimators: int = 500
    num_leaves: int = 63
    random_state: int = 42

    models_: dict[float, object] = field(default_factory=dict, init=False)
    last_X_test_: pd.DataFrame | None = field(default=None, init=False)

    def fit(  # type: ignore[override]
        self, y_train: pd.Series, X_train: pd.DataFrame | None = None
    ) -> "QuantileLightGBMForecaster":
        if X_train is None:
            raise ValueError(f"{self.name} requires X_train.")
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError("lightgbm is required for QuantileLightGBM.") from exc

        for q in self.quantiles:
            model = LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(X_train.values, y_train.values)
            self.models_[q] = model
        self.fitted_ = True
        return self

    def predict(  # type: ignore[override]
        self, n_periods: int, X_test: pd.DataFrame | None = None
    ) -> np.ndarray:
        """Return median (q=0.5) prediction. Use :meth:`predict_quantiles`
        for the full distribution."""
        preds = self.predict_quantiles(n_periods, X_test=X_test)
        return preds.get(0.5, next(iter(preds.values())))

    def predict_quantiles(
        self, n_periods: int, X_test: pd.DataFrame | None = None
    ) -> dict[float, np.ndarray]:
        if not self.fitted_:
            raise RuntimeError(f"{self.name} not fitted.")
        if X_test is None or len(X_test) != n_periods:
            raise ValueError("X_test must have n_periods rows.")
        self.last_X_test_ = X_test
        out: dict[float, np.ndarray] = {}
        Xv = X_test.values
        for q, model in self.models_.items():
            out[q] = np.asarray(model.predict(Xv)).ravel()
        # Enforce non-crossing via per-row sorting (rare but possible).
        qs_sorted = sorted(out.keys())
        stacked = np.column_stack([out[q] for q in qs_sorted])
        stacked = np.sort(stacked, axis=1)
        return {q: stacked[:, i] for i, q in enumerate(qs_sorted)}


@dataclass
class QuantileGBRForecaster(ForecastModel):
    """sklearn ``GradientBoostingRegressor(loss='quantile')`` fallback."""

    name: str = "QuantileGBR"
    family: str = "probabilistic"
    requires_features: bool = True
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    random_state: int = 42

    models_: dict[float, object] = field(default_factory=dict, init=False)

    def fit(  # type: ignore[override]
        self, y_train: pd.Series, X_train: pd.DataFrame | None = None
    ) -> "QuantileGBRForecaster":
        from sklearn.ensemble import GradientBoostingRegressor

        if X_train is None:
            raise ValueError(f"{self.name} requires X_train.")
        for q in self.quantiles:
            m = GradientBoostingRegressor(
                loss="quantile", alpha=q,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )
            m.fit(X_train.values, y_train.values)
            self.models_[q] = m
        self.fitted_ = True
        return self

    def predict(  # type: ignore[override]
        self, n_periods: int, X_test: pd.DataFrame | None = None
    ) -> np.ndarray:
        return self.predict_quantiles(n_periods, X_test=X_test)[0.5]

    def predict_quantiles(
        self, n_periods: int, X_test: pd.DataFrame | None = None
    ) -> dict[float, np.ndarray]:
        if X_test is None or len(X_test) != n_periods:
            raise ValueError("X_test must have n_periods rows.")
        return {q: np.asarray(m.predict(X_test.values)).ravel()
                for q, m in self.models_.items()}
