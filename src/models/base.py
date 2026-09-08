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

"""Common forecaster interfaces.

Every model in this project — classical, ML or the novel FAHE — exposes the
same minimal :class:`ForecastModel` API so the experiment runner and the
Streamlit frontend can treat them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


@dataclass
class ForecastModel(ABC):
    """Abstract base class for every forecaster."""

    name: str
    family: str = "uncategorised"
    requires_features: bool = False
    fitted_: bool = field(default=False, init=False)

    @abstractmethod
    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "ForecastModel":
        ...

    @abstractmethod
    def predict(self, n_periods: int, X_test: pd.DataFrame | None = None) -> np.ndarray:
        ...

    def fit_predict(
        self,
        y_train: pd.Series,
        n_periods: int,
        X_train: pd.DataFrame | None = None,
        X_test: pd.DataFrame | None = None,
    ) -> np.ndarray:
        self.fit(y_train, X_train=X_train)
        return self.predict(n_periods, X_test=X_test)


@dataclass
class SupervisedForecaster(ForecastModel):
    """Wraps any sklearn-compatible regressor as a forecaster.

    The regressor consumes the engineered feature matrix produced by
    :func:`src.features.build_supervised_frame`. All target alignment is
    handled by the orchestrator, not the model itself.
    """

    estimator: BaseEstimator | None = None
    requires_features: bool = True

    def fit(  # type: ignore[override]
        self, y_train: pd.Series, X_train: pd.DataFrame | None = None
    ) -> "SupervisedForecaster":
        if X_train is None:
            raise ValueError(f"{self.name}: SupervisedForecaster requires X_train.")
        if self.estimator is None:
            raise ValueError(f"{self.name}: estimator is not set.")
        self.estimator.fit(X_train.values, y_train.values)
        self.fitted_ = True
        return self

    def predict(  # type: ignore[override]
        self, n_periods: int, X_test: pd.DataFrame | None = None
    ) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError(f"{self.name}: predict before fit.")
        if X_test is None:
            raise ValueError(f"{self.name}: SupervisedForecaster requires X_test.")
        if len(X_test) != n_periods:
            raise ValueError(
                f"{self.name}: X_test rows ({len(X_test)}) != n_periods ({n_periods})."
            )
        return np.asarray(self.estimator.predict(X_test.values)).ravel()

    def get_underlying_estimator(self) -> Any:
        return self.estimator
