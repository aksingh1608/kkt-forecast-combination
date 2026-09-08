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

"""Registry for declaring which models the experiment runner should use."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import RANDOM_STATE
from .base import ForecastModel
from .classical import (
    ARIMAForecaster,
    HoltWintersForecaster,
    NaiveForecaster,
    SARIMAForecaster,
    SeasonalNaiveForecaster,
)
from .lite_dl import (
    DLinearForecaster,
    NaiveDriftForecaster,
    NLinearForecaster,
    ThetaForecaster,
)
from .ml import build_ml_zoo
from .novel import FAHEForecaster
from .quantile import QuantileGBRForecaster, QuantileLightGBMForecaster
from .optional_dl import build_optional_dl_zoo


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing how to instantiate a model."""

    name: str
    family: str
    factory: callable
    requires_features: bool


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _freeze_ml_factory(forecaster) -> callable:
    """Return a fresh-cloned factory for a built ML forecaster."""
    def factory(template=forecaster):
        from sklearn.base import clone
        from .base import SupervisedForecaster

        return SupervisedForecaster(
            name=template.name,
            family=template.family,
            estimator=clone(template.estimator),
        )

    return factory


def build_registry(
    *,
    seasonal_period: int = 24,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    include_heavy_classical: bool = True,
    include_lite_dl: bool = True,
    include_quantile: bool = True,
    include_optional_dl: bool = False,
    optional_dl_horizon: int = 24,
    optional_dl_max_steps: int = 200,
    optional_dl_input_size: int | None = None,
) -> list[ModelSpec]:
    """Return the full list of model specs the runner should evaluate."""
    specs: list[ModelSpec] = []

    # --- Classical persistence baselines ---
    specs.append(ModelSpec("Naive", "classical",
                           lambda: NaiveForecaster(), False))
    specs.append(ModelSpec("NaiveDrift", "classical",
                           lambda: NaiveDriftForecaster(), False))
    specs.append(ModelSpec("SeasonalNaive", "classical",
                           lambda p=seasonal_period: SeasonalNaiveForecaster(season_length=p),
                           False))

    if include_heavy_classical:
        specs.append(ModelSpec("ARIMA", "classical",
                               lambda: ARIMAForecaster(order=(2, 1, 2)), False))
        specs.append(ModelSpec("SARIMA", "classical",
                               lambda p=seasonal_period: SARIMAForecaster(
                                   order=(1, 1, 1), seasonal_order=(1, 1, 1, p)),
                               False))
        specs.append(ModelSpec("Holt-Winters", "classical",
                               lambda p=seasonal_period: HoltWintersForecaster(
                                   seasonal_periods=p),
                               False))
        specs.append(ModelSpec("Theta", "classical",
                               lambda p=seasonal_period: ThetaForecaster(
                                   seasonal_periods=p),
                               False))

    # --- Lightweight 'DL-style' linear baselines (no torch) ---
    if include_lite_dl:
        specs.append(ModelSpec("NLinear", "lite-dl",
                               lambda: NLinearForecaster(lookback=168), False))
        specs.append(ModelSpec("DLinear", "lite-dl",
                               lambda: DLinearForecaster(lookback=168), False))

    # --- ML / GBM zoo ---
    for forecaster in build_ml_zoo(random_state=random_state, n_jobs=n_jobs):
        specs.append(ModelSpec(
            name=forecaster.name,
            family=forecaster.family,
            factory=_freeze_ml_factory(forecaster),
            requires_features=True,
        ))

    # --- Probabilistic (point-forecast metric still works on the median) ---
    if include_quantile:
        specs.append(ModelSpec(
            name="QuantileLightGBM",
            family="probabilistic",
            factory=lambda: QuantileLightGBMForecaster(),
            requires_features=True,
        ))

    # --- Novel architecture ---
    specs.append(ModelSpec(
        name="FAHE (Ours)",
        family="novel-hybrid",
        factory=lambda p=seasonal_period: FAHEForecaster(seasonal_periods=p),
        requires_features=True,
    ))

    # --- Optional deep learning / foundation models -------------------------
    if include_optional_dl:
        for forecaster in build_optional_dl_zoo(
            seasonal_periods=seasonal_period,
            horizon=optional_dl_horizon,
            max_steps=optional_dl_max_steps,
            input_size=optional_dl_input_size,
        ):
            specs.append(ModelSpec(
                name=forecaster.name,
                family=forecaster.family,
                factory=_make_dl_factory(forecaster),
                requires_features=False,
            ))

    return specs


def _make_dl_factory(template) -> callable:
    """Re-instantiate a deep-learning forecaster on each call."""
    cls = type(template)
    init_kwargs = {
        f.name: getattr(template, f.name)
        for f in template.__dataclass_fields__.values()
        if f.init and f.name != "fitted_"
    }

    def factory():
        return cls(**init_kwargs)

    return factory


__all__ = ["ModelSpec", "build_registry"]
