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

"""Forecasting models — classical, ML/GBM, lite-DL, probabilistic, and novel FAHE."""

from .base import ForecastModel, SupervisedForecaster
from .classical import (
    NaiveForecaster, SeasonalNaiveForecaster,
    ARIMAForecaster, SARIMAForecaster, HoltWintersForecaster,
)
from .lite_dl import (
    NaiveDriftForecaster, ThetaForecaster,
    NLinearForecaster, DLinearForecaster,
)
from .ml import build_ml_zoo
from .novel import FAHEForecaster, CSEForecaster
from .quantile import QuantileLightGBMForecaster, QuantileGBRForecaster
from .optional_dl import (
    neuralforecast_available, chronos_available,
    NeuralForecastWrapper, ChronosZeroShot, build_optional_dl_zoo,
)
from .registry import build_registry, ModelSpec
from .recursive import recursive_predict, RecursiveDiagnostics

__all__ = [
    "ForecastModel", "SupervisedForecaster",
    "NaiveForecaster", "SeasonalNaiveForecaster",
    "ARIMAForecaster", "SARIMAForecaster", "HoltWintersForecaster",
    "NaiveDriftForecaster", "ThetaForecaster",
    "NLinearForecaster", "DLinearForecaster",
    "build_ml_zoo",
    "FAHEForecaster", "CSEForecaster",
    "QuantileLightGBMForecaster", "QuantileGBRForecaster",
    "neuralforecast_available", "chronos_available",
    "NeuralForecastWrapper", "ChronosZeroShot", "build_optional_dl_zoo",
    "build_registry", "ModelSpec",
    "recursive_predict", "RecursiveDiagnostics",
]
