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

"""Project-wide configuration.

A single source of truth for paths, default hyperparameters and constants.
Keeping these in one place is critical for reproducibility — every script,
notebook and the Streamlit frontend share the same defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = ROOT_DIR / "data"
RESULTS_DIR: Final[Path] = ROOT_DIR / "results"
FIGURES_DIR: Final[Path] = RESULTS_DIR / "figures"
METRICS_DIR: Final[Path] = RESULTS_DIR / "metrics"
MODELS_DIR: Final[Path] = RESULTS_DIR / "models"
CACHE_DIR: Final[Path] = DATA_DIR / "cache"

# Only ensure cache exists at import; do not auto-create results output trees.
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATASET_PATH: Final[Path] = DATA_DIR / "household_power_consumption.txt"

# ---------------------------------------------------------------------------
# Project scope — exactly five benchmark datasets (no more, no less)
# ---------------------------------------------------------------------------
# 1. UCI  — bundled ``household_power_consumption.txt`` (assignment dataset)
# 2–5.    — CSV files under ``data/external/`` (see extra_loaders)
PROJECT_DATASETS: Final[tuple[str, ...]] = (
    "UCI",
    "ETTh1",
    "ETTh2",
    "OPSD_DE_household",
    "OPSD_DE_total",
)
EXTERNAL_DATASETS: Final[tuple[str, ...]] = PROJECT_DATASETS[1:]

# ---------------------------------------------------------------------------
# Dataset constants — UCI Individual Household Electric Power Consumption
# ---------------------------------------------------------------------------
TARGET_COLUMN: Final[str] = "Global_active_power"
DATETIME_COLUMN: Final[str] = "datetime"
RAW_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
)
MISSING_TOKEN: Final[str] = "?"

# ---------------------------------------------------------------------------
# Modelling defaults
# ---------------------------------------------------------------------------
RANDOM_STATE: Final[int] = 42

# Default resampling frequency for the bundled UCI dataset.
# Hourly keeps ~34.5k rows — enough seasonality with reasonable runtime.
DEFAULT_FREQ: Final[str] = "H"

# Seasonal period m used by SARIMA / seasonal naive / Fourier features.
# Hourly data → m = 24 (daily seasonality); daily data → m = 7 (weekly).
FREQ_TO_SEASON: Final[dict[str, int]] = {
    "T": 60,        # minute → hourly seasonality
    "min": 60,
    "H": 24,        # hourly → daily seasonality
    "h": 24,
    "D": 7,         # daily → weekly seasonality
    "W": 52,        # weekly → yearly seasonality
    "M": 12,
}


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for the feature engineering stage."""

    lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24, 48, 168)
    rolling_windows: tuple[int, ...] = (3, 6, 12, 24, 48, 168)
    fourier_orders: tuple[tuple[int, int], ...] = ((24, 3), (168, 2))
    add_calendar: bool = True
    drop_na: bool = True


@dataclass(frozen=True)
class SplitConfig:
    """Chronological train/test split.

    For time series we MUST split chronologically — random shuffles leak
    future information into the training set and inflate metrics.
    """

    test_size: float = 0.2
    val_size: float = 0.1  # taken from the *end* of the training set


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    freq: str = DEFAULT_FREQ
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    random_state: int = RANDOM_STATE
    n_jobs: int = -1
