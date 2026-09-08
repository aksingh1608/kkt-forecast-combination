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

"""Advanced Time-Series Forecasting for Household Energy Consumption.

This package implements the experimental pipeline for the MSc AI Machine
Learning Fundamentals assignment (BSBI, Feb 2026) and accompanies the
research paper draft in ``paper/research_paper.md``.

Sub-packages
------------
data            Loading, cleaning and resampling utilities.
features        Feature engineering for time-series regression.
eda             Exploratory data analysis & publication-quality plots.
models          Classical, ML/GBM and the novel FAHE architecture.
evaluation      Metrics, walk-forward validation and statistical tests.
"""

from __future__ import annotations

import warnings as _warnings

# Suppress sklearn's harmless "X does not have valid feature names" warning
# that fires when LightGBM/XGBoost are predicted on a NumPy array — we
# intentionally strip column names for speed and uniformity across models.
_warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

__version__ = "1.0.0"
__author__ = "BSBI MSc AI — MLF Final Assignment"
