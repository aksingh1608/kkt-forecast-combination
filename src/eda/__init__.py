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

"""Exploratory data analysis utilities (Matplotlib + Plotly)."""

from .analysis import (
    summary_table,
    seasonal_profile,
    plot_overview,
    plot_decomposition,
    plot_correlation_heatmap,
    plot_acf_pacf,
    plot_seasonal_profiles,
)
from .extra_plots import (
    plot_box_by,
    plot_hour_dow_heatmap,
    plot_target_histogram,
    plot_residuals_over_time,
    plot_error_heatmap,
    plot_fan_chart,
    plot_calibration,
)
from .stationarity import stationarity_report, StationarityReport

__all__ = [
    "summary_table",
    "seasonal_profile",
    "plot_overview",
    "plot_decomposition",
    "plot_correlation_heatmap",
    "plot_acf_pacf",
    "plot_seasonal_profiles",
    "plot_box_by",
    "plot_hour_dow_heatmap",
    "plot_target_histogram",
    "plot_residuals_over_time",
    "plot_error_heatmap",
    "plot_fan_chart",
    "plot_calibration",
    "stationarity_report",
    "StationarityReport",
]
