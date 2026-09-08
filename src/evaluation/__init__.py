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

"""Evaluation metrics, walk-forward validation and statistical tests."""

from .metrics import (
    mae, rmse, mape, smape, r2, mase, compute_all_metrics,
)
from .walk_forward import walk_forward_predict, walk_forward_splits
from .statistical_tests import (
    diebold_mariano, friedman_test,
    pairwise_dm_matrix, cohens_d, cliffs_delta,
    holm_correct, benjamini_hochberg, plot_dm_pvalue_heatmap,
)
from .probabilistic import (
    pinball_loss, average_pinball_loss, crps_from_quantiles,
    coverage, sharpness, winkler_score,
    evaluate_probabilistic, ProbabilisticReport,
)
from .conformal import (
    split_conformal_intervals, adaptive_conformal_intervals, ACIState,
)
from .horizon import (
    per_horizon_metrics, per_horizon_for_models,
    rolling_origin_horizon_metrics, per_horizon_leaderboard,
)
from .bootstrap import bootstrap_metric_ci
from .robustness import (
    noise_injection_curve, missing_injection_curve,
    distribution_shift_eval, train_size_curve,
)
from .meyer_metrics import maeh, apneh, both_metrics as meyer_metrics
from .cd_diagram import critical_difference, plot_cd_diagram, CDResult
from .residuals import ljung_box, ljung_box_for_models, LjungBoxResult
from .leakage import assess_leakage, leakage_report, LeakageHit
from .kkt_analysis import (
    CSEDiagnostics, compute_cse_diagnostics, kkt_conditions,
    proposition1_frontier, plot_kkt_boundary,
)

__all__ = [
    "mae", "rmse", "mape", "smape", "r2", "mase", "compute_all_metrics",
    "walk_forward_predict", "walk_forward_splits",
    "diebold_mariano", "friedman_test",
    "pairwise_dm_matrix", "cohens_d", "cliffs_delta",
    "holm_correct", "benjamini_hochberg", "plot_dm_pvalue_heatmap",
    "pinball_loss", "average_pinball_loss", "crps_from_quantiles",
    "coverage", "sharpness", "winkler_score",
    "evaluate_probabilistic", "ProbabilisticReport",
    "split_conformal_intervals", "adaptive_conformal_intervals", "ACIState",
    "per_horizon_metrics", "per_horizon_for_models",
    "rolling_origin_horizon_metrics", "per_horizon_leaderboard",
    "bootstrap_metric_ci",
    "noise_injection_curve", "missing_injection_curve",
    "distribution_shift_eval", "train_size_curve",
    "maeh", "apneh", "meyer_metrics",
    "critical_difference", "plot_cd_diagram", "CDResult",
    "ljung_box", "ljung_box_for_models", "LjungBoxResult",
    "assess_leakage", "leakage_report", "LeakageHit",
    "CSEDiagnostics", "compute_cse_diagnostics", "kkt_conditions",
    "proposition1_frontier", "plot_kkt_boundary",
]
