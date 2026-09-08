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

"""Cross-dataset analytics.

Once two or more :class:`~src.results_store.BenchmarkSnapshot` records exist,
this module answers the questions that matter for a comparative paper:

* **Which (dataset, algorithm) pair is the global champion?**
* **Which algorithm wins most often** across the suite (average rank)?
* **How consistent** is each algorithm — does it always finish near the top
  or is it volatile?
* **Are the differences statistically significant?** (Friedman omnibus)

Cross-dataset comparison requires **scale-invariant metrics**. We therefore
report rankings primarily on MASE (Hyndman & Koehler, 2006) and on the
per-dataset rank — never on raw RMSE/MAE which are not commensurable
across series of different magnitudes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .evaluation.statistical_tests import friedman_test
from .results_store import BenchmarkSnapshot

logger = logging.getLogger(__name__)

# Metrics where *smaller* is better.
LOWER_IS_BETTER = {"MAE", "RMSE", "MAPE", "sMAPE", "MASE"}
# Default cross-dataset metric — scale invariant.
DEFAULT_CROSS_METRIC: str = "MASE"


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------
def build_metric_matrix(
    snapshots: Sequence[BenchmarkSnapshot],
    metric: str = DEFAULT_CROSS_METRIC,
) -> pd.DataFrame:
    """Return a DataFrame with rows=datasets, columns=models, values=metric."""
    rows: list[pd.Series] = []
    index: list[str] = []
    for snap in snapshots:
        if metric not in snap.leaderboard.columns:
            logger.warning("Snapshot %s lacks metric %s — skipped.", snap.name, metric)
            continue
        rows.append(snap.leaderboard[metric].rename(snap.name))
        index.append(snap.name)
    if not rows:
        return pd.DataFrame()
    matrix = pd.concat(rows, axis=1).T
    matrix.index.name = "dataset"
    matrix.columns.name = "model"
    return matrix


def build_rank_matrix(
    snapshots: Sequence[BenchmarkSnapshot],
    metric: str = DEFAULT_CROSS_METRIC,
) -> pd.DataFrame:
    """Per-dataset ranks. Rank 1 = best model on that dataset."""
    matrix = build_metric_matrix(snapshots, metric=metric)
    if matrix.empty:
        return matrix
    ascending = metric in LOWER_IS_BETTER
    return matrix.rank(axis=1, ascending=ascending, method="min", na_option="bottom")


# ---------------------------------------------------------------------------
# Champion tables
# ---------------------------------------------------------------------------
@dataclass
class GlobalChampion:
    dataset: str
    model: str
    metric: str
    value: float


def champions_per_dataset(
    snapshots: Sequence[BenchmarkSnapshot],
    metric: str = DEFAULT_CROSS_METRIC,
) -> pd.DataFrame:
    """One row per dataset: best model, second best and the gap between them."""
    rows = []
    ascending = metric in LOWER_IS_BETTER
    for snap in snapshots:
        if metric not in snap.leaderboard.columns:
            continue
        col = snap.leaderboard[metric].dropna()
        if col.empty:
            continue
        sorted_col = col.sort_values(ascending=ascending)
        winner = sorted_col.index[0]
        runner_up = sorted_col.index[1] if len(sorted_col) > 1 else None
        gap = (
            sorted_col.iloc[1] - sorted_col.iloc[0]
            if len(sorted_col) > 1 else float("nan")
        )
        rows.append({
            "Dataset": snap.name,
            "Winner": winner,
            f"Winner {metric}": float(sorted_col.iloc[0]),
            "Runner-up": runner_up,
            f"Runner-up {metric}": float(sorted_col.iloc[1]) if runner_up else float("nan"),
            "Gap": float(gap),
            "n_models": int(col.count()),
        })
    return pd.DataFrame(rows)


def global_champion(
    snapshots: Sequence[BenchmarkSnapshot],
    metric: str = DEFAULT_CROSS_METRIC,
) -> GlobalChampion | None:
    """The single best (dataset, model) pair across every snapshot."""
    matrix = build_metric_matrix(snapshots, metric=metric)
    if matrix.empty:
        return None
    ascending = metric in LOWER_IS_BETTER
    stacked = matrix.stack(future_stack=True).dropna()
    if stacked.empty:
        return None
    idx = stacked.idxmin() if ascending else stacked.idxmax()
    dataset, model = idx
    return GlobalChampion(
        dataset=dataset, model=model, metric=metric, value=float(stacked.loc[idx])
    )


# ---------------------------------------------------------------------------
# Per-algorithm summary
# ---------------------------------------------------------------------------
def algorithm_summary(
    snapshots: Sequence[BenchmarkSnapshot],
    metric: str = DEFAULT_CROSS_METRIC,
) -> pd.DataFrame:
    """For every algorithm: mean rank, wins, top-3 finishes, mean metric.

    The headline column is ``mean_rank`` — the lower, the more consistently
    that algorithm wins across the dataset suite (this is the
    well-known *average-rank* heuristic from Demšar, 2006).
    """
    ranks = build_rank_matrix(snapshots, metric=metric)
    values = build_metric_matrix(snapshots, metric=metric)
    if ranks.empty:
        return pd.DataFrame()
    n_datasets = len(ranks)

    out = pd.DataFrame({
        "mean_rank": ranks.mean(axis=0),
        "median_rank": ranks.median(axis=0),
        "std_rank": ranks.std(axis=0, ddof=0),
        "wins": (ranks == 1).sum(axis=0),
        "top3": (ranks <= 3).sum(axis=0),
        f"mean_{metric}": values.mean(axis=0),
        f"std_{metric}": values.std(axis=0, ddof=0),
    })
    out["win_rate_%"] = (out["wins"] / n_datasets * 100).round(1)
    out["top3_rate_%"] = (out["top3"] / n_datasets * 100).round(1)
    return out.sort_values("mean_rank")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def friedman_across_datasets(
    snapshots: Sequence[BenchmarkSnapshot],
    metric: str = DEFAULT_CROSS_METRIC,
) -> tuple[float, float] | None:
    """Friedman test on the (dataset × algorithm) metric matrix.

    Low p-value ⇒ at least one algorithm performs significantly
    differently from the others across the dataset suite.
    """
    matrix = build_metric_matrix(snapshots, metric=metric).dropna(axis=1, how="any")
    if matrix.shape[0] < 2 or matrix.shape[1] < 3:
        return None
    try:
        return friedman_test(matrix)
    except Exception as exc:  # pragma: no cover
        logger.warning("Friedman test failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Convenience for the UI
# ---------------------------------------------------------------------------
def best_pair_overall(
    snapshots: Sequence[BenchmarkSnapshot],
    metric: str = DEFAULT_CROSS_METRIC,
) -> dict | None:
    """High-level summary feeding the headline card in the UI."""
    champ = global_champion(snapshots, metric=metric)
    if champ is None:
        return None
    algo_table = algorithm_summary(snapshots, metric=metric)
    most_consistent = algo_table.index[0] if not algo_table.empty else None
    return {
        "best_pair": champ,
        "most_consistent_algorithm": most_consistent,
        "n_snapshots": len(snapshots),
        "n_models": int(
            build_metric_matrix(snapshots, metric=metric).shape[1]
        ) if snapshots else 0,
    }
