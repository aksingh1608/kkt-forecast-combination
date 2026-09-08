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

"""Compute / sustainability profiling.

* Parameter / leaf counts for tree models.
* Inference latency benchmark (median over N runs).
* "Accuracy per second" trade-off — useful for the Pareto plot in the
  paper (the 2026 reviewer-favourite).
* Optional CO₂ estimate via :mod:`codecarbon` if installed.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------
def _unwrap(estimator: Any) -> Any:
    if hasattr(estimator, "named_steps"):
        return list(estimator.named_steps.values())[-1]
    return estimator


def estimate_param_count(estimator: Any) -> int:
    """Return a rough parameter / leaf count for any estimator."""
    inner = _unwrap(estimator)

    # XGBoost
    if hasattr(inner, "get_booster"):
        try:
            booster = inner.get_booster()
            dump = booster.get_dump()
            return sum(d.count("leaf=") for d in dump)
        except Exception:  # pragma: no cover
            pass

    # LightGBM
    if hasattr(inner, "booster_") and hasattr(inner.booster_, "num_trees"):
        try:
            stats = inner.booster_.dump_model()
            return sum(t.get("num_leaves", 0) for t in stats.get("tree_info", []))
        except Exception:  # pragma: no cover
            pass

    # CatBoost
    if hasattr(inner, "tree_count_"):
        try:
            return int(inner.tree_count_) * int(getattr(inner, "depth", 6)) ** 2
        except Exception:  # pragma: no cover
            pass

    # sklearn forest / tree ensembles
    if hasattr(inner, "estimators_"):
        try:
            return int(sum(e.tree_.node_count
                           if hasattr(e, "tree_") else 1 for e in inner.estimators_))
        except Exception:  # pragma: no cover
            pass

    # Linear models
    if hasattr(inner, "coef_"):
        coef = np.asarray(inner.coef_).ravel()
        return int(coef.size + (1 if hasattr(inner, "intercept_") else 0))

    return -1   # unknown


# ---------------------------------------------------------------------------
# Inference latency
# ---------------------------------------------------------------------------
def measure_inference_latency(
    predict_callable, X, *, n_runs: int = 5, warmup: int = 1
) -> dict[str, float]:
    """Median inference latency in milliseconds (per call) over ``n_runs``."""
    for _ in range(max(0, warmup)):
        predict_callable(X)
    times = []
    for _ in range(max(1, n_runs)):
        t0 = time.perf_counter()
        predict_callable(X)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "n_rows": int(len(X)),
        "rows_per_second": float(len(X) / (np.median(times) / 1000.0)),
    }


# ---------------------------------------------------------------------------
# Accuracy / second trade-off
# ---------------------------------------------------------------------------
def accuracy_per_second(
    leaderboard: pd.DataFrame,
    *,
    metric: str = "RMSE",
    time_col: str = "TrainTime(s)",
) -> pd.DataFrame:
    """Compute (accuracy, total_time, accuracy/second) for a leaderboard."""
    if metric not in leaderboard.columns or time_col not in leaderboard.columns:
        return leaderboard
    df = leaderboard.copy()
    # Use 1/RMSE so larger = better, then rescale so the best gets 1.
    raw_acc = 1.0 / df[metric].replace(0, np.nan)
    df["accuracy_score"] = raw_acc / raw_acc.max()
    df["total_time_s"] = df[time_col] + df.get("PredictTime(s)", 0.0)
    df["acc_per_sec"] = df["accuracy_score"] / df["total_time_s"].clip(lower=1e-3)
    return df


# ---------------------------------------------------------------------------
# Optional codecarbon
# ---------------------------------------------------------------------------
@contextmanager
def gpu_memory_tracker():
    """Context manager that yields a dict with peak VRAM allocation (MB).

    Wraps ``torch.cuda.reset_peak_memory_stats`` /
    ``torch.cuda.max_memory_allocated``. On CPU-only systems the dict
    just contains zeros so callers don't need to branch.
    """
    info = {"peak_mb": 0.0, "alloc_mb_at_exit": 0.0, "available": False}
    try:
        import torch
        if not torch.cuda.is_available():
            yield info
            return
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        info["available"] = True
        try:
            yield info
        finally:
            torch.cuda.synchronize()
            info["peak_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            info["alloc_mb_at_exit"] = float(torch.cuda.memory_allocated() / (1024 ** 2))
    except ImportError:
        yield info


@contextmanager
def co2_emissions_estimate(*, project_name: str = "energy-forecasting"):
    """Context manager wrapping `codecarbon.OfflineEmissionsTracker`.

    Yields the tracker so callers can inspect the running totals; emits
    a dict with `kwh` and `co2_kg` once the context exits.
    Falls back to a no-op if codecarbon is not installed.
    """
    try:
        from codecarbon import OfflineEmissionsTracker
    except ImportError:  # pragma: no cover
        logger.warning("codecarbon not installed — CO2 estimate skipped.")
        yield {"kwh": 0.0, "co2_kg": 0.0, "available": False}
        return

    result = {"kwh": 0.0, "co2_kg": 0.0, "available": True}
    tracker = OfflineEmissionsTracker(
        project_name=project_name, country_iso_code="FRA",
        log_level="error", save_to_file=False,
    )
    tracker.start()
    try:
        yield result
    finally:
        emissions = tracker.stop() or 0.0
        result["co2_kg"] = float(emissions)
        result["kwh"] = float(getattr(tracker.final_emissions_data, "energy_consumed", 0.0))
