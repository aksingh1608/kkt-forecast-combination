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

"""Symmetric recursive multi-step protocol.

The paper's central claim about tree-vs-classical dominance depends on
evaluating **every** model under the same recursive protocol. By default,
supervised learners in this repository score a "teacher-forced" one-step
target: their lag / rolling features at test step *t* come from the *true*
past, not from the model's own past predictions.

This module provides :func:`recursive_predict` which rolls a fitted
supervised model out **using only its own previous forecasts** to update
the lag / rolling columns of the feature matrix. Calendar and Fourier
features are static (they depend on the timestamp, not on ``y``), so they
are copied from the pre-computed ``X_test`` as-is.

The rollout is designed to be a drop-in replacement for
``model.predict(n_periods, X_test=X_test)``:

    y_pred = recursive_predict(model, X_test, y_history, target=target,
                               feature_cfg=feature_cfg)

so the pipeline can trivially benchmark every supervised learner under
the *identical* recursive protocol that ARIMA / SARIMA / Holt-Winters
face by default (per Eq. 2 of the manuscript).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import FeatureConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature-column naming (must match src.features.engineering)
# ---------------------------------------------------------------------------
def _lag_col(target: str, lag: int) -> str:
    return f"{target}_lag_{lag}"


def _roll_col(target: str, stat: str, w: int) -> str:
    return f"{target}_roll_{stat}_{w}"


ROLL_STATS: tuple[str, ...] = ("mean", "std", "min", "max")


# ---------------------------------------------------------------------------
# Rollout implementation
# ---------------------------------------------------------------------------
@dataclass
class RecursiveDiagnostics:
    """Optional per-step diagnostics returned when ``return_diagnostics=True``."""

    updated_columns: list[str]
    history_len_start: int
    n_steps: int


def _rolling_stats(values: np.ndarray, w: int) -> dict[str, float]:
    """Compute rolling stats on the last ``w`` (or fewer) elements of ``values``.

    Uses the same ``min_periods = max(1, w // 2)`` policy as
    :func:`src.features.engineering.add_rolling_features` so the causal
    features match the training-time semantics exactly.
    """
    min_periods = max(1, w // 2)
    tail = values[-w:] if len(values) >= 1 else values
    if len(tail) < min_periods:
        return {stat: float("nan") for stat in ROLL_STATS}
    return {
        "mean": float(tail.mean()),
        "std":  float(tail.std(ddof=1)) if len(tail) > 1 else 0.0,
        "min":  float(tail.min()),
        "max":  float(tail.max()),
    }


def _updateable_columns(
    columns: Iterable[str], target: str, cfg: FeatureConfig
) -> tuple[list[tuple[str, int]], list[tuple[str, str, int]]]:
    """Return the (lag, roll) column groups present in ``columns``."""
    col_set = set(columns)
    lag_cols = [
        (_lag_col(target, k), k)
        for k in cfg.lags if _lag_col(target, k) in col_set
    ]
    roll_cols: list[tuple[str, str, int]] = []
    for w in cfg.rolling_windows:
        for stat in ROLL_STATS:
            name = _roll_col(target, stat, w)
            if name in col_set:
                roll_cols.append((name, stat, w))
    return lag_cols, roll_cols


def recursive_predict(
    model,
    X_test: pd.DataFrame,
    y_history: pd.Series | np.ndarray,
    *,
    target: str,
    feature_cfg: FeatureConfig | None = None,
    clip_range: tuple[float, float] | None = None,
    return_diagnostics: bool = False,
):
    """Roll out a supervised forecaster **using only its own predictions**.

    The implementation operates on a raw numpy view of ``X_test`` for the
    inner loop and reconstructs a lightweight one-row DataFrame per step
    only for the model's public ``predict`` API. When the model exposes an
    underlying sklearn estimator (via :meth:`SupervisedForecaster.get_underlying_estimator`),
    the DataFrame construction is skipped entirely and the numpy row is
    passed directly to ``estimator.predict`` — a ~10-50× speedup versus
    the naive pandas path on high-cardinality feature frames.

    Parameters
    ----------
    model
        A fitted :class:`src.models.base.ForecastModel` that expects
        engineered features. Only ``model.predict(n_periods=1, X_test=one_row)``
        is required for the fallback path; ``SupervisedForecaster`` instances
        take the fast numpy path.
    X_test
        The pre-computed feature frame for the test window. Calendar and
        Fourier columns are used as-is; lag and rolling columns of the
        target are **overwritten** at each step from the recursive history.
    y_history
        The complete target history *up to* (but excluding) the first test
        step. This is the training target — typically ``y_train``. Must be
        long enough to supply the largest lag.
    target
        Target column name used to build the lag / rolling column names.
    feature_cfg
        Feature configuration used at training time. Defaults to
        :class:`FeatureConfig` — must match the pipeline configuration.
    clip_range
        Optional ``(lo, hi)`` clip applied to each recursive prediction.
        Prevents numerical blow-up on unstable models.
    return_diagnostics
        If True, also return a :class:`RecursiveDiagnostics` object.

    Returns
    -------
    preds : ndarray of shape (n_steps,)
        The recursive multi-step forecast. Same length as ``X_test``.
    """
    cfg = feature_cfg or FeatureConfig()
    n_steps = len(X_test)
    if n_steps == 0:
        return np.array([])

    lag_cols, roll_cols = _updateable_columns(X_test.columns, target, cfg)
    if not lag_cols and not roll_cols:
        logger.warning(
            "recursive_predict: no lag/rolling columns of %r found in X_test — "
            "the rollout reduces to a static feature prediction.", target
        )

    y_hist = np.asarray(y_history, dtype=float).ravel()
    max_lag = max((k for _, k in lag_cols), default=0)
    max_win = max((w for _, _, w in roll_cols), default=0)
    if max_lag and len(y_hist) < max_lag:
        raise ValueError(
            f"y_history has {len(y_hist)} points; need at least "
            f"max_lag={max_lag} to seed the recursion."
        )

    # Precompute column-name → integer index mapping once.
    columns = list(X_test.columns)
    col_index = {c: i for i, c in enumerate(columns)}
    lag_col_idx: list[tuple[int, int]] = [
        (col_index[name], k) for name, k in lag_cols
    ]
    # Group rolling columns by window size for cache reuse.
    roll_col_idx: dict[int, list[tuple[int, str]]] = {}
    for name, stat, w in roll_cols:
        roll_col_idx.setdefault(w, []).append((col_index[name], stat))

    # Detect the fast path: unwrap SupervisedForecaster → underlying sklearn
    # estimator so we can call estimator.predict(numpy_row) directly.
    fast_estimator = None
    try:
        get_est = getattr(model, "get_underlying_estimator", None)
        if callable(get_est):
            fast_estimator = get_est()
    except Exception:
        fast_estimator = None

    X_arr = X_test.values.astype(float, copy=True)
    preds = np.empty(n_steps, dtype=float)

    warmup_len = max(max_lag, max_win, 1)
    history_deque = np.asarray(
        y_hist[-warmup_len - n_steps:] if len(y_hist) >= warmup_len else y_hist,
        dtype=float,
    ).copy()
    # We keep ``history`` as a plain Python list to allow O(1) appends;
    # rolling windows convert to numpy on demand.
    history: list[float] = history_deque.tolist()

    for h in range(n_steps):
        for col_i, k in lag_col_idx:
            X_arr[h, col_i] = history[-k]

        if roll_col_idx:
            # We only need the last ``max_win`` history points to recompute
            # any rolling statistic (min_periods logic mirrors training).
            tail_np = np.asarray(history[-max_win:], dtype=float)
            for w, stat_positions in roll_col_idx.items():
                stats = _rolling_stats(tail_np if w >= max_win else tail_np[-w:], w)
                for col_i, stat in stat_positions:
                    X_arr[h, col_i] = stats[stat]

        row = X_arr[h:h + 1]
        if fast_estimator is not None:
            y_hat = float(np.asarray(fast_estimator.predict(row)).ravel()[0])
        else:
            row_df = pd.DataFrame(row, columns=columns, index=X_test.index[h:h + 1])
            try:
                y_hat = float(np.asarray(model.predict(1, X_test=row_df)).ravel()[0])
            except TypeError:
                y_hat = float(np.asarray(model.predict(row)).ravel()[0])

        if not np.isfinite(y_hat):
            y_hat = float(history[-1])
        if clip_range is not None:
            lo, hi = clip_range
            y_hat = float(np.clip(y_hat, lo, hi))

        preds[h] = y_hat
        history.append(y_hat)

    if return_diagnostics:
        diag = RecursiveDiagnostics(
            updated_columns=[c for c, _ in lag_cols] + [c for c, _, _ in roll_cols],
            history_len_start=len(y_hist),
            n_steps=n_steps,
        )
        return preds, diag
    return preds


__all__ = ["recursive_predict", "RecursiveDiagnostics"]
