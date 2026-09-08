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

"""Expanding-window walk-forward validation utilities."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def walk_forward_splits(
    n_samples: int,
    *,
    n_splits: int = 5,
    min_train_size: int | None = None,
    horizon: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) pairs for expanding-window CV.

    For ``n_splits = K`` we hold out the final ``K * horizon`` observations
    as test folds. Each fold uses ALL previous observations as the training
    set — i.e. a strictly expanding window, which is the standard scheme for
    time-series cross-validation (Bergmeir & Benítez, 2012).
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if horizon is None:
        horizon = max(1, n_samples // (n_splits * 5))
    min_train = min_train_size or n_samples - n_splits * horizon
    if min_train <= 0:
        raise ValueError(
            f"n_samples={n_samples} too small for n_splits={n_splits}, "
            f"horizon={horizon}"
        )
    for k in range(n_splits):
        train_end = min_train + k * horizon
        test_end = train_end + horizon
        yield np.arange(train_end), np.arange(train_end, test_end)


def walk_forward_predict(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Run walk-forward CV and return per-fold metrics.

    ``model_factory`` must be a zero-arg callable returning a fresh sklearn-
    compatible regressor. Returns a DataFrame with one row per fold.
    """
    from .metrics import compute_all_metrics

    rows = []
    for fold, (tr_idx, te_idx) in enumerate(
        walk_forward_splits(len(X), n_splits=n_splits, horizon=horizon)
    ):
        model = model_factory()
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = model.predict(X.iloc[te_idx])
        metrics = compute_all_metrics(
            y.iloc[te_idx], pred, y_train=y.iloc[tr_idx]
        )
        rows.append({"fold": fold, **metrics})
    return pd.DataFrame(rows)
