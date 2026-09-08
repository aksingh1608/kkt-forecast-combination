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

"""Partial-dependence plots — model-agnostic.

We avoid sklearn's ``PartialDependenceDisplay`` because it requires a
``BaseEstimator`` interface and does not gracefully support our pipeline
wrappers; the simple manual implementation below works for any callable
``predict``-style object.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def partial_dependence_grid(
    estimator,
    X: pd.DataFrame,
    feature: str,
    *,
    grid_size: int = 30,
    sample_size: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a (grid_value, mean_prediction) DataFrame for one feature."""
    if feature not in X.columns:
        raise KeyError(f"feature {feature!r} not in X.")
    rng = np.random.default_rng(seed)
    if len(X) > sample_size:
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_sub = X.iloc[idx].copy()
    else:
        X_sub = X.copy()
    grid = np.quantile(X[feature].dropna(), np.linspace(0.02, 0.98, grid_size))
    # Deduplicate (e.g. integer-valued columns).
    grid = np.unique(grid)
    means = []
    Xv = X_sub.copy()
    for v in grid:
        Xv[feature] = v
        preds = estimator.predict(Xv.values)
        means.append(float(np.mean(preds)))
    return pd.DataFrame({feature: grid, "mean_prediction": means})


def partial_dependence_for_features(
    estimator,
    X: pd.DataFrame,
    features: Iterable[str],
    **kwargs,
) -> dict[str, pd.DataFrame]:
    return {f: partial_dependence_grid(estimator, X, f, **kwargs) for f in features}
