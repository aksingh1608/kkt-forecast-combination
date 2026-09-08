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

"""Permutation feature importance.

Wraps sklearn's ``permutation_importance`` with sensible defaults for
tabular time-series regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_tabular(
    estimator,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    n_repeats: int = 10,
    seed: int = 42,
    top_k: int = 20,
    scoring: str = "neg_root_mean_squared_error",
) -> pd.DataFrame:
    """Permutation importance, returned as a tidy mean ± std DataFrame."""
    res = permutation_importance(
        estimator, X.values, np.asarray(y).ravel(),
        n_repeats=n_repeats, random_state=seed,
        scoring=scoring, n_jobs=-1,
    )
    out = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": res.importances_mean,
        "importance_std": res.importances_std,
    })
    return out.sort_values("importance_mean", ascending=False).head(top_k).reset_index(drop=True)
