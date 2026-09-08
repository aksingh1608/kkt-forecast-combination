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

"""Machine-learning regressors wrapped as :class:`SupervisedForecaster`.

The zoo deliberately spans the classical→modern continuum so the comparison
in the paper has breadth:

* **Linear**:   Ridge, Lasso, ElasticNet.
* **Distance**: KNN.
* **Kernel**:   SVR (RBF kernel).
* **Tree ensembles**: RandomForest, ExtraTrees, GradientBoosting,
  HistGradientBoosting.
* **Boosting libraries**: XGBoost, LightGBM, CatBoost.
"""

from __future__ import annotations

import logging
from typing import Callable

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ..config import RANDOM_STATE
from .base import SupervisedForecaster

logger = logging.getLogger(__name__)


def _wrap(name: str, family: str, estimator) -> SupervisedForecaster:
    return SupervisedForecaster(name=name, family=family, estimator=estimator)


def build_ml_zoo(
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> list[SupervisedForecaster]:
    """Construct the canonical ML zoo used in the experiments."""
    zoo: list[SupervisedForecaster] = []

    # --- Linear ---------------------------------------------------------------
    zoo.append(_wrap("Ridge", "linear", Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=random_state)),
    ])))
    zoo.append(_wrap("Lasso", "linear", Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.001, random_state=random_state, max_iter=20000)),
    ])))
    zoo.append(_wrap("ElasticNet", "linear", Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(
            alpha=0.001, l1_ratio=0.5, random_state=random_state, max_iter=20000
        )),
    ])))

    # --- Distance / kernel ----------------------------------------------------
    zoo.append(_wrap("KNN", "instance", Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=10, n_jobs=n_jobs)),
    ])))
    zoo.append(_wrap("SVR", "kernel", Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    ])))

    # --- Tree ensembles -------------------------------------------------------
    zoo.append(_wrap("RandomForest", "tree-ensemble", RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=n_jobs,
        random_state=random_state,
    )))
    zoo.append(_wrap("ExtraTrees", "tree-ensemble", ExtraTreesRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=n_jobs,
        random_state=random_state,
    )))
    zoo.append(_wrap("GradientBoosting", "boosting", GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_state,
    )))
    zoo.append(_wrap("HistGradientBoosting", "boosting", HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_depth=8,
        random_state=random_state,
    )))

    # --- Modern boosting libraries -------------------------------------------
    try:
        from xgboost import XGBRegressor

        zoo.append(_wrap("XGBoost", "boosting", XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=0,
        )))
    except ImportError:  # pragma: no cover
        logger.warning("xgboost not installed — skipping XGBoost model.")

    try:
        from lightgbm import LGBMRegressor

        zoo.append(_wrap("LightGBM", "boosting", LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=-1,
        )))
    except ImportError:  # pragma: no cover
        logger.warning("lightgbm not installed — skipping LightGBM model.")

    try:
        from catboost import CatBoostRegressor

        zoo.append(_wrap("CatBoost", "boosting", CatBoostRegressor(
            iterations=600,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3.0,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )))
    except ImportError:  # pragma: no cover
        logger.warning("catboost not installed — skipping CatBoost model.")

    return zoo


# Convenience callable for walk-forward CV factories.
def make_factory(estimator) -> Callable:
    """Return a zero-arg factory that clones the given estimator."""
    from sklearn.base import clone

    def factory():
        return clone(estimator)

    return factory
