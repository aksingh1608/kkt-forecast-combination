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

"""Bayesian hyperparameter optimisation via Optuna.

Search spaces are kept narrow on purpose — the goal is competition-grade
defaults, not a $10 cloud bill. Use ``n_trials=30`` for a quick sweep
and ``n_trials=200`` for a paper-grade run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..evaluation.metrics import rmse

logger = logging.getLogger(__name__)


@dataclass
class OptunaResult:
    best_params: dict
    best_value: float
    n_trials: int
    history: pd.DataFrame


def _split_validation(
    X: pd.DataFrame, y: pd.Series, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    n_val = max(50, int(len(y) * val_frac))
    return (X.iloc[:-n_val], X.iloc[-n_val:],
            y.iloc[:-n_val], y.iloc[-n_val:])


def tune_lightgbm(
    X: pd.DataFrame, y: pd.Series, *,
    n_trials: int = 30, val_frac: float = 0.15,
    seed: int = 42, n_jobs: int = -1,
) -> OptunaResult:
    """Tune LightGBM via Optuna's TPE sampler.

    Returns the best params; the caller can plug them into a fresh
    LGBMRegressor for the headline benchmark.
    """
    import optuna
    from lightgbm import LGBMRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    Xtr, Xva, ytr, yva = _split_validation(X, y, val_frac)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
        }
        model = LGBMRegressor(
            random_state=seed, n_jobs=n_jobs, verbosity=-1, **params
        )
        model.fit(Xtr.values, ytr.values)
        preds = model.predict(Xva.values)
        return rmse(yva.values, preds)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    history = pd.DataFrame([
        {"trial": t.number, "value": t.value, **t.params}
        for t in study.trials if t.value is not None
    ])
    return OptunaResult(
        best_params=study.best_params,
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        history=history,
    )


def tune_catboost(
    X: pd.DataFrame, y: pd.Series, *,
    n_trials: int = 30, val_frac: float = 0.15,
    seed: int = 42,
) -> OptunaResult:
    """Tune CatBoost via Optuna."""
    import optuna
    from catboost import CatBoostRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    Xtr, Xva, ytr, yva = _split_validation(X, y, val_frac)

    def objective(trial):
        params = dict(
            iterations=trial.suggest_int("iterations", 200, 1500, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        )
        model = CatBoostRegressor(
            random_seed=seed, verbose=False, allow_writing_files=False, **params,
        )
        model.fit(Xtr.values, ytr.values)
        return rmse(yva.values, model.predict(Xva.values))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    history = pd.DataFrame([
        {"trial": t.number, "value": t.value, **t.params}
        for t in study.trials if t.value is not None
    ])
    return OptunaResult(
        best_params=study.best_params,
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        history=history,
    )


# ---------------------------------------------------------------------------
# Deep-learning HPO via NeuralForecast
# ---------------------------------------------------------------------------
def tune_neuralforecast_arch(
    y: pd.Series, *,
    arch: str = "nhits",
    n_trials: int = 30,
    val_frac: float = 0.15,
    seed: int = 42,
    horizon: int = 24,
) -> OptunaResult:
    """Generic Optuna search over a NeuralForecast architecture.

    Searches over input_size, max_steps, learning_rate, hidden_size /
    n_blocks where applicable. Returns the best params; the caller can
    instantiate :class:`NeuralForecastWrapper` with them for the final
    paper run.
    """
    import optuna
    from ..models.optional_dl import NeuralForecastWrapper

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_val = max(48, int(len(y) * val_frac))
    y_train = y.iloc[:-n_val]
    y_val = y.iloc[-n_val:]

    def objective(trial):
        params = dict(
            arch=arch,
            horizon=horizon,
            input_size=trial.suggest_categorical("input_size", [96, 168, 336, 512]),
            max_steps=trial.suggest_int("max_steps", 100, 600, step=50),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        )
        m = NeuralForecastWrapper(name=arch.upper(), **params)
        try:
            m.fit(y_train)
            preds = m.predict(n_val)
        except Exception as exc:  # pragma: no cover
            logger.warning("NF trial failed (%s)", exc)
            return float("inf")
        if not np.all(np.isfinite(preds)):
            return float("inf")
        return rmse(y_val.values, preds)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    history = pd.DataFrame([
        {"trial": t.number, "value": t.value, **t.params}
        for t in study.trials if t.value is not None
    ])
    return OptunaResult(
        best_params=study.best_params,
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        history=history,
    )


def tune_lite_linear_model(
    y: pd.Series, *,
    model_name: str = "dlinear",
    n_trials: int = 40,
    val_frac: float = 0.15,
    seed: int = 42,
) -> OptunaResult:
    """Tune the torch-free NLinear / DLinear baselines.

    These models forecast recursively from their own history. The objective
    therefore validates on a full recursive rollout over the final validation
    slice, matching how the headline one-step/recursive rows are produced for
    non-feature-consuming forecasters in the main pipeline.
    """
    import optuna
    from ..models.lite_dl import DLinearForecaster, NLinearForecaster

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_val = max(48, int(len(y) * val_frac))
    y_train = y.iloc[:-n_val]
    y_val = y.iloc[-n_val:]
    name = model_name.lower()
    if name not in {"nlinear", "dlinear"}:
        raise ValueError("model_name must be 'nlinear' or 'dlinear'")

    max_lookback = max(24, min(336, len(y_train) // 3))
    lookbacks = [lb for lb in (24, 48, 96, 168, 336) if lb <= max_lookback]
    if not lookbacks:
        lookbacks = [max(4, max_lookback)]

    def objective(trial: "optuna.Trial") -> float:
        lookback = trial.suggest_categorical("lookback", lookbacks)
        clip_quantile = trial.suggest_categorical(
            "clip_quantile", [0.95, 0.975, 0.99, 0.995]
        )
        try:
            if name == "nlinear":
                model = NLinearForecaster(
                    lookback=int(lookback),
                    ridge=trial.suggest_float("ridge", 1e-4, 10.0, log=True),
                    clip_quantile=float(clip_quantile),
                )
            else:
                # Optuna requires a stable categorical value space for a
                # parameter name across all trials. Suggest from the full
                # set, then reject values that are too large for the chosen
                # lookback.
                kernel_size = int(trial.suggest_categorical("kernel_size", [3, 7, 13, 25, 49]))
                if kernel_size > int(lookback):
                    return float("inf")
                model = DLinearForecaster(
                    lookback=int(lookback),
                    kernel_size=kernel_size,
                    ridge_trend=trial.suggest_float("ridge_trend", 1e-4, 10.0, log=True),
                    ridge_seasonal=trial.suggest_float("ridge_seasonal", 1e-3, 100.0, log=True),
                    use_revin=trial.suggest_categorical("use_revin", [True, False]),
                    clip_quantile=float(clip_quantile),
                )
            model.fit(y_train)
            preds = model.predict(len(y_val))
        except Exception as exc:
            logger.warning("%s trial failed: %s", model_name, exc)
            return float("inf")
        if not np.all(np.isfinite(preds)):
            return float("inf")
        return rmse(y_val.values, preds)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    history = pd.DataFrame([
        {"trial": t.number, "value": t.value, **t.params}
        for t in study.trials if t.value is not None
    ])
    return OptunaResult(
        best_params=study.best_params,
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        history=history,
    )


def tune_xgboost(
    X: pd.DataFrame, y: pd.Series, *,
    n_trials: int = 30, val_frac: float = 0.15,
    seed: int = 42, n_jobs: int = -1,
) -> OptunaResult:
    """Tune XGBoost via Optuna."""
    import optuna
    from xgboost import XGBRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    Xtr, Xva, ytr, yva = _split_validation(X, y, val_frac)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        }
        model = XGBRegressor(
            tree_method="hist", random_state=seed, n_jobs=n_jobs,
            verbosity=0, **params,
        )
        model.fit(Xtr.values, ytr.values)
        preds = model.predict(Xva.values)
        return rmse(yva.values, preds)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    history = pd.DataFrame([
        {"trial": t.number, "value": t.value, **t.params}
        for t in study.trials if t.value is not None
    ])
    return OptunaResult(
        best_params=study.best_params,
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        history=history,
    )
