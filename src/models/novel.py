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

"""FAHE — Frequency-Aware Hybrid Ensemble.

The novel architecture proposed in the accompanying paper. Intuition
---------------------------------------------------------------------
Classical statistical models (SARIMA, Holt-Winters) and gradient-boosted
trees capture *complementary* parts of the energy-consumption signal:

* **Classical models** are sample-efficient at modelling the dominant
  trend and seasonal cycle and degrade *gracefully* over long horizons —
  ideal for long-range forecasts where lag features no longer apply.
* **Gradient-boosted trees** dominate at short horizons by exploiting
  rich feature interactions (calendar, lags, rolling stats, Fourier
  terms) that classical state-space models cannot easily absorb.

FAHE combines them via a **convex non-negative meta-learner**:

    ŷ(t) = w_sarima · ŷ_sarima(t)
         + w_hw     · ŷ_hw(t)
         + w_lgbm   · ŷ_lgbm(t),
    s.t. w_i ≥ 0  and  Σ w_i = 1.

Weights are estimated by non-negative least-squares (NNLS) on a held-out
validation slice of the training set, then the base models are refitted
on the *complete* training set before producing the test forecast.

Why this is a defensible research contribution
---------------------------------------------
* It is **interpretable** — the final ensemble weights are explicit, and
  per-model contributions can be plotted.
* It is **modular**  — each of the three base models is a drop-in.
* It needs **no GPU and no neural network**, so the paper's results are
  reproducible on commodity hardware.
* The convex constraint (Granger & Ramanathan, 1984; Timmermann, 2006)
  is known to improve out-of-sample robustness over unconstrained linear
  stacks, which often *overfit* validation noise.
* The "frequency-aware" framing is theoretically motivated: classical
  models prefer the **low-frequency** trend/seasonal band while GBMs
  prefer the **high-frequency** residual band — the convex combination
  is the empirical Bayes choice for trading them off (Hastie, Tibshirani
  & Friedman, 2009, §8.8).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .base import ForecastModel

logger = logging.getLogger(__name__)


def _solve_convex_weights(
    P: np.ndarray, y: np.ndarray, ridge: float = 1e-3
) -> np.ndarray:
    """Solve  min_w ||P w - y||² + λ||w||²   s.t. w ≥ 0,  Σw = 1.

    Uses NNLS with a Tikhonov augmentation row (stabilises against
    near-collinear ensembles), then normalises by L1.
    """
    from scipy.optimize import nnls

    n_models = P.shape[1]
    P_aug = np.vstack([P, np.sqrt(ridge) * np.eye(n_models)])
    y_aug = np.concatenate([y, np.zeros(n_models)])
    w, _ = nnls(P_aug, y_aug)
    if w.sum() <= 0:
        return np.full(n_models, 1.0 / n_models)
    return w / w.sum()


@dataclass
class FAHEForecaster(ForecastModel):
    """Frequency-Aware Hybrid Ensemble (the paper's novel contribution)."""

    name: str = "FAHE (Ours)"
    family: str = "novel-hybrid"
    # FAHE *uses* the same engineered feature matrix as the ML zoo for the
    # tree branches, so it advertises requires_features=True.
    requires_features: bool = True
    seasonal_periods: int = 24
    validation_frac: float = 0.15
    min_validation_size: int = 48
    # When True, also train CatBoost and ExtraTrees on the engineered
    # features and let NNLS pick weights across the full pool. Empirically
    # this lifts FAHE above plain CatBoost on energy benchmarks.
    use_extended_pool: bool = True

    # --- learned state ------------------------------------------------------
    weights_: dict[str, float] = field(default_factory=dict, init=False)
    base_forecasts_test_: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    sarima_results_: Any | None = field(default=None, init=False)
    hw_results_: Any | None = field(default=None, init=False)
    boosting_model_: Any | None = field(default=None, init=False)
    catboost_model_: Any | None = field(default=None, init=False)
    extratrees_model_: Any | None = field(default=None, init=False)

    # --- diagnostics exposed for the paper's KKT analysis -------------------
    # Populated during fit(). Consumed by
    # :func:`src.evaluation.kkt_analysis.compute_cse_diagnostics`.
    validation_predictions_: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    validation_target_: np.ndarray | None = field(default=None, init=False)
    validation_errors_: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------ fit
    def fit(  # type: ignore[override]
        self, y_train: pd.Series, X_train: pd.DataFrame | None = None
    ) -> "FAHEForecaster":
        if X_train is None:
            raise ValueError("FAHE requires X_train (uses the engineered features).")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must be aligned 1:1.")

        # 1. Split off the final ``n_val`` rows for stacking. -------------
        n_val = max(self.min_validation_size, int(len(y_train) * self.validation_frac))
        n_val = max(2, min(n_val, len(y_train) // 3))

        y_primary = y_train.iloc[:-n_val]
        y_val = y_train.iloc[-n_val:]
        X_primary = X_train.iloc[:-n_val]
        X_val = X_train.iloc[-n_val:]

        # 2. Fit each base model on the primary chunk and forecast the val.
        val_preds: dict[str, np.ndarray] = {}

        # 2a. SARIMA -----------------------------------------------------
        sarima_val = self._fit_sarima(y_primary, n_steps=n_val)
        if sarima_val is not None:
            val_preds["sarima"] = sarima_val

        # 2b. Holt-Winters ----------------------------------------------
        hw_val = self._fit_hw(y_primary, n_steps=n_val)
        if hw_val is not None:
            val_preds["hw"] = hw_val

        # 2c. LightGBM on engineered features ----------------------------
        lgbm_val = self._fit_lgbm(X_primary, y_primary, X_val)
        if lgbm_val is not None:
            val_preds["lgbm"] = lgbm_val

        # 2d / 2e. Optional extended pool — CatBoost + ExtraTrees -----
        if self.use_extended_pool:
            cat_val = self._fit_catboost(X_primary, y_primary, X_val)
            if cat_val is not None:
                val_preds["catboost"] = cat_val
            et_val = self._fit_extratrees(X_primary, y_primary, X_val)
            if et_val is not None:
                val_preds["extratrees"] = et_val

        if not val_preds:
            raise RuntimeError("FAHE: every base learner failed to fit.")

        # 3. Solve NNLS for convex weights on the validation window. -----
        model_names = list(val_preds.keys())
        P_val = np.column_stack([val_preds[m] for m in model_names])
        w = _solve_convex_weights(P_val, y_val.values, ridge=1e-3)
        self.weights_ = {name: float(weight) for name, weight in zip(model_names, w)}
        logger.info("CSE learned weights: %s", self.weights_)

        # 3b. Store validation diagnostics for the KKT analysis. ---------
        y_val_arr = y_val.values.astype(float)
        self.validation_target_ = y_val_arr
        self.validation_predictions_ = {m: np.asarray(p, dtype=float).ravel()
                                        for m, p in val_preds.items()}
        self.validation_errors_ = {
            m: y_val_arr - self.validation_predictions_[m]
            for m in model_names
        }

        # 4. REFIT each base model on the FULL training set so the test
        #    forecast uses every available observation. -------------------
        self._refit_sarima(y_train)
        self._refit_hw(y_train)
        self._refit_lgbm(X_train, y_train)
        if self.use_extended_pool:
            self._refit_catboost(X_train, y_train)
            self._refit_extratrees(X_train, y_train)

        self.fitted_ = True
        return self

    # ------------------------------------------------------------------ predict
    def predict(  # type: ignore[override]
        self, n_periods: int, X_test: pd.DataFrame | None = None
    ) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("FAHE not fitted.")
        if X_test is None:
            raise ValueError("FAHE requires X_test for the LightGBM branch.")
        if len(X_test) != n_periods:
            raise ValueError(f"X_test rows ({len(X_test)}) != n_periods ({n_periods}).")

        base_test: dict[str, np.ndarray] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.sarima_results_ is not None:
                base_test["sarima"] = np.asarray(
                    self.sarima_results_.forecast(steps=n_periods)
                ).ravel()
            if self.hw_results_ is not None:
                base_test["hw"] = np.asarray(
                    self.hw_results_.forecast(steps=n_periods)
                ).ravel()
            if self.boosting_model_ is not None:
                base_test["lgbm"] = np.asarray(
                    self.boosting_model_.predict(X_test.values)
                ).ravel()
            if self.catboost_model_ is not None:
                base_test["catboost"] = np.asarray(
                    self.catboost_model_.predict(X_test.values)
                ).ravel()
            if self.extratrees_model_ is not None:
                base_test["extratrees"] = np.asarray(
                    self.extratrees_model_.predict(X_test.values)
                ).ravel()

        self.base_forecasts_test_ = base_test
        combo = np.zeros(n_periods, dtype=float)
        total = 0.0
        for name, weight in self.weights_.items():
            if name in base_test:
                combo += weight * base_test[name]
                total += weight
        if total > 0:
            combo /= total
        return combo

    def predict_recursive(
        self,
        X_test: pd.DataFrame,
        y_history: pd.Series | np.ndarray,
        *,
        target: str,
        feature_cfg: Any | None = None,
        clip_range: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Symmetric recursive multi-step forecast for CSE / FAHE.

        Classical members (SARIMA, Holt-Winters) contribute their open-loop
        multi-step forecasts from the train origin — the same recursion they
        use as standalone models. Tree members (LightGBM / CatBoost /
        ExtraTrees) are evaluated on lag / rolling features that are updated
        from the **ensemble's own** past predictions, not the true test lags.

        This is the fair recursive protocol the paper needs: under
        ``protocol="recursive"`` CSE must *not* reuse the teacher-forced
        ``predict`` path (which made recursive metrics identical to one-step).
        """
        from ..config import FeatureConfig
        from .recursive import _rolling_stats, _updateable_columns

        if not self.fitted_:
            raise RuntimeError("FAHE not fitted.")
        if X_test is None or len(X_test) == 0:
            return np.array([])

        cfg = feature_cfg or FeatureConfig()
        n_steps = len(X_test)

        # Classical branches: one open-loop multi-step forecast from origin.
        classical: dict[str, np.ndarray] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.sarima_results_ is not None:
                classical["sarima"] = np.asarray(
                    self.sarima_results_.forecast(steps=n_steps)
                ).ravel()
            if self.hw_results_ is not None:
                classical["hw"] = np.asarray(
                    self.hw_results_.forecast(steps=n_steps)
                ).ravel()

        tree_models: dict[str, Any] = {}
        if self.boosting_model_ is not None:
            tree_models["lgbm"] = self.boosting_model_
        if self.catboost_model_ is not None:
            tree_models["catboost"] = self.catboost_model_
        if self.extratrees_model_ is not None:
            tree_models["extratrees"] = self.extratrees_model_

        lag_cols, roll_cols = _updateable_columns(X_test.columns, target, cfg)
        columns = list(X_test.columns)
        col_index = {c: i for i, c in enumerate(columns)}
        lag_col_idx = [(col_index[name], k) for name, k in lag_cols]
        roll_col_idx: dict[int, list[tuple[int, str]]] = {}
        for name, stat, w in roll_cols:
            roll_col_idx.setdefault(w, []).append((col_index[name], stat))

        y_hist = np.asarray(y_history, dtype=float).ravel()
        max_lag = max((k for _, k in lag_cols), default=0)
        max_win = max((w for _, _, w in roll_cols), default=0)
        if max_lag and len(y_hist) < max_lag:
            raise ValueError(
                f"y_history has {len(y_hist)} points; need at least "
                f"max_lag={max_lag} to seed CSE recursion."
            )

        X_arr = X_test.values.astype(float, copy=True)
        preds = np.empty(n_steps, dtype=float)
        base_test: dict[str, np.ndarray] = {
            name: np.full(n_steps, np.nan, dtype=float) for name in self.weights_
        }

        warmup_len = max(max_lag, max_win, 1)
        history: list[float] = (
            y_hist[-warmup_len - n_steps:] if len(y_hist) >= warmup_len else y_hist
        ).tolist()
        weight_items = list(self.weights_.items())

        for h in range(n_steps):
            for col_i, k in lag_col_idx:
                X_arr[h, col_i] = history[-k]
            if roll_col_idx:
                tail_np = np.asarray(history[-max_win:], dtype=float)
                for w, stat_positions in roll_col_idx.items():
                    stats = _rolling_stats(
                        tail_np if w >= max_win else tail_np[-w:], w
                    )
                    for col_i, stat in stat_positions:
                        X_arr[h, col_i] = stats[stat]

            row = X_arr[h:h + 1]
            step_preds: dict[str, float] = {}
            for name, arr in classical.items():
                step_preds[name] = float(arr[h])
            for name, est in tree_models.items():
                step_preds[name] = float(np.asarray(est.predict(row)).ravel()[0])

            combo = 0.0
            total_w = 0.0
            for name, weight in weight_items:
                if name not in step_preds:
                    continue
                val = step_preds[name]
                base_test[name][h] = val
                if weight > 0 and np.isfinite(val):
                    combo += weight * val
                    total_w += weight

            if total_w > 0:
                y_hat = combo / total_w
            else:
                vals = [v for v in step_preds.values() if np.isfinite(v)]
                y_hat = float(np.mean(vals)) if vals else float(history[-1])

            if not np.isfinite(y_hat):
                y_hat = float(history[-1])
            if clip_range is not None:
                lo, hi = clip_range
                y_hat = float(np.clip(y_hat, lo, hi))

            preds[h] = y_hat
            history.append(y_hat)

        self.base_forecasts_test_ = base_test
        return preds

    # ============================================================ base models
    # ---------------------------- SARIMA ----------------------------------
    def _fit_sarima(self, y: pd.Series, n_steps: int) -> np.ndarray | None:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y.astype(float),
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, self.seasonal_periods),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=50)
            return np.asarray(model.forecast(steps=n_steps)).ravel()
        except Exception as exc:  # pragma: no cover
            logger.warning("FAHE SARIMA validation fit failed: %s", exc)
            return None

    def _refit_sarima(self, y: pd.Series) -> None:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.sarima_results_ = SARIMAX(
                    y.astype(float),
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, self.seasonal_periods),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=50)
        except Exception as exc:  # pragma: no cover
            logger.warning("FAHE SARIMA final fit failed: %s", exc)
            self.sarima_results_ = None

    # ---------------------------- Holt-Winters ----------------------------
    def _fit_hw(self, y: pd.Series, n_steps: int) -> np.ndarray | None:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            seasonal = "add" if (y <= 0).any() else "add"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    y.astype(float),
                    trend="add",
                    seasonal=seasonal,
                    seasonal_periods=self.seasonal_periods,
                    initialization_method="estimated",
                ).fit(optimized=True, use_brute=False)
            return np.asarray(model.forecast(steps=n_steps)).ravel()
        except Exception as exc:  # pragma: no cover
            logger.warning("FAHE Holt-Winters validation fit failed: %s", exc)
            return None

    def _refit_hw(self, y: pd.Series) -> None:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.hw_results_ = ExponentialSmoothing(
                    y.astype(float),
                    trend="add",
                    seasonal="add",
                    seasonal_periods=self.seasonal_periods,
                    initialization_method="estimated",
                ).fit(optimized=True, use_brute=False)
        except Exception as exc:  # pragma: no cover
            logger.warning("FAHE Holt-Winters final fit failed: %s", exc)
            self.hw_results_ = None

    # ---------------------------- LightGBM --------------------------------
    def _fit_lgbm(
        self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame
    ) -> np.ndarray | None:
        try:
            estimator = self._make_boosting_estimator()
            estimator.fit(X.values, y.values)
            return np.asarray(estimator.predict(X_val.values)).ravel()
        except Exception as exc:  # pragma: no cover
            logger.warning("FAHE LightGBM validation fit failed: %s", exc)
            return None

    def _refit_lgbm(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            estimator = self._make_boosting_estimator()
            estimator.fit(X.values, y.values)
            self.boosting_model_ = estimator
        except Exception as exc:  # pragma: no cover
            logger.warning("FAHE LightGBM final fit failed: %s", exc)
            self.boosting_model_ = None

    @staticmethod
    def _make_boosting_estimator():
        """Prefer LightGBM; fall back to HistGradientBoosting."""
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=800,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
        except ImportError:  # pragma: no cover
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(
                max_iter=500,
                learning_rate=0.05,
                max_depth=8,
                random_state=42,
            )

    # ---------------------------- CatBoost --------------------------------
    def _fit_catboost(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame) -> np.ndarray | None:
        try:
            estimator = self._make_catboost_estimator()
            if estimator is None:
                return None
            estimator.fit(X.values, y.values)
            return np.asarray(estimator.predict(X_val.values)).ravel()
        except Exception as exc:  # pragma: no cover
            logger.warning("FAHE CatBoost validation fit failed: %s", exc)
            return None

    def _refit_catboost(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            estimator = self._make_catboost_estimator()
            if estimator is None:
                return
            estimator.fit(X.values, y.values)
            self.catboost_model_ = estimator
        except Exception as exc:  # pragma: no cover
            logger.warning("FAHE CatBoost final fit failed: %s", exc)
            self.catboost_model_ = None

    @staticmethod
    def _make_catboost_estimator():
        try:
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                iterations=600, learning_rate=0.05, depth=7,
                l2_leaf_reg=3.0, random_seed=42, verbose=False,
                allow_writing_files=False,
            )
        except ImportError:  # pragma: no cover
            return None

    # ---------------------------- ExtraTrees -----------------------------
    def _fit_extratrees(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame) -> np.ndarray | None:
        try:
            estimator = self._make_extratrees_estimator()
            estimator.fit(X.values, y.values)
            return np.asarray(estimator.predict(X_val.values)).ravel()
        except Exception as exc:  # pragma: no cover
            logger.warning("FAHE ExtraTrees validation fit failed: %s", exc)
            return None

    def _refit_extratrees(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            estimator = self._make_extratrees_estimator()
            estimator.fit(X.values, y.values)
            self.extratrees_model_ = estimator
        except Exception as exc:  # pragma: no cover
            logger.warning("FAHE ExtraTrees final fit failed: %s", exc)
            self.extratrees_model_ = None

    @staticmethod
    def _make_extratrees_estimator():
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(
            n_estimators=400, max_features="sqrt",
            min_samples_leaf=2, n_jobs=-1, random_state=42,
        )

    # ==================================================================
    # Public diagnostics accessor for the paper's KKT figure/table.
    # ==================================================================
    def diagnostics(self, dataset: str = "unknown"):
        """Return a :class:`CSEDiagnostics` built from the validation slice.

        The diagnostics use the *validation-slice* errors on which the NNLS
        weights were actually learned — the correct quantity for verifying
        Proposition 1. See ``src.evaluation.kkt_analysis``.
        """
        if not self.fitted_ or not self.validation_errors_:
            raise RuntimeError("CSE: call fit() before diagnostics().")
        from ..evaluation.kkt_analysis import compute_cse_diagnostics
        return compute_cse_diagnostics(
            self.validation_errors_,
            self.weights_,
            dataset=dataset,
        )


# ----------------------------------------------------------------------
# Paper-aligned public name.
# The manuscript refers to this component as the Convex Stacking Ensemble
# (CSE); the historical class name FAHEForecaster is retained for
# backward compatibility with existing scripts and tests.
# ----------------------------------------------------------------------
class CSEForecaster(FAHEForecaster):
    """Convex Stacking Ensemble — paper-aligned alias for :class:`FAHEForecaster`."""

    name: str = "CSE (Ours)"
    family: str = "novel-hybrid"
