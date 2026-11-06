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

"""Generic simplex-constrained stacking over any set of registry models.

:class:`FAHEForecaster` hard-codes its five members. This module stacks an
arbitrary list of :class:`~src.models.registry.ModelSpec` objects with the
same recipe (validation tail, exact simplex solve, refit of the active
members) so the sensitivity of the collapse to the member pool can be
measured. It is used by ``scripts/membership_sensitivity.py`` and is scored
under the one-step protocol only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .novel import solve_simplex_qp
from .registry import ModelSpec

logger = logging.getLogger(__name__)


@dataclass
class ConvexStacker:
    """Stack ``specs`` on the simplex using a validation tail of the training set."""

    specs: Sequence[ModelSpec]
    validation_frac: float = 0.15
    min_validation_size: int = 48
    refit_only_active: bool = True
    active_tol: float = 1e-9

    weights_: dict[str, float] = field(default_factory=dict, init=False)
    validation_errors_: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    validation_target_: np.ndarray | None = field(default=None, init=False)
    models_: dict[str, object] = field(default_factory=dict, init=False)
    fit_times_: dict[str, float] = field(default_factory=dict, init=False)
    failed_: list[str] = field(default_factory=list, init=False)
    base_forecasts_test_: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------ fit
    def fit(self, y_train: pd.Series, X_train: pd.DataFrame) -> "ConvexStacker":
        n_val = max(self.min_validation_size, int(len(y_train) * self.validation_frac))
        n_val = max(2, min(n_val, len(y_train) // 3))
        y_primary, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]
        X_primary, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]

        val_preds: dict[str, np.ndarray] = {}
        self.fit_times_ = {}
        self.failed_ = []
        for spec in self.specs:
            t0 = time.perf_counter()
            try:
                model = spec.factory()
                if spec.requires_features:
                    model.fit(y_primary, X_train=X_primary)
                    p = model.predict(n_val, X_test=X_val)
                else:
                    model.fit(y_primary)
                    p = model.predict(n_val)
                p = np.asarray(p, dtype=float).ravel()
                if len(p) != n_val or not np.all(np.isfinite(p)):
                    raise RuntimeError("bad validation forecast")
                val_preds[spec.name] = p
            except Exception as exc:  # pragma: no cover
                logger.warning("stacker: %s failed on the validation fit: %s", spec.name, exc)
                self.failed_.append(spec.name)
            self.fit_times_[f"val:{spec.name}"] = time.perf_counter() - t0
        if not val_preds:
            raise RuntimeError("ConvexStacker: every member failed.")

        names = list(val_preds)
        P = np.column_stack([val_preds[n] for n in names])
        y_v = y_val.to_numpy(dtype=float)
        E = y_v[:, None] - P
        S = (E.T @ E) / len(y_v)
        w = solve_simplex_qp(S)
        self.weights_ = {n: float(x) for n, x in zip(names, w)}
        self.validation_target_ = y_v
        self.validation_errors_ = {n: E[:, i] for i, n in enumerate(names)}

        self.models_ = {}
        for spec in self.specs:
            if spec.name not in names:
                continue
            if self.refit_only_active and self.weights_[spec.name] <= self.active_tol:
                continue
            t0 = time.perf_counter()
            model = spec.factory()
            if spec.requires_features:
                model.fit(y_train, X_train=X_train)
            else:
                model.fit(y_train)
            self.models_[spec.name] = model
            self.fit_times_[f"refit:{spec.name}"] = time.perf_counter() - t0
        return self

    # -------------------------------------------------------------- predict
    def predict(self, n_periods: int, X_test: pd.DataFrame) -> np.ndarray:
        base: dict[str, np.ndarray] = {}
        for spec in self.specs:
            m = self.models_.get(spec.name)
            if m is None:
                continue
            p = m.predict(n_periods, X_test=X_test) if spec.requires_features else m.predict(n_periods)
            base[spec.name] = np.asarray(p, dtype=float).ravel()
        self.base_forecasts_test_ = base
        combo = np.zeros(n_periods)
        total = 0.0
        for n, w in self.weights_.items():
            if w > 0 and n in base:
                combo += w * base[n]
                total += w
        return combo / total if total > 0 else combo

    def diagnostics(self, dataset: str = "unknown"):
        from ..evaluation.kkt_analysis import compute_cse_diagnostics
        return compute_cse_diagnostics(self.validation_errors_, self.weights_, dataset=dataset)


__all__ = ["ConvexStacker"]
