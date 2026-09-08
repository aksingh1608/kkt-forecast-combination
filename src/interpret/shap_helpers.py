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

"""SHAP utilities for the tree-based forecasters."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _unwrap(estimator: Any) -> Any:
    """Reach the underlying tree estimator inside an sklearn Pipeline."""
    if hasattr(estimator, "named_steps"):
        last = list(estimator.named_steps.values())[-1]
        return last
    return estimator


def shap_values_for_tree_model(
    estimator: Any, X: pd.DataFrame, *, max_samples: int = 1000, seed: int = 42
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (shap_values, X_sample) for any tree-based estimator.

    Falls back to ``shap.Explainer`` for non-tree models. The returned
    ``shap_values`` array has shape (n_samples, n_features).
    """
    import shap

    inner = _unwrap(estimator)
    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X_sample = X.iloc[idx]
    else:
        X_sample = X

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            explainer = shap.TreeExplainer(inner)
            sv = explainer.shap_values(X_sample.values)
        except Exception as exc:  # pragma: no cover
            logger.info("TreeExplainer failed (%s); falling back to shap.Explainer.", exc)
            explainer = shap.Explainer(inner.predict, X_sample.values)
            sv = explainer(X_sample.values).values

    sv = np.asarray(sv)
    if sv.ndim == 3 and sv.shape[-1] == 1:
        sv = sv[..., 0]
    return sv, X_sample


def shap_summary(
    estimator: Any,
    X: pd.DataFrame,
    *,
    max_samples: int = 500,
    top_k: int = 15,
) -> pd.DataFrame:
    """Return a tidy 'mean(|SHAP|) per feature' table sorted desc."""
    sv, X_sample = shap_values_for_tree_model(
        estimator, X, max_samples=max_samples
    )
    mean_abs = np.mean(np.abs(sv), axis=0)
    out = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return out.head(top_k)
