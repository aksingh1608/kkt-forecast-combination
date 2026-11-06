"""Tests for the generic simplex stacker and the refit-only-active option."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FeatureConfig
from src.features.engineering import build_supervised_frame
from src.models.classical import NaiveForecaster, SeasonalNaiveForecaster
from src.models.ml import build_ml_zoo
from src.models.registry import ModelSpec
from src.models.stacking import ConvexStacker


def _series(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 10 + 3 * np.sin(2 * np.pi * t / 24) + 0.8 * np.sin(2 * np.pi * t / 168) + rng.normal(0, 0.5, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"y": y}, index=idx)


def _specs():
    zoo = {m.name: m for m in build_ml_zoo(random_state=0, n_jobs=1)}
    ridge = zoo["Ridge"]
    return [
        ModelSpec("Naive", "classical", lambda: NaiveForecaster(), False),
        ModelSpec("SeasonalNaive", "classical", lambda: SeasonalNaiveForecaster(season_length=24), False),
        ModelSpec("Ridge", "linear", lambda r=ridge: type(r)(**{k: v for k, v in vars(r).items() if not k.endswith("_")}), True),
    ]


def test_convex_stacker_weights_on_simplex_and_kkt():
    df = _series()
    X, y, _ = build_supervised_frame(df, "y", FeatureConfig())
    n = len(X)
    cut = int(n * 0.8)
    st = ConvexStacker(_specs()).fit(y.iloc[:cut], X.iloc[:cut])
    w = np.array(list(st.weights_.values()))
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()
    diag = st.diagnostics("synthetic")
    lam = diag.kkt_lambda
    for i, name in enumerate(diag.member_names):
        res = diag.kkt_residuals[i] / lam
        if diag.weights[i] > 1e-9:
            assert abs(res) < 1e-8
        else:
            assert res > -1e-8
    pred = st.predict(n - cut, X.iloc[cut:])
    assert pred.shape == (n - cut,) and np.all(np.isfinite(pred))
    # members with zero weight were not refit
    for name, wt in st.weights_.items():
        assert (name in st.models_) == (wt > 1e-9)


def test_convex_stacker_blend_not_worse_than_best_member_on_validation():
    df = _series(seed=3)
    X, y, _ = build_supervised_frame(df, "y", FeatureConfig())
    cut = int(len(X) * 0.8)
    st = ConvexStacker(_specs(), refit_only_active=False).fit(y.iloc[:cut], X.iloc[:cut])
    E = np.column_stack([st.validation_errors_[n] for n in st.validation_errors_])
    w = np.array([st.weights_[n] for n in st.validation_errors_])
    blend = np.mean((E @ w) ** 2)
    assert blend <= np.min(np.mean(E ** 2, axis=0)) + 1e-12
