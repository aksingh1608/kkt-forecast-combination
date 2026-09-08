"""Correctness tests for the symmetric recursive multi-step protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FeatureConfig
from src.data.preprocessing import train_test_split_temporal
from src.features.engineering import build_supervised_frame
from src.models.recursive import recursive_predict
from src.models.base import SupervisedForecaster
from sklearn.linear_model import Ridge


def _make_synth_series(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.arange(n)
    y = (
        5.0
        + 2.0 * np.sin(2 * np.pi * t / 24)
        + rng.normal(0, 0.3, size=n)
    )
    return pd.DataFrame({"y": y}, index=idx)


def test_recursive_predict_matches_length_and_is_finite():
    df = _make_synth_series()
    cfg = FeatureConfig(lags=(1, 2, 24), rolling_windows=(3, 24),
                       fourier_orders=((24, 2),), add_calendar=True)
    X, y, _ = build_supervised_frame(df, "y", cfg=cfg)
    split_X = train_test_split_temporal(X, test_size=0.2)
    split_y = train_test_split_temporal(y.to_frame(), test_size=0.2)
    X_train, X_test = split_X["train"], split_X["test"]
    y_train = split_y["train"]["y"]

    model = SupervisedForecaster(
        name="Ridge", family="linear", estimator=Ridge(alpha=1.0)
    )
    model.fit(y_train, X_train=X_train)

    preds = recursive_predict(
        model, X_test, y_train.values, target="y", feature_cfg=cfg
    )
    assert len(preds) == len(X_test)
    assert np.all(np.isfinite(preds))
    assert preds.min() > -50 and preds.max() < 50   # sanity


def test_recursive_differs_from_teacher_forced():
    """The recursive rollout should differ from the one-step prediction because
    lag features are replaced by the model's own past outputs."""
    df = _make_synth_series()
    cfg = FeatureConfig(lags=(1, 2, 3, 24), rolling_windows=(3, 6),
                       fourier_orders=((24, 2),), add_calendar=True)
    X, y, _ = build_supervised_frame(df, "y", cfg=cfg)
    split_X = train_test_split_temporal(X, test_size=0.2)
    split_y = train_test_split_temporal(y.to_frame(), test_size=0.2)
    X_train, X_test = split_X["train"], split_X["test"]
    y_train = split_y["train"]["y"]

    model = SupervisedForecaster(
        name="Ridge", family="linear", estimator=Ridge(alpha=1.0)
    )
    model.fit(y_train, X_train=X_train)

    one_step = model.predict(len(X_test), X_test=X_test)
    recursive = recursive_predict(
        model, X_test, y_train.values, target="y", feature_cfg=cfg
    )
    # At step 0, both regimes see the true lag y_{t-1} = last training value
    # so predictions are identical for the very first step; from step 1 they
    # must diverge because recursive uses its own step-0 prediction as lag_1.
    assert abs(one_step[0] - recursive[0]) < 1e-6
    diff = np.abs(one_step[1:] - recursive[1:]).mean()
    assert diff > 1e-3, "recursive rollout should diverge from teacher-forced after step 1"


def test_recursive_predict_needs_enough_history():
    import pytest
    df = _make_synth_series(n=400)
    cfg = FeatureConfig(lags=(1, 2, 3, 168), rolling_windows=(3,),
                       fourier_orders=(), add_calendar=False)
    X, y, _ = build_supervised_frame(df, "y", cfg=cfg)
    split_X = train_test_split_temporal(X, test_size=0.2)
    X_test = split_X["test"]

    class _Dummy:
        def predict(self, n, X_test=None):
            return np.zeros(n)

    with pytest.raises(ValueError):
        recursive_predict(_Dummy(), X_test, y_history=np.zeros(10),
                          target="y", feature_cfg=cfg)


def test_cse_recursive_differs_from_one_step():
    """CSE under recursion must not reuse teacher-forced tree features."""
    from src.models.novel import FAHEForecaster

    df = _make_synth_series(n=600)
    cfg = FeatureConfig(lags=(1, 2, 3, 24), rolling_windows=(3, 6),
                       fourier_orders=((24, 2),), add_calendar=True)
    X, y, _ = build_supervised_frame(df, "y", cfg=cfg)
    split_X = train_test_split_temporal(X, test_size=0.2)
    split_y = train_test_split_temporal(y.to_frame(), test_size=0.2)
    X_train, X_test = split_X["train"], split_X["test"]
    y_train = split_y["train"]["y"]

    # Keep the pool small for a fast unit test.
    cse = FAHEForecaster(
        seasonal_periods=24,
        use_extended_pool=False,
        validation_frac=0.2,
        min_validation_size=48,
    )
    cse.fit(y_train, X_train=X_train)

    one_step = cse.predict(len(X_test), X_test=X_test)
    recursive = cse.predict_recursive(
        X_test, y_train.values, target="y", feature_cfg=cfg
    )
    assert len(recursive) == len(X_test)
    assert np.all(np.isfinite(recursive))
    # First step shares the true lag seed; later steps must diverge once
    # tree members carry positive weight (typical on this synthetic series).
    tree_w = sum(cse.weights_.get(k, 0.0) for k in ("lgbm", "catboost", "extratrees"))
    if tree_w > 0.05:
        diff = np.abs(one_step[1:] - recursive[1:]).mean()
        assert diff > 1e-4, (
            f"CSE recursive should diverge from one-step when trees have "
            f"weight (tree_w={tree_w:.3f}, mean|Δ|={diff:.2e})"
        )
