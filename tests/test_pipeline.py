"""Smoke-level tests for the public API.

These tests are deliberately lightweight — they exercise every code path
on a tiny synthetic dataset so they finish in a few seconds and remain
useful in pre-commit checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import clean_dataset, train_test_split_temporal
from src.evaluation.metrics import compute_all_metrics
from src.features.engineering import build_supervised_frame
from src.models import FAHEForecaster
from src.pipeline import run_experiment


def _make_synth_series(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.arange(n)
    y = (
        10.0
        + 0.001 * t                     # mild trend
        + 3.0 * np.sin(2 * np.pi * t / 24)   # daily season
        + 0.8 * np.sin(2 * np.pi * t / 168)  # weekly season
        + rng.normal(0, 0.5, size=n)         # noise
    )
    return pd.DataFrame({"y": y}, index=idx)


def test_pipeline_runs_end_to_end():
    df = _make_synth_series()
    df = clean_dataset(df)
    result = run_experiment(
        df, target="y",
        seasonal_period=24,
        test_size=0.2,
        include_heavy_classical=False,
    )
    lb = result.leaderboard()
    assert not lb.empty
    # At least 80 % of registered models should succeed.
    success_rate = lb["MAE"].notna().mean()
    assert success_rate > 0.8


def test_metrics_basic_properties():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    p = y + rng.normal(scale=0.1, size=200)
    out = compute_all_metrics(y, p, y_train=y, seasonality=1)
    assert out["MAE"] >= 0
    assert out["RMSE"] >= out["MAE"] - 1e-9     # by Jensen's inequality
    assert -1 <= out["R2"] <= 1 + 1e-9


def test_train_test_split_temporal_is_chronological():
    df = _make_synth_series(n=500)
    splits = train_test_split_temporal(df, test_size=0.2)
    assert splits["train"].index.max() < splits["test"].index.min()


def test_fahe_learns_weights():
    df = _make_synth_series(n=1000)
    df = clean_dataset(df)
    X, y, _ = build_supervised_frame(df, "y")
    splits_X = train_test_split_temporal(X, test_size=0.2)
    splits_y = train_test_split_temporal(y.to_frame(), test_size=0.2)
    fahe = FAHEForecaster(seasonal_periods=24, validation_frac=0.2)
    fahe.fit(splits_y["train"]["y"], X_train=splits_X["train"])
    assert fahe.weights_   # populated
    pred = fahe.predict(len(splits_X["test"]), X_test=splits_X["test"])
    assert pred.shape == (len(splits_X["test"]),)
    assert np.all(np.isfinite(pred))
