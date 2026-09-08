"""Tests for the Q1-paper-grade extensions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import clean_dataset
from src.eda.stationarity import stationarity_report
from src.evaluation import (
    adaptive_conformal_intervals, bootstrap_metric_ci, evaluate_probabilistic,
    missing_injection_curve, noise_injection_curve, per_horizon_metrics,
    split_conformal_intervals,
)
from src.evaluation.probabilistic import pinball_loss, crps_from_quantiles
from src.features import add_holiday_features
from src.features.engineering import build_supervised_frame
from src.models import (
    DLinearForecaster, NLinearForecaster, NaiveDriftForecaster,
    QuantileLightGBMForecaster, ThetaForecaster,
)


def _make_series(seed: int = 0, n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.arange(n)
    y = (
        10.0 + 0.001 * t
        + 3.0 * np.sin(2 * np.pi * t / 24)
        + 0.6 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, 0.4, n)
    )
    return pd.DataFrame({"y": y}, index=idx)


# ---------------------------------------------------------------------------
# Stationarity, holidays
# ---------------------------------------------------------------------------
def test_stationarity_report_returns_finite_values():
    s = _make_series(n=600)["y"]
    rep = stationarity_report(s)
    assert np.isfinite(rep.adf_stat)
    assert 0 <= rep.adf_pvalue <= 1
    assert 0 <= rep.kpss_pvalue <= 1
    assert isinstance(rep.verdict, str) and rep.verdict


def test_holiday_features_added():
    df = _make_series(n=400)
    out = add_holiday_features(df, country="FR")
    for col in ("is_holiday", "is_holiday_eve", "days_to_holiday"):
        assert col in out.columns
    assert out["is_holiday"].isin([0, 1]).all()


# ---------------------------------------------------------------------------
# Lite-DL baselines
# ---------------------------------------------------------------------------
def test_lite_dl_baselines_train_and_predict():
    df = clean_dataset(_make_series())
    n = len(df); n_test = max(48, int(n * 0.2))
    y_tr, y_te = df["y"].iloc[:-n_test], df["y"].iloc[-n_test:]
    for cls in (NaiveDriftForecaster, ThetaForecaster, NLinearForecaster, DLinearForecaster):
        m = cls()
        if cls is NLinearForecaster or cls is DLinearForecaster:
            m = cls(lookback=72)
        m.fit(y_tr)
        preds = m.predict(len(y_te))
        assert preds.shape == (len(y_te),)
        assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# Probabilistic + conformal
# ---------------------------------------------------------------------------
def test_quantile_lightgbm_and_probabilistic_eval():
    df = clean_dataset(_make_series(n=800))
    X, y, _ = build_supervised_frame(df, "y")
    n = len(X); n_test = max(40, int(n * 0.2))
    Xtr, ytr = X.iloc[:-n_test], y.iloc[:-n_test]
    Xte, yte = X.iloc[-n_test:], y.iloc[-n_test:]
    qm = QuantileLightGBMForecaster(quantiles=(0.1, 0.5, 0.9), n_estimators=120)
    qm.fit(ytr, X_train=Xtr)
    qpred = qm.predict_quantiles(len(Xte), X_test=Xte)
    assert set(qpred.keys()) == {0.1, 0.5, 0.9}
    # Quantile non-crossing (after our internal sort).
    assert np.all(qpred[0.1] <= qpred[0.5] + 1e-6)
    assert np.all(qpred[0.5] <= qpred[0.9] + 1e-6)

    rep = evaluate_probabilistic(yte.values, qpred, intervals=((0.1, 0.9),))
    assert np.isfinite(rep.crps)
    assert np.isfinite(rep.mean_pinball)
    nominal = next(iter(rep.coverage_per_alpha.keys()))
    assert 0.0 <= rep.coverage_per_alpha[nominal] <= 1.0
    assert rep.width_per_alpha[nominal] > 0


def test_pinball_basic():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([0.5, 2.5, 2.5])
    assert pinball_loss(y, p, 0.5) > 0
    # CRPS from a single 0.5 quantile is degenerate (NaN by design).
    assert np.isnan(crps_from_quantiles(y, {0.5: p}))


def test_split_and_adaptive_conformal():
    rng = np.random.default_rng(0)
    cal_resid = rng.normal(0, 1, size=200)
    test_pred = np.linspace(0, 5, 50)
    test_actual = test_pred + rng.normal(0, 1, size=50)
    lo, hi, q = split_conformal_intervals(cal_resid, test_pred, alpha=0.1)
    assert q > 0
    assert (hi >= lo).all()
    cov = float(np.mean((test_actual >= lo) & (test_actual <= hi)))
    assert 0.7 <= cov <= 1.0  # finite-sample coverage near 90%

    lo2, hi2, state = adaptive_conformal_intervals(
        cal_resid, test_pred, test_actual=test_actual, alpha=0.1)
    assert (hi2 >= lo2).all()
    assert len(state.history_alpha) == len(test_pred) + 1


# ---------------------------------------------------------------------------
# Per-horizon, bootstrap CIs, robustness
# ---------------------------------------------------------------------------
def test_per_horizon_breakdown():
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    p = y + rng.normal(scale=0.5, size=300)
    out = per_horizon_metrics(y, p, horizons=(1, 6, 24, 100))
    assert set(out["horizon"]) == {1, 6, 24, 100}
    assert (out["RMSE"] >= 0).all()


def test_bootstrap_ci_shape():
    rng = np.random.default_rng(2)
    y = rng.normal(size=400)
    p = y + rng.normal(scale=0.3, size=400)
    out = bootstrap_metric_ci(y, p, metric="RMSE", n_bootstrap=120, block_size=12)
    assert out["lo"] <= out["point"] <= out["hi"]
    assert out["metric"] == "RMSE"


def test_robustness_curves_run():
    rng = np.random.default_rng(3)
    y = rng.normal(size=300)
    p = y + rng.normal(scale=0.3, size=300)
    nz = noise_injection_curve(y, p, sigmas=(0.0, 0.1, 0.5), n_repeats=2)
    miss = missing_injection_curve(y, p, fractions=(0.0, 0.1, 0.5), n_repeats=2)
    for d in (nz, miss):
        assert "RMSE" in d.columns
        # Deterioration is monotone (or near-monotone) for these toy series.
        assert d["RMSE"].iloc[0] <= d["RMSE"].iloc[-1] + 1e-6
