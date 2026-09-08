"""Unit tests for the effect-size and multiple-comparison helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.statistical_tests import (
    cohens_d, cliffs_delta, holm_correct, benjamini_hochberg,
    pairwise_dm_matrix, diebold_mariano,
)


def test_cohens_d_signs():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, size=500)
    pred_good = y + rng.normal(0, 0.1, size=500)
    pred_bad = y + rng.normal(0, 1.0, size=500)
    d_good_vs_bad = cohens_d(y, pred_good, pred_bad)
    d_bad_vs_good = cohens_d(y, pred_bad, pred_good)
    assert d_good_vs_bad < 0, "pred_a better than pred_b ⇒ negative d"
    assert d_bad_vs_good > 0
    assert abs(d_good_vs_bad + d_bad_vs_good) < 1e-8


def test_cliffs_delta_bounds():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, size=200)
    pred = y + rng.normal(0, 0.5, size=200)
    delta = cliffs_delta(y, pred, pred)  # identical forecasts
    assert abs(delta) < 1e-8, "identical forecasts ⇒ δ=0"
    delta_range = cliffs_delta(y, pred, np.zeros_like(pred))
    assert -1.0 <= delta_range <= 1.0


def test_holm_monotone_and_bounded():
    p_raw = np.array([0.001, 0.02, 0.03, 0.04, 0.5])
    adj = holm_correct(p_raw)
    assert (adj <= 1.0).all()
    # After Holm, larger raw p ⇒ larger adjusted p (monotone in same order).
    order = np.argsort(p_raw)
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_benjamini_hochberg_monotone():
    p_raw = np.array([0.01, 0.02, 0.03, 0.5, 0.8])
    adj = benjamini_hochberg(p_raw)
    assert (adj <= 1.0).all()
    order = np.argsort(p_raw)
    # BH is monotone non-decreasing in the sorted order.
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_pairwise_dm_matrix_shape_and_columns():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, size=400)
    preds = {
        "good": y + rng.normal(0, 0.2, size=400),
        "medium": y + rng.normal(0, 0.5, size=400),
        "bad": y + rng.normal(0, 1.0, size=400),
    }
    dm = pairwise_dm_matrix(y, preds, correction="holm", h=1)
    assert set(["reference", "model", "dm_stat", "p_raw", "p_adj",
                "significant_5pct"]).issubset(dm.columns)
    assert dm.shape[0] == 2
    assert dm["reference"].nunique() == 1


def test_diebold_mariano_favours_better_forecast():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, size=400)
    good = y + rng.normal(0, 0.1, size=400)
    bad = y + rng.normal(0, 0.7, size=400)
    res = diebold_mariano(y, good, bad, h=1)
    assert res.statistic < 0, "pred_a better ⇒ negative DM statistic"
    assert res.p_value < 0.01
