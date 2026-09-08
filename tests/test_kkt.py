"""Numerical sanity tests for the KKT analysis module (paper's Prop. 1)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.kkt_analysis import (
    compute_cse_diagnostics, kkt_conditions, proposition1_frontier,
)


def test_proposition1_frontier_maps_to_reciprocal():
    ratios = np.array([1.0, 1.5, 2.0, 4.0])
    curve = proposition1_frontier(ratios)
    assert np.allclose(curve, 1.0 / ratios)


def test_diagnostics_zero_weight_matches_kkt():
    """Two-member case matching Prop. 1 explicitly."""
    rng = np.random.default_rng(0)
    n = 500
    # Best member: small variance.
    e_best = rng.normal(0, 0.2, size=n)
    # Second member: 5x larger variance, strongly correlated errors (ρ ≈ 0.8)
    # ⇒ Prop. 1 predicts zero weight because ρ (0.8) > σ_best/σ_bad (0.2).
    e_bad = 0.8 * (e_best / 0.2) * 1.0 + np.sqrt(1 - 0.8 ** 2) * rng.normal(0, 1, size=n)
    errors = {"best": e_best, "bad": e_bad}
    # Any weights on the simplex with all mass on 'best' satisfy the KKT
    # conditions in this construction.
    diag = compute_cse_diagnostics(errors, {"best": 1.0, "bad": 0.0},
                                  dataset="synthetic")
    assert diag.member_names == ["best", "bad"]
    assert diag.best_member_idx == 0
    kkt = kkt_conditions(diag)
    # Active member (best): equality — residual ~ 0.
    assert abs(kkt.loc[0, "kkt_residual"]) < 1e-6
    # Inactive member (bad): inequality — residual ≥ 0.
    assert kkt.loc[1, "kkt_residual"] >= -1e-6


def test_diagnostics_weights_normalise():
    rng = np.random.default_rng(0)
    n = 400
    e = {n_name: rng.normal(0, s, size=n)
         for n_name, s in [("m1", 0.3), ("m2", 0.5), ("m3", 0.8)]}
    diag = compute_cse_diagnostics(e, {"m1": 2.0, "m2": 3.0, "m3": 5.0})
    assert abs(diag.weights.sum() - 1.0) < 1e-9
    assert diag.rho.shape == (3, 3)
    assert np.allclose(np.diag(diag.rho), 1.0)


def test_to_frame_columns():
    rng = np.random.default_rng(0)
    n = 200
    e = {"a": rng.normal(0, 0.4, size=n), "b": rng.normal(0, 0.9, size=n)}
    diag = compute_cse_diagnostics(e, {"a": 0.7, "b": 0.3}, dataset="D1")
    df = diag.to_frame()
    assert set(["dataset", "member", "weight", "sigma", "sigma_ratio_vs_best",
                "rho_vs_best", "kkt_residual", "is_best", "is_active"]).issubset(df.columns)
    assert df.loc[df["is_best"], "sigma_ratio_vs_best"].iloc[0] == 1.0
