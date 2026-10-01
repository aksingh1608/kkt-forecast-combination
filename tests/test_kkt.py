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


# ---------------------------------------------------------------------------
# Exact simplex solver (v1.1): the estimator must solve the problem the
# paper analyses, min w'Sw on the simplex with S = E'E/n (uncentred).
# ---------------------------------------------------------------------------
def _two_member_closed_form(S):
    s11, s22, s12 = S[0, 0], S[1, 1], S[0, 1]
    w2 = (s11 - s12) / (s11 + s22 - 2 * s12)
    w2 = min(max(w2, 0.0), 1.0)
    return np.array([1 - w2, w2])


def test_simplex_solver_matches_two_member_closed_form():
    from src.models.novel import solve_simplex_qp
    rng = np.random.default_rng(1)
    for _ in range(50):
        E = rng.normal(size=(300, 2)) * rng.uniform(0.2, 3.0, size=2)
        E[:, 1] += rng.uniform(-1, 1) * E[:, 0]
        S = E.T @ E / len(E)
        assert np.allclose(solve_simplex_qp(S), _two_member_closed_form(S), atol=1e-9)


def test_simplex_solver_satisfies_full_kkt_and_beats_random_points():
    from src.models.novel import solve_simplex_qp
    rng = np.random.default_rng(2)
    for K in (3, 5, 7):
        A = rng.normal(size=(400, K)) + rng.normal(size=(400, 1))  # shared component
        E = A * rng.uniform(0.5, 4.0, size=K) + rng.normal(0, 0.3, size=K)  # with bias
        S = E.T @ E / len(E)
        w = solve_simplex_qp(S)
        assert abs(w.sum() - 1) < 1e-12 and (w >= 0).all()
        lam = w @ S @ w
        g = S @ w
        assert np.all(g[w > 1e-9] - lam < 1e-8 * lam + 1e-12)
        assert np.all(np.abs(g[w > 1e-9] - lam) < 1e-8 * lam)
        assert np.all(g[w <= 1e-9] - lam > -1e-8 * lam)
        for _ in range(200):
            v = rng.dirichlet(np.ones(K))
            assert v @ S @ v >= lam - 1e-12


def test_solve_convex_weights_is_scale_free_and_ignores_nothing():
    """Weights must not change when the target is rescaled (kW vs MW)."""
    from src.models.novel import _solve_convex_weights
    rng = np.random.default_rng(3)
    y = rng.normal(size=500)
    P = np.column_stack([y + rng.normal(0, s, size=500) + b
                         for s, b in [(0.3, 0.0), (0.4, 0.1), (2.0, 1.0)]])
    w1 = _solve_convex_weights(P, y)
    w2 = _solve_convex_weights(P * 1e6, y * 1e6)
    assert np.allclose(w1, w2, atol=1e-9)
    assert abs(w1.sum() - 1) < 1e-12


def test_diagnostics_use_uncentred_moments_by_default():
    rng = np.random.default_rng(4)
    e = {"a": rng.normal(0, 0.5, size=1000), "b": rng.normal(2.0, 0.5, size=1000)}
    diag = compute_cse_diagnostics(e, {"a": 1.0, "b": 0.0})
    # member b is biased: its uncentred scale is ~sqrt(4 + 0.25), not 0.5
    assert diag.sigma[1] > 1.9
    old = compute_cse_diagnostics(e, {"a": 1.0, "b": 0.0}, centred=True)
    assert old.sigma[1] < 0.6
    # lambda equals the blend's mean squared validation error
    assert np.isclose(diag.kkt_lambda, np.mean(diag.blend_error ** 2))
