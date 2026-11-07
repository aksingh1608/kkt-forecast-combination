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

"""KKT diagnostics for the simplex-constrained stacking problem.

The paper (Section V) proves that at the optimum of

    min_{w ∈ Δ^{K-1}}  w' Σ w,   Σ_ij = <e_i, e_j> / n

every member with strictly positive weight satisfies ``(Σ w)_i = λ`` and
every member with zero weight satisfies ``(Σ w)_i ≥ λ`` (Karush-Kuhn-Tucker
conditions). Proposition 1 specialises this to two members and yields the
closed-form zero-weight condition

    ρ_{12} ≥ σ_1 / σ_2,   (σ_1 ≤ σ_2)

i.e. the weaker member gets zero weight whenever its error correlates
with the stronger member's error above the ratio of their standard
deviations.

This module operationalises those results as first-class artefacts:

* :class:`CSEDiagnostics` — the per-base-learner ``σ_i``, ``ρ_ij`` and the
  learned simplex weights, computed on the *validation* slice (the correct
  quantity — the paper's earlier draft mistakenly used test-slice values).
* :func:`kkt_conditions` — evaluates the KKT residuals ``(Σ w)_i - λ``
  and their signs for every member, so the paper can *report* the
  numerical verification of the theorem.
* :func:`proposition1_frontier` — the (σ_ratio, ρ) exclusion frontier
  used in the KKT-boundary figure.
* :func:`plot_kkt_boundary` — the signature paper figure (Fig. KKT):
  each base learner plotted at its (σ_i/σ_best, ρ_{i,best}) coordinate,
  overlaid with the theoretical exclusion curve. Points above the curve
  are the KKT-excluded members; the classical members should all sit
  there. Optionally rendered as small multiples over datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Diagnostics dataclass
# =============================================================================
@dataclass
class CSEDiagnostics:
    """Per-dataset diagnostics for the Convex Stacking Ensemble.

    All second-order statistics are computed on the **validation slice**
    (the tail of the training set held out for the weights), because that is the
    slice on which the weights are learned. This is the correct quantity
    for verifying Proposition 1; the test-slice equivalents are also
    stored for reference but should not be used in the theorem's
    statement.
    """

    dataset: str
    member_names: list[str]
    weights: np.ndarray                    # shape (K,), sum = 1
    sigma: np.ndarray                      # shape (K,), validation RMSE (uncentred)
    rho: np.ndarray                        # shape (K, K), uncentred error correlation
    Sigma: np.ndarray                      # shape (K, K), uncentred second moments E'E/n
    validation_errors: np.ndarray          # shape (K, n_val), signed errors
    blend_error: np.ndarray                # shape (n_val,), signed error of the blend
    best_member_idx: int                   # argmin sigma[i]
    kkt_lambda: float                      # λ = w'Σw, equal to (Σw)_i for active members
    kkt_residuals: np.ndarray              # shape (K,), (Σw)_i - λ (≥ 0 at optimum)
    n_validation: int
    extra: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Tidy per-member DataFrame — one row per base learner."""
        best = self.best_member_idx
        s_best = self.sigma[best]
        rows = []
        for i, name in enumerate(self.member_names):
            rows.append({
                "dataset": self.dataset,
                "member": name,
                "weight": float(self.weights[i]),
                "sigma": float(self.sigma[i]),
                "sigma_ratio_vs_best": float(self.sigma[i] / s_best) if s_best > 0 else float("nan"),
                "rho_vs_best": float(self.rho[i, best]) if best != i else 1.0,
                "kkt_residual": float(self.kkt_residuals[i]),
                "is_best": (i == best),
                "is_active": bool(self.weights[i] > 1e-6),
            })
        return pd.DataFrame(rows)


# =============================================================================
# Core computations
# =============================================================================
def _covariance_and_correlation(
    errors: np.ndarray, centred: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Σ, ρ, σ) from an (n, K) array of validation errors.

    By default the moments are **uncentred**: ``Σ = E'E / n``, ``σ_i`` is
    the validation RMSE of member ``i`` and ``ρ_ij = Σ_ij / (σ_i σ_j)``.
    That is the matrix in the stacking objective ``w' Σ w``, so it is the
    one Proposition 1 and the KKT conditions refer to. A biased member
    (large mean error) is penalised for its bias, as the optimiser does.

    ``centred=True`` reproduces the pre-v1.1 behaviour (covariance and
    Pearson correlation). It does not match the objective and should only
    be used for comparison.
    """
    n = errors.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 validation points for covariance.")
    E = errors - errors.mean(axis=0, keepdims=True) if centred else errors
    Sigma = (E.T @ E) / n
    sigma = np.sqrt(np.clip(np.diag(Sigma), 0.0, None))
    denom = np.outer(sigma, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(denom > 1e-12, Sigma / denom, 0.0)
    return Sigma, rho, sigma


def compute_cse_diagnostics(
    validation_errors: Mapping[str, np.ndarray] | pd.DataFrame,
    weights: Mapping[str, float] | np.ndarray,
    *,
    dataset: str = "unknown",
    extra: dict | None = None,
    centred: bool = False,
) -> CSEDiagnostics:
    """Assemble a :class:`CSEDiagnostics` from validation errors + simplex weights.

    Parameters
    ----------
    validation_errors
        Either a mapping ``{member_name: errors_1d}`` or a DataFrame whose
        columns are member names.
    weights
        Either a mapping ``{member_name: weight}`` (missing members are
        assumed zero) or an already-aligned array in the same order as
        ``validation_errors``' keys.
    dataset
        Free-form label carried through to the figure title.
    """
    if isinstance(validation_errors, pd.DataFrame):
        names = list(validation_errors.columns)
        err_matrix = validation_errors.values.astype(float)
    else:
        names = list(validation_errors.keys())
        err_matrix = np.column_stack([
            np.asarray(validation_errors[n], dtype=float).ravel() for n in names
        ])
    K = err_matrix.shape[1]

    if isinstance(weights, Mapping):
        w = np.array([float(weights.get(n, 0.0)) for n in names], dtype=float)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if len(w) != K:
            raise ValueError(f"weights length {len(w)} != K={K}")
    total = w.sum()
    if total > 0:
        w = w / total

    Sigma, rho, sigma = _covariance_and_correlation(err_matrix, centred=centred)
    Sigma_w = Sigma @ w
    # At the simplex optimum the multiplier equals the objective value:
    # λ = w'Σw (multiply the stationarity condition by w and sum).
    lam = float(w @ Sigma_w) if total > 0 else float("nan")
    kkt_residuals = Sigma_w - lam

    blend_error = err_matrix @ w
    best_idx = int(np.argmin(sigma))

    return CSEDiagnostics(
        dataset=dataset,
        member_names=names,
        weights=w,
        sigma=sigma,
        rho=rho,
        Sigma=Sigma,
        validation_errors=err_matrix.T,
        blend_error=blend_error,
        best_member_idx=best_idx,
        kkt_lambda=lam,
        kkt_residuals=kkt_residuals,
        n_validation=err_matrix.shape[0],
        extra=extra or {},
    )


def kkt_conditions(diag: CSEDiagnostics, tol: float = 1e-4) -> pd.DataFrame:
    """Human-readable KKT verification table for a single dataset.

    ``tol`` is relative to λ, so the check is scale-free (kW or MW).
    """
    scale = abs(diag.kkt_lambda) if np.isfinite(diag.kkt_lambda) and diag.kkt_lambda != 0 else 1.0
    rel = diag.kkt_residuals / scale
    rows = []
    for i, name in enumerate(diag.member_names):
        rows.append({
            "member": name,
            "weight": float(diag.weights[i]),
            "sigma_w_i": float((diag.Sigma @ diag.weights)[i]),
            "lambda": float(diag.kkt_lambda),
            "kkt_residual": float(diag.kkt_residuals[i]),
            "kkt_residual_rel": float(rel[i]),
            "condition": (
                ("active (equality)" if abs(rel[i]) <= tol else "VIOLATION")
                if diag.weights[i] > 1e-9
                else ("inactive (inequality satisfied)"
                      if rel[i] >= -tol else "VIOLATION")
            ),
        })
    return pd.DataFrame(rows)


def proposition1_frontier(sigma_ratio: np.ndarray) -> np.ndarray:
    """Return ρ = σ_best / σ_i evaluated at each ``sigma_ratio = σ_i / σ_best``.

    Points **above** this curve (higher ρ) are excluded by Proposition 1;
    points **below** can enter the blend with positive weight.
    """
    r = np.asarray(sigma_ratio, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(r > 0, 1.0 / r, np.nan)


# =============================================================================
# Signature paper figure
# =============================================================================
DATASET_LABEL = {"UCI": "UCI", "OPSD_DE_household": "OPSD-H", "OPSD_DE_total": "OPSD-N",
                 "ETTh1": "ETTh1", "ETTh2": "ETTh2"}
MEMBER_LABEL = {"sarima": "SARIMA", "hw": "Holt-Winters", "lgbm": "LightGBM",
                "catboost": "CatBoost", "extratrees": "ExtraTrees"}

def plot_kkt_boundary(
    diagnostics: CSEDiagnostics | list[CSEDiagnostics],
    *,
    title: str | None = None,
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
    ncols: int = 3,
):
    """Render the (σ_ratio, ρ) exclusion diagram of Proposition 1.

    Each non-leading member is plotted at (σ_i/σ_best, ρ_{i,best}) on a
    logarithmic ratio axis, against the curve ρ = σ_best/σ_i. Members above
    the curve are excluded by Proposition 1. Markers: filled circle for a
    positive weight, cross for an excluded member with zero weight, and an
    open circle for an admissible member that the full KKT system zeroed
    (collinearity). Colours are from the Okabe-Ito set, which is safe for
    colour-vision deficiency, and the three cases also differ in marker
    shape so they do not depend on colour alone.
    """
    import matplotlib.pyplot as plt

    diags = [diagnostics] if isinstance(diagnostics, CSEDiagnostics) else list(diagnostics)
    n = len(diags)
    if n == 0:
        raise ValueError("plot_kkt_boundary: no diagnostics supplied.")

    c_pos, c_excl, c_coll, c_lead = "#009E73", "#D55E00", "#E69F00", "#000000"
    label_of = lambda n: MEMBER_LABEL.get(n, n)  # noqa: E731
    ncols = min(n, ncols)
    nrows = int(np.ceil(n / ncols))
    fs = figsize or (3.4 * ncols, 2.9 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=fs, squeeze=False)

    for k, diag in enumerate(diags):
        ax = axes[k // ncols][k % ncols]
        best = diag.best_member_idx
        s_best = diag.sigma[best]
        ratios = [diag.sigma[i] / s_best for i in range(len(diag.member_names)) if i != best]
        x_max = max(4.0, 1.6 * max(ratios)) if ratios else 4.0
        x_grid = np.logspace(0, np.log10(x_max), 300)
        y_curve = proposition1_frontier(x_grid)
        ax.fill_between(x_grid, y_curve, 1.08, color=c_excl, alpha=0.07, lw=0)
        ax.fill_between(x_grid, -0.3, y_curve, color=c_pos, alpha=0.06, lw=0)
        ax.plot(x_grid, y_curve, color="black", ls="--", lw=1.0,
                label=r"$\rho=\sigma_{\mathrm{best}}/\sigma_i$")

        placed: list[tuple[float, float]] = []
        for i, name in enumerate(diag.member_names):
            if i == best:
                continue
            xr = diag.sigma[i] / s_best
            yr = diag.rho[i, best]
            active = diag.weights[i] > 1e-9
            admissible = yr < 1.0 / xr
            if active:
                ax.scatter([xr], [yr], s=48, color=c_pos, marker="o", edgecolor="black",
                           linewidth=0.6, zorder=3, label="admitted ($w_i>0$)")
            elif admissible:
                ax.scatter([xr], [yr], s=52, facecolor="white", edgecolor=c_coll, marker="o",
                           linewidth=1.6, zorder=3, label="admissible, zeroed by full KKT")
            else:
                ax.scatter([xr], [yr], s=52, color=c_excl, marker="X", edgecolor="black",
                           linewidth=0.5, zorder=3, label="excluded ($w_i=0$)")
            if annotate:
                w = diag.weights[i]
                wl = "0" if w <= 1e-9 else (f"{w:.2f}" if w >= 5e-3 else f"{w:.0e}")
                # Alternate the label side when a neighbour sits close by, so
                # labels of near-collinear members do not overlap.
                crowded = sum(1 for (px, py) in placed
                              if abs(np.log10(xr) - np.log10(px)) < 0.12 and abs(yr - py) < 0.12)
                offsets = [(6, 5), (6, -16), (-52, 9), (-52, -18)]
                dx, dy = offsets[crowded % len(offsets)]
                placed.append((xr, yr))
                ax.annotate(f"{label_of(name)}, $w$={wl}", (xr, yr), textcoords="offset points",
                            xytext=(dx, dy), fontsize=6.3)
        ax.scatter([1.0], [1.0], s=90, color="#F0E442", marker="*", edgecolor=c_lead,
                   linewidth=0.6, zorder=4, label="leader (1, 1)")

        ax.set_xscale("log")
        ax.set_xlim(0.93, x_max)
        from matplotlib.ticker import FixedLocator, NullFormatter
        from matplotlib.ticker import FuncFormatter
        base = (1, 1.5, 2, 3, 5, 10, 20) if x_max <= 60 else (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
        ticks = [t for t in base if t <= x_max]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(-0.12 if min([diag.rho[i, best] for i in range(len(diag.member_names))]) > -0.1 else -0.3, 1.08)
        ax.set_xlabel(r"$\sigma_i/\sigma_{\mathrm{best}}$ (log)", fontsize=8)
        ax.set_ylabel(r"$\rho_{i,\mathrm{best}}$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{DATASET_LABEL.get(diag.dataset, diag.dataset)}  (leader: {label_of(diag.member_names[best])})", fontsize=8.5)
        ax.grid(alpha=0.2, which="both", lw=0.4)

    # one shared legend, deduplicated
    handles, labels = [], []
    for row in axes:
        for ax in row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
    for k in range(n, nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        ax.axis("off")
        ax.legend(handles, labels, loc="center", fontsize=7.5, frameon=True)
    if n == nrows * ncols:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), fontsize=7.5)
        fig.subplots_adjust(bottom=0.16)

    if title:
        fig.suptitle(title, fontsize=9.5)
    fig.tight_layout()
    return fig


__all__ = [
    "CSEDiagnostics",
    "compute_cse_diagnostics",
    "kkt_conditions",
    "proposition1_frontier",
    "plot_kkt_boundary",
]
