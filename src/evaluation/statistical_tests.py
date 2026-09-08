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

"""Statistical tests and effect-size measures for comparing forecasts.

This module implements the significance suite required by an IEEE-Transactions
submission:

Pairwise equal-accuracy tests
-----------------------------
* :func:`diebold_mariano` — pairwise DM test with Harvey-Leybourne-Newbold
  small-sample correction (Diebold & Mariano, 1995; Harvey et al., 1997).
* :func:`pairwise_dm_matrix` — HLN-corrected DM statistics and Holm-adjusted
  p-values for every pair of models against a reference (typically the best
  single model). Returns a tidy DataFrame suitable for :func:`plot_dm_pvalue_heatmap`.

Effect sizes (distribution-free & parametric)
---------------------------------------------
* :func:`cohens_d` — standardised mean difference on paired absolute errors.
* :func:`cliffs_delta` — rank-based, distribution-free effect size for
  paired samples. Values in [-1, 1]; |δ| ≥ 0.474 is "large" by
  Romano et al. (2006).

Multiple-comparison correction
------------------------------
* :func:`holm_correct`, :func:`benjamini_hochberg` — family-wise error rate
  and false-discovery-rate p-value adjustments.

Omnibus tests
-------------
* :func:`friedman_test` — omnibus rank-test across ≥3 models.

Visualisation
-------------
* :func:`plot_dm_pvalue_heatmap` — Holm-corrected DM p-value heatmap; the
  standard "significance matrix" figure of IEEE-TSG / T-PWRS forecasting
  papers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Diebold-Mariano
# =============================================================================
@dataclass
class DMResult:
    statistic: float
    p_value: float
    h: int
    power: float


def diebold_mariano(
    y_true: np.ndarray | pd.Series,
    pred_a: np.ndarray | pd.Series,
    pred_b: np.ndarray | pd.Series,
    *,
    h: int = 1,
    power: float = 2.0,
) -> DMResult:
    """Diebold–Mariano test with Harvey-Leybourne-Newbold (HLN) correction.

    H0: the two forecasts have equal expected loss.
    Negative statistic ⇒ ``pred_a`` is more accurate; positive ⇒ ``pred_b``.
    """
    e_a = np.asarray(y_true).ravel() - np.asarray(pred_a).ravel()
    e_b = np.asarray(y_true).ravel() - np.asarray(pred_b).ravel()
    d = np.abs(e_a) ** power - np.abs(e_b) ** power
    n = len(d)
    if n < 2:
        return DMResult(float("nan"), float("nan"), h, power)

    mean_d = d.mean()
    # Newey-West long-run variance with bandwidth h-1.
    gamma0 = d.var(ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        cov = np.cov(d[:-lag], d[lag:], ddof=0)[0, 1]
        var_d += 2 * (1 - lag / h) * cov
    var_d = max(var_d, 1e-12)
    dm_stat = mean_d / np.sqrt(var_d / n)

    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    hln_stat = dm_stat * correction
    p_value = 2 * (1 - stats.t.cdf(np.abs(hln_stat), df=n - 1))
    return DMResult(float(hln_stat), float(p_value), h, power)


def pairwise_dm_matrix(
    y_true: np.ndarray | pd.Series,
    predictions: Mapping[str, np.ndarray | pd.Series],
    *,
    reference: str | None = None,
    h: int = 1,
    power: float = 2.0,
    correction: str = "holm",
) -> pd.DataFrame:
    """Return a tidy DM table comparing every model against a reference.

    Parameters
    ----------
    y_true, predictions
        The test-set actuals and a mapping ``{model_name: predictions}``.
    reference
        Name of the reference model in ``predictions``. When ``None``, the
        model with the lowest RMSE is chosen automatically.
    h
        Multi-step DM bandwidth (Newey-West). Use the forecast horizon.
    power
        Loss exponent (2 = squared error, 1 = absolute error).
    correction
        Multiple-comparison correction on the raw p-values. One of
        ``"holm"`` (family-wise), ``"bh"`` (false-discovery), or ``"none"``.
    """
    if reference is None:
        rmses = {n: float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(p)) ** 2)))
                 for n, p in predictions.items()}
        reference = min(rmses, key=rmses.get)

    if reference not in predictions:
        raise KeyError(f"reference {reference!r} not in predictions")

    ref_pred = np.asarray(predictions[reference]).ravel()
    rows = []
    for name, pred in predictions.items():
        if name == reference:
            continue
        res = diebold_mariano(y_true, ref_pred, pred, h=h, power=power)
        rows.append({
            "reference": reference,
            "model": name,
            "dm_stat": res.statistic,
            "p_raw": res.p_value,
        })
    out = pd.DataFrame(rows).sort_values("p_raw", na_position="last").reset_index(drop=True)
    if correction == "holm":
        out["p_adj"] = holm_correct(out["p_raw"].values)
    elif correction == "bh":
        out["p_adj"] = benjamini_hochberg(out["p_raw"].values)
    elif correction == "none":
        out["p_adj"] = out["p_raw"]
    else:
        raise ValueError(f"correction must be holm/bh/none, got {correction!r}")
    out["significant_5pct"] = out["p_adj"] < 0.05
    return out


# =============================================================================
# Effect sizes
# =============================================================================
def cohens_d(
    y_true: np.ndarray | pd.Series,
    pred_a: np.ndarray | pd.Series,
    pred_b: np.ndarray | pd.Series,
) -> float:
    """Cohen's d on paired absolute-error differences.

    ``d = mean(|e_a| - |e_b|) / sd(|e_a| - |e_b|)``. Negative ⇒ ``pred_a``
    has smaller errors; positive ⇒ ``pred_b``. By convention |d|≥0.2 is
    small, ≥0.5 medium, ≥0.8 large (Cohen, 1988).
    """
    yt = np.asarray(y_true).ravel()
    ea = np.abs(yt - np.asarray(pred_a).ravel())
    eb = np.abs(yt - np.asarray(pred_b).ravel())
    diff = ea - eb
    sd = diff.std(ddof=1)
    if sd < 1e-12 or not np.isfinite(sd):
        return float("nan")
    return float(diff.mean() / sd)


def cliffs_delta(
    y_true: np.ndarray | pd.Series,
    pred_a: np.ndarray | pd.Series,
    pred_b: np.ndarray | pd.Series,
) -> float:
    """Cliff's δ on paired absolute errors.

    δ = P(|e_a| > |e_b|) - P(|e_a| < |e_b|), computed exactly by counting
    dominance. Values in [-1, 1]; negative ⇒ ``pred_a`` has smaller errors.
    Thresholds (Romano et al., 2006): |δ|<0.147 negligible, <0.33 small,
    <0.474 medium, else large.
    """
    yt = np.asarray(y_true).ravel()
    ea = np.abs(yt - np.asarray(pred_a).ravel())
    eb = np.abs(yt - np.asarray(pred_b).ravel())
    n = len(ea)
    if n == 0:
        return float("nan")
    # Vectorised pairwise dominance using rank ranks scale O(n log n).
    combined = np.concatenate([ea, eb])
    ranks = stats.rankdata(combined)
    r_a = ranks[:n].sum()
    r_b = ranks[n:].sum()
    # Cliff's δ = (R_a - R_b) / (n*n) using Mann–Whitney relationship.
    # See Cliff (1993) — for equal-sized paired samples this reduces to a
    # normalised rank-sum difference on [-1, 1].
    delta = (r_a - r_b) / (n * n)
    return float(delta)


# =============================================================================
# Multiple-comparison correction
# =============================================================================
def holm_correct(p_values: Sequence[float]) -> np.ndarray:
    """Holm–Bonferroni step-down family-wise-error-rate correction."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running_max = 0.0
    for i, idx in enumerate(order):
        val = (n - i) * p[idx]
        running_max = max(running_max, val)
        adj[idx] = min(1.0, running_max)
    return adj


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini–Hochberg step-up FDR correction."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]
        val = p[idx] * n / (i + 1)
        running_min = min(running_min, val)
        adj[idx] = min(1.0, running_min)
    return adj


# =============================================================================
# Omnibus test
# =============================================================================
def friedman_test(metric_table: pd.DataFrame) -> tuple[float, float]:
    """Friedman omnibus test on a (folds × models) metric DataFrame."""
    if metric_table.shape[1] < 3:
        raise ValueError("Friedman test requires at least 3 models.")
    stat, p_value = stats.friedmanchisquare(
        *[metric_table[col].values for col in metric_table.columns]
    )
    return float(stat), float(p_value)


# =============================================================================
# Visualisation
# =============================================================================
def plot_dm_pvalue_heatmap(
    dm_table: pd.DataFrame,
    *,
    p_col: str = "p_adj",
    title: str | None = None,
    cmap: str = "viridis_r",
):
    """Render a DM p-value heatmap (reference on the y-axis, models on x-axis).

    ``dm_table`` is the output of :func:`pairwise_dm_matrix`. Cells shade the
    Holm-adjusted p-value; the diagonal (reference vs. itself) is masked.
    """
    import matplotlib.pyplot as plt

    if dm_table.empty:
        raise ValueError("dm_table is empty — nothing to plot.")

    ref = dm_table["reference"].iloc[0]
    models = [ref] + dm_table["model"].tolist()
    n = len(models)
    grid = np.full((n, n), np.nan)
    labels = np.full((n, n), "", dtype=object)
    for _, row in dm_table.iterrows():
        j = models.index(row["model"])
        grid[0, j] = row[p_col]
        labels[0, j] = "*" if row["p_adj"] < 0.05 else ""

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * n), 2.4))
    im = ax.imshow(grid[:1, :], vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
    for j in range(n):
        v = grid[0, j]
        if np.isfinite(v):
            ax.text(j, 0, f"{v:.3f}{labels[0, j]}",
                    ha="center", va="center",
                    color="white" if v < 0.5 else "black", fontsize=9)
    ax.set_xticks(range(n))
    ax.set_xticklabels(models, rotation=40, ha="right")
    ax.set_yticks([0])
    ax.set_yticklabels([f"{ref} (ref)"])
    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.35,
                 label=f"{p_col} (Holm-adj.) — * significant at α=0.05")
    ax.set_title(title or f"Diebold–Mariano vs. {ref} (HLN, Holm-corrected)")
    fig.tight_layout()
    return fig


__all__ = [
    "DMResult",
    "diebold_mariano",
    "pairwise_dm_matrix",
    "cohens_d",
    "cliffs_delta",
    "holm_correct",
    "benjamini_hochberg",
    "friedman_test",
    "plot_dm_pvalue_heatmap",
]
