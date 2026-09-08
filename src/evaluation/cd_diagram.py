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

"""Critical-Difference (CD) diagrams — Demšar (2006).

A CD diagram is the standard visualisation for the Friedman/Nemenyi
post-hoc test in classifier/forecaster comparison studies. We compute
the average rank of every model across datasets, then connect groups
that are *not* statistically different with a horizontal bar of
length ≤ the critical-difference distance.

Returns a Matplotlib figure suitable for the paper.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Studentised-range statistic q_α at α=0.05 for k models in {2, …, 50}.
# Source: Demšar (2006) Table 5; values for k > 10 interpolated from
# scipy.stats.studentized_range with df=∞.
_Q_ALPHA_005 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
    8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
    14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517,
    20: 3.544, 25: 3.658, 30: 3.749, 40: 3.892, 50: 3.998,
}


def _q_alpha(k: int, alpha: float = 0.05) -> float:
    """Look up / approximate the studentised-range critical value."""
    if alpha != 0.05:
        try:
            from scipy.stats import studentized_range
            return float(studentized_range.ppf(1 - alpha, k, np.inf))
        except Exception:
            logger.warning("Falling back to α=0.05 CD value (scipy unavailable).")
    if k in _Q_ALPHA_005:
        return _Q_ALPHA_005[k]
    keys = sorted(_Q_ALPHA_005)
    # Linear interpolation between the two nearest k.
    lo = max(x for x in keys if x <= k) if any(x <= k for x in keys) else keys[0]
    hi = min(x for x in keys if x >= k) if any(x >= k for x in keys) else keys[-1]
    if lo == hi:
        return _Q_ALPHA_005[lo]
    frac = (k - lo) / (hi - lo)
    return _Q_ALPHA_005[lo] + frac * (_Q_ALPHA_005[hi] - _Q_ALPHA_005[lo])


@dataclass
class CDResult:
    avg_ranks: pd.Series       # average rank per model (lower = better)
    cd: float                  # critical-difference distance
    n_datasets: int
    alpha: float
    figure: object | None      # matplotlib Figure (only if requested)


def critical_difference(
    rank_matrix: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> tuple[float, pd.Series]:
    """Return (CD, average_ranks).

    ``rank_matrix`` is a DataFrame whose rows are datasets and columns
    are models, containing the per-dataset ranks (1 = best).
    """
    n_datasets = rank_matrix.shape[0]
    k = rank_matrix.shape[1]
    if n_datasets < 2 or k < 2:
        raise ValueError("Need ≥ 2 datasets and ≥ 2 models for a CD diagram.")
    q = _q_alpha(k, alpha)
    cd = q * math.sqrt(k * (k + 1) / (6.0 * n_datasets))
    avg_ranks = rank_matrix.mean(axis=0).sort_values()
    return cd, avg_ranks


def plot_cd_diagram(
    rank_matrix: pd.DataFrame,
    *,
    alpha: float = 0.05,
    title: str = "Critical-Difference (Nemenyi α = 0.05)",
):
    """Render a Demšar CD diagram. Returns a matplotlib Figure."""
    import matplotlib.pyplot as plt

    cd, avg_ranks = critical_difference(rank_matrix, alpha=alpha)
    n = len(avg_ranks)
    names = avg_ranks.index.tolist()
    ranks = avg_ranks.values

    fig, ax = plt.subplots(figsize=(10, 1.5 + 0.25 * n))
    ax.set_xlim(0, n + 1)
    ax.set_ylim(-1.0 - 0.25 * n, 1.5)
    ax.invert_yaxis()
    ax.axis("off")

    # Top axis with rank scale
    ax.plot([0.5, n + 0.5], [0, 0], "k", lw=1.2)
    for r in range(1, n + 1):
        ax.plot([r, r], [0, -0.05 * (1 + 0.05 * n)], "k", lw=1.2)
        ax.text(r, -0.10 * (1 + 0.05 * n), f"{r}",
                ha="center", va="top", fontsize=9)

    # Names + ranks: split left/right of midpoint for readability.
    midpoint = (n + 1) / 2
    for i, (name, rank) in enumerate(zip(names, ranks)):
        side_left = rank <= midpoint
        y_text = -0.30 - 0.18 * (i if side_left else (n - 1 - i))
        x_text = 0.05 if side_left else n + 0.95
        ax.plot([rank, rank], [0, y_text + 0.05], "k:", lw=0.8)
        ax.plot([rank, x_text + (0 if side_left else 0)], [y_text + 0.05, y_text + 0.05],
                "k", lw=0.8)
        ax.text(
            x_text, y_text + 0.05,
            f"{name}  ({rank:.2f})" if side_left else f"({rank:.2f})  {name}",
            ha="left" if side_left else "right", va="center", fontsize=10,
        )

    # CD bar (legend)
    ax.plot([1, 1 + cd], [0.6, 0.6], "k", lw=2.5)
    ax.text(1 + cd / 2, 0.85, f"CD = {cd:.2f}",
            ha="center", va="center", fontsize=10)

    # Connect non-significantly-different cliques
    # Build cliques: groups where max(rank) - min(rank) <= cd.
    clique_levels: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if ranks[j] - ranks[i] > cd:
                continue
            # Find a vertical level that doesn't collide.
            level = 0.15
            while any(abs(level - lv) < 0.03 for lv in clique_levels):
                level += 0.05
            clique_levels.append(level)
            ax.plot([ranks[i] - 0.05, ranks[j] + 0.05],
                    [level, level], "k", lw=4, solid_capstyle="round")
            break

    ax.set_title(title, pad=20)
    fig.tight_layout()
    return fig, CDResult(
        avg_ranks=avg_ranks, cd=cd, n_datasets=rank_matrix.shape[0],
        alpha=alpha, figure=fig,
    )


__all__ = ["critical_difference", "plot_cd_diagram", "CDResult"]
