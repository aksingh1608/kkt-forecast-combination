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

"""Three manuscript figures that read directly from ``results/``.

* ``figures/fig4_dm_pvalues.pdf``   Holm-adjusted Diebold-Mariano p-values of
  every common model against the best model, one row per dataset and
  protocol (replaces the single-dataset strip of earlier drafts).
* ``figures/fig5_error_drift_uci.pdf``   daily mean and cumulative absolute
  error over the UCI test period, one-step and recursive, for a fixed set of
  seven models.
* ``figures/fig6_horizon_ranks.pdf``   rolling-origin mean rank against
  horizon, by family.

Run::

    python scripts/build_paper_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASETS = ["UCI", "OPSD_DE_household", "OPSD_DE_total", "ETTh1", "ETTh2"]
DS_LABEL = {"UCI": "UCI", "OPSD_DE_household": "OPSD-H", "OPSD_DE_total": "OPSD-N",
            "ETTh1": "ETTh1", "ETTh2": "ETTh2"}
MODEL_ORDER = ["Naive", "NaiveDrift", "SeasonalNaive", "NLinear", "DLinear", "Ridge", "Lasso",
               "ElasticNet", "KNN", "SVR", "RandomForest", "ExtraTrees", "GradientBoosting",
               "HistGradientBoosting", "XGBoost", "LightGBM", "CatBoost", "QuantileLightGBM",
               "FAHE (Ours)"]
DISPLAY = {"FAHE (Ours)": "CSE (ours)", "FAHE_(Ours)": "CSE (ours)"}
FAMILY = {"Naive": "persistence", "NaiveDrift": "persistence", "SeasonalNaive": "persistence",
          "NLinear": "lite-DL", "DLinear": "lite-DL", "Ridge": "linear (lagged)",
          "Lasso": "linear (lagged)", "ElasticNet": "linear (lagged)", "KNN": "instance",
          "SVR": "kernel", "RandomForest": "trees / boosting", "ExtraTrees": "trees / boosting",
          "GradientBoosting": "trees / boosting", "HistGradientBoosting": "trees / boosting",
          "XGBoost": "trees / boosting", "LightGBM": "trees / boosting",
          "CatBoost": "trees / boosting", "QuantileLightGBM": "trees / boosting",
          "FAHE (Ours)": "CSE (ours)"}
# Okabe-Ito, safe for colour-vision deficiency; assigned in fixed order
OI = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#0072B2", "#CC79A7", "#F0E442"]
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.8, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
                     "axes.linewidth": 0.5, "lines.linewidth": 0.9})


def disp(m: str) -> str:
    return DISPLAY.get(m, m)


def fig_dm(results: Path, out: Path) -> None:
    rows, labels, refs = [], [], []
    for ds in DATASETS:
        for proto, pl in (("one_step", "one-step"), ("recursive", "recursive")):
            dm = pd.read_csv(results / "significance" / f"dm_matrix__{ds}__{proto}.csv")
            dm["model"] = dm.model.str.replace("_", " ")
            ref = dm.reference.iloc[0].replace("_", " ")
            p = dm.set_index("model").p_adj
            row = []
            for m in MODEL_ORDER:
                if m == ref:
                    row.append(np.nan)           # the reference itself
                elif m in p.index:
                    row.append(float(p[m]))
                else:
                    row.append(-1.0)             # model absent on this series
            rows.append(row); labels.append(f"{DS_LABEL[ds]}, {pl}"); refs.append(disp(ref))
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    cmap = plt.get_cmap("Greys").copy()
    shown = np.where(M < 0, np.nan, M)
    im = ax.imshow(np.clip(shown, 0, 0.2), cmap=cmap, vmin=0, vmax=0.2, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="////",
                                           edgecolor="#555555", lw=0))
            elif v < 0:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=5, color="#777777")
            elif v >= 0.05:
                ax.text(j, i, "tie", ha="center", va="center", fontsize=5.5,
                        color="black" if v < 0.12 else "white", fontweight="bold")
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([disp(m) for m in MODEL_ORDER], rotation=55, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"{l}  (best: {r})" for l, r in zip(labels, refs)])
    ax.set_xticks(np.arange(-.5, len(MODEL_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Holm-adjusted $p$ (clipped at 0.2)")
    cb.ax.axhline(0.05, color="#D55E00", lw=1.2)
    ax.set_title("Holm-adjusted Diebold-Mariano $p$ of every model against the best model: "
                 "white = separable at 5%, shaded \"tie\" = not separable, hatched = the reference",
                 fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fig4_dm_pvalues.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "fig4_dm_pvalues.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote fig4_dm_pvalues")


def fig_drift(results: Path, out: Path, dataset: str = "UCI") -> None:
    models = ["FAHE_(Ours)", "CatBoost", "ExtraTrees", "QuantileLightGBM", "Ridge", "SARIMA", "Naive"]
    colors = dict(zip(models, OI[:len(models)]))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2), sharex=True)
    for r, (proto, pl) in enumerate((("one_step", "One-step (teacher forced)"),
                                     ("recursive", "Recursive (full test window)"))):
        yt = pd.read_parquet(results / "predictions" / f"{dataset}__{proto}__y_true.parquet")["y_true"]
        for m in models:
            f = results / "predictions" / f"{dataset}__{proto}__{m}.parquet"
            if not f.exists():
                continue
            yp = pd.read_parquet(f)["y_pred"]
            ae = (yt - yp).abs()
            daily = ae.resample("D").mean()
            lw = 1.5 if m.startswith("FAHE") else 0.9
            axes[r, 0].plot(daily.index, daily.values, color=colors[m], lw=lw, label=disp(m).replace("_", " "))
            axes[r, 1].plot(ae.index, ae.cumsum().values, color=colors[m], lw=lw)
        axes[r, 0].set_ylabel(f"{pl}\ndaily mean |error| (kW)")
        axes[r, 1].set_ylabel("cumulative |error| (kW)")
        for a in axes[r]:
            a.grid(alpha=0.25, lw=0.4)
            a.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_title("Daily mean absolute error")
    axes[0, 1].set_title("Cumulative absolute error")
    for a in axes[1]:
        a.tick_params(axis="x", rotation=30)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models), frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / f"fig5_error_drift_{dataset.lower()}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"fig5_error_drift_{dataset.lower()}.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote fig5_error_drift_{dataset.lower()}")


def fig_horizon(results: Path, out: Path) -> None:
    long = pd.read_csv(results / "multistep_rolling" / "long_all__rolling.csv")
    piv = long.pivot_table(index=["Model", "Dataset"], columns="horizon", values="RMSE").unstack("Dataset")
    piv = piv.loc[[m for m in MODEL_ORDER if m in piv.index]].dropna(how="any")
    H = [1, 3, 6, 12, 24]
    rank = pd.DataFrame({h: piv.xs(h, axis=1, level="horizon").rank(axis=0, method="average").mean(axis=1) for h in H})
    rank["family"] = [FAMILY[m] for m in rank.index]
    fam = rank.groupby("family")[H].mean()
    order = ["trees / boosting", "CSE (ours)", "linear (lagged)", "lite-DL", "kernel", "instance", "persistence"]
    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    for i, f in enumerate(order):
        if f not in fam.index:
            continue
        ax.plot(H, fam.loc[f, H].values, marker="o", ms=3.2, color=OI[i], lw=1.6 if f == "CSE (ours)" else 1.0,
                label=f)
    ax.set_xscale("log"); ax.set_xticks(H); ax.set_xticklabels([str(h) for h in H])
    ax.minorticks_off()
    ax.set_xlabel("forecast horizon $h$ (hours)"); ax.set_ylabel("mean rank over 5 datasets (lower is better)")
    ax.invert_yaxis()
    ax.grid(alpha=0.25, lw=0.4); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(out / "fig6_horizon_ranks.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "fig6_horizon_ranks.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote fig6_horizon_ranks")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig_dm(a.results, a.out)
    fig_drift(a.results, a.out)
    fig_horizon(a.results, a.out)


if __name__ == "__main__":
    main()
