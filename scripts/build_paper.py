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

"""Generate every manuscript table and inline number from ``results/``.

This script writes LaTeX fragments to
``paper/generated/`` which ``paper/main.tex`` includes:

* ``tab_main.tex``        test RMSE, both protocols, mean rank (Table II)
* ``tab_family.tex``      degradation ratios and rolling-origin ranks (Table III)
* ``tab_weights.tex``     simplex weights and tree mass (Table IV)
* ``tab_agree.tex``       exclusion boundary against learned weight (Table V)
* ``tab_sensitivity.tex`` member-pool sensitivity (Table VI)
* ``tab_folds.tex``       expanding-window folds (Table VII)
* ``tab_dm.tex``          Diebold-Mariano, CSE against the best model (Table VIII)
* ``tab_tuned.tex``       tuned lite-DL configurations (Table IX)
* ``tab_refit.tex``       CSE time when only active members are refit (Table X)
* ``tab_cost.tex``        training times (Table XI)
* ``tab_app.tex``         rolling-origin RMSE by horizon (Table XII, appendix)
* ``numbers.tex``         ``\\newcommand`` macros for every inline number

CSE rows of datasets that have not been refit with the exact solver (those
missing from ``results/tables/cse_refit_times.csv``) are wrapped in
``\\pending{}`` so they show up highlighted until ``scripts/rerun_cse.py``
has been run on them.

Run::

    python scripts/build_paper.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.kkt_analysis import compute_cse_diagnostics  # noqa: E402
from src.models.novel import solve_simplex_qp  # noqa: E402

CSE_RAW = "FAHE (Ours)"
CSE_SAFE = "FAHE_(Ours)"
DATASETS = ["UCI", "OPSD_DE_household", "OPSD_DE_total", "ETTh1", "ETTh2"]
DS_LABEL = {"UCI": "UCI", "OPSD_DE_household": "OPSD-H", "OPSD_DE_total": "OPSD-N",
            "ETTh1": "ETTh1", "ETTh2": "ETTh2"}
MODEL_ORDER = ["Naive", "NaiveDrift", "SeasonalNaive", "ARIMA", "SARIMA", "Holt-Winters",
               "Theta", "NLinear", "DLinear", "Ridge", "Lasso", "ElasticNet", "KNN", "SVR",
               "RandomForest", "ExtraTrees", "GradientBoosting", "HistGradientBoosting",
               "XGBoost", "LightGBM", "CatBoost", "QuantileLightGBM", CSE_RAW]
DISPLAY = {CSE_RAW: r"\textbf{CSE (ours)}"}
FAMILY = {"Naive": "persist.", "NaiveDrift": "persist.", "SeasonalNaive": "persist.",
          "ARIMA": "class.", "SARIMA": "class.", "Holt-Winters": "class.", "Theta": "class.",
          "NLinear": "lite-DL", "DLinear": "lite-DL", "Ridge": "linear", "Lasso": "linear",
          "ElasticNet": "linear", "KNN": "inst.", "SVR": "kernel", "RandomForest": "tree",
          "ExtraTrees": "tree", "GradientBoosting": "boost", "HistGradientBoosting": "boost",
          "XGBoost": "boost", "LightGBM": "boost", "CatBoost": "boost",
          "QuantileLightGBM": "boost", CSE_RAW: "hybrid"}
FAMILY_GROUP = {"Naive": "persistence", "NaiveDrift": "persistence", "SeasonalNaive": "persistence",
                "ARIMA": "classical", "SARIMA": "classical", "Holt-Winters": "classical",
                "Theta": "classical", "NLinear": "lite-DL", "DLinear": "lite-DL",
                "Ridge": "linear (lagged)", "Lasso": "linear (lagged)", "ElasticNet": "linear (lagged)",
                "KNN": "instance", "SVR": "kernel", "RandomForest": "trees / boosting",
                "ExtraTrees": "trees / boosting", "GradientBoosting": "trees / boosting",
                "HistGradientBoosting": "trees / boosting", "XGBoost": "trees / boosting",
                "LightGBM": "trees / boosting", "CatBoost": "trees / boosting",
                "QuantileLightGBM": "trees / boosting", CSE_RAW: "CSE (ours)"}
FAMILY_ORDER = ["persistence", "classical", "lite-DL", "linear (lagged)", "instance", "kernel",
                "trees / boosting", "CSE (ours)"]
MEMBER_LABEL = {"sarima": "SARIMA", "hw": "Holt-Winters", "lgbm": "LightGBM",
                "catboost": "CatBoost", "extratrees": "ExtraTrees"}
TREES = {"lgbm", "catboost", "extratrees"}
HORIZONS = [1, 3, 6, 12, 24]


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------
def fnum(x: float, sig: int = 4) -> str:
    """4 significant digits, thousands separated, scientific above 1e6."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    ax = abs(x)
    if ax >= 1e6:
        e = int(np.floor(np.log10(ax)))
        m = x / 10 ** e
        return rf"${m:.1f}{{\times}}10^{{{e}}}$"
    if ax >= 1000:
        return f"{int(round(x)):,}".replace(",", r"\,")
    if ax >= 100:
        return f"{x:.1f}"
    if ax >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def fp(p: float) -> str:
    """Math-mode fragment with its operator: use as ``$p \\macro$``."""
    if p < 1e-4:
        return r"< 10^{-4}"
    return f"= {p:.3f}"


def fpcell(p: float) -> str:
    """Table cell version."""
    if p < 1e-4:
        return r"$< 10^{-4}$"
    return f"{p:.3f}"


def disp(model: str) -> str:
    return DISPLAY.get(model, model)


def _times(n: int) -> str:
    words = {0: "never", 1: "once", 2: "twice", 3: "three times", 4: "four times", 5: "five times"}
    return words.get(n, f"{n} times")


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def load(results: Path):
    lb = pd.read_csv(results / "leaderboards.csv")
    done = set()
    p = results / "tables" / "cse_refit_times.csv"
    if p.exists():
        done = set(pd.read_csv(p)["dataset"])
    return lb, done


def diag_tables(results: Path):
    """Weights and boundary table from the stored validation errors (exact solver)."""
    wrows, brows, masses = [], [], {}
    for ds in DATASETS:
        path = results / "cse_diagnostics" / f"{ds}.pkl"
        if not path.exists():
            continue
        with open(path, "rb") as fh:
            old = pickle.load(fh)
        names = list(old.member_names)
        E = np.asarray(old.validation_errors, dtype=float)
        S = (E @ E.T) / E.shape[1]
        w = solve_simplex_qp(S)
        d = compute_cse_diagnostics({n: E[i] for i, n in enumerate(names)}, w, dataset=ds)
        b = d.best_member_idx
        masses[ds] = float(sum(w[i] for i, n in enumerate(names) if n in TREES))
        for i, n in enumerate(names):
            wrows.append({"dataset": ds, "member": n, "weight": float(w[i])})
            if i == b:
                continue
            ratio = d.sigma[i] / d.sigma[b]
            rho = d.rho[i, b]
            thr = 1.0 / ratio
            pred = "adm." if rho < thr else "excl."
            active = w[i] > 1e-9
            agree = ("\\checkmark" if (pred == "excl.") == (not active) else "coll.") \
                if not (pred == "excl." and active) else "VIOL."
            brows.append({"dataset": ds, "member": n, "ratio": ratio, "rho": rho, "thr": thr,
                          "pred": pred, "w": float(w[i]), "agree": agree,
                          "kkt_rel": float((d.kkt_residuals[i]) / d.kkt_lambda)})
    return pd.DataFrame(wrows), pd.DataFrame(brows), masses


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def tab_main(lb: pd.DataFrame, done: set, out: Path, N: dict) -> None:
    piv = {}
    ranks = {}
    for proto in ("one_step", "recursive"):
        sub = lb[lb.Protocol == proto].pivot_table(index="Model", columns="Dataset",
                                                   values="RMSE", aggfunc="first")
        piv[proto] = sub
        common = sub.dropna(axis=0, how="any")
        ranks[proto] = common.rank(axis=0, method="average").mean(axis=1)
        N[f"nCommon{'One' if proto=='one_step' else 'Rec'}"] = str(len(common))
    lines = [r"\begin{tabular}{llrrrrrrrrrrrr}", r"\toprule",
             r"& & \multicolumn{6}{c}{One-step (teacher forced)} & \multicolumn{6}{c}{Recursive multi-step (full test window)} \\",
             r"\cmidrule(lr){3-8}\cmidrule(lr){9-14}",
             r"Model & Family & UCI & OPSD-H & OPSD-N & ETTh1 & ETTh2 & rank & UCI & OPSD-H & OPSD-N & ETTh1 & ETTh2 & rank \\",
             r"\midrule"]
    best = {p: {ds: piv[p][ds].min() for ds in DATASETS} for p in piv}
    for m in MODEL_ORDER:
        cells = [disp(m), FAMILY[m]]
        for proto in ("one_step", "recursive"):
            for ds in DATASETS:
                v = piv[proto].at[m, ds] if (m in piv[proto].index and ds in piv[proto].columns) else np.nan
                if np.isnan(v):
                    cells.append("--")
                    continue
                s = fnum(v)
                if abs(v - best[proto][ds]) <= 5e-4 * max(1.0, abs(best[proto][ds])):
                    s = rf"\textbf{{{s}}}"
                if m == CSE_RAW and ds not in done:
                    s = rf"\pending{{{s}}}"
                cells.append(s)
            r = ranks[proto].get(m, np.nan)
            cells.append("--" if np.isnan(r) else f"{r:.1f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_main.tex").write_text("\n".join(lines) + "\n")

    # headline numbers
    for proto, key in (("one_step", "One"), ("recursive", "Rec")):
        r = ranks[proto].sort_values()
        N[f"cseRank{key}"] = f"{r.get(CSE_RAW, np.nan):.1f}"
        others = r.drop(CSE_RAW, errors="ignore")
        N[f"bestOtherRank{key}"] = f"{others.iloc[0]:.1f}"
        N[f"bestOtherRank{key}Name"] = disp(others.index[0])
        N[f"secondOtherRank{key}"] = f"{others.iloc[1]:.1f}"
        N[f"secondOtherRank{key}Name"] = disp(others.index[1])
        N[f"csePlace{key}"] = str(int((r < r.get(CSE_RAW, np.inf) - 1e-9).sum()) + 1)
    one = piv["one_step"]
    for ds in DATASETS:
        col = one[ds].dropna().sort_values()
        N[f"bestOne{ds}"] = disp(col.index[0])
        N[f"bestOne{ds}Val"] = fnum(col.iloc[0])
        colr = piv["recursive"][ds].dropna().sort_values()
        N[f"bestRec{ds}"] = disp(colr.index[0])
        N[f"bestRec{ds}Val"] = fnum(colr.iloc[0])
        N[f"cseOne{ds}"] = fnum(one.at[CSE_RAW, ds])
        N[f"cseRec{ds}"] = fnum(piv["recursive"].at[CSE_RAW, ds])
    tree_models = {m for m, f in FAMILY_GROUP.items() if f == "trees / boosting"}
    tw = [(ds, one[ds].dropna().idxmin()) for ds in DATASETS]
    tw = [(ds, m) for ds, m in tw if m in tree_models]
    N["nTreeWinsOne"] = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}[len(tw)]
    N["treeWinsOneList"] = ", ".join(f"{disp(m)} on {DS_LABEL[ds]}" for ds, m in tw) or "none"
    # ---- sentences that depend on which model wins -------------------------
    persist = {"Naive", "NaiveDrift", "SeasonalNaive"}
    rec = piv["recursive"]
    # recursive winners and exact ties at the top
    tie_notes = []
    for ds in DATASETS:
        col = rec[ds].dropna().sort_values()
        top = col[(col - col.iloc[0]).abs() <= 5e-4 * max(1.0, abs(col.iloc[0]))]
        if len(top) > 1:
            unit = "kW" if ds in ("UCI", "OPSD_DE_household") else ("MW" if ds == "OPSD_DE_total" else "")
            names = " and ".join(disp(m) for m in top.index)
            tie_notes.append(f"On {DS_LABEL[ds]} under recursion, {names} tie at {fnum(col.iloc[0])}~{unit}.".replace("~.", "."))
    N["recTieNote"] = " ".join(tie_notes)
    # OPSD-H recursive sentence
    col = rec["OPSD_DE_household"].dropna().sort_values()
    naive = rec.at["Naive", "OPSD_DE_household"]
    learned = [m for m in col.index if m not in persist]
    beaten = [m for m in learned if col[m] > naive + 1e-12]
    if col.index[0] in persist:
        N["recOPSDHSentence"] = (f"On OPSD-H the best model is persistence ({disp(col.index[0])}, {fnum(col.iloc[0])}~kW), "
                                 f"that is, no model at all: {len(beaten)} of the {len(learned)} learned predictors are beaten "
                                 f"by Naive persistence once they must consume their own outputs over 7\\,059 hours.")
    else:
        N["recOPSDHSentence"] = (f"On OPSD-H the best model is {disp(col.index[0])} ({fnum(col.iloc[0])}~kW), and {len(beaten)} of "
                                 f"the {len(learned)} learned predictors are beaten by Naive persistence once they must consume "
                                 f"their own outputs over 7\\,059 hours.")
    N["bestRecOPSDHIsPersistence"] = "1" if col.index[0] in persist else "0"
    trees = {m for m, f in FAMILY_GROUP.items() if f == "trees / boosting"}
    def _top(colr):
        colr = colr.dropna().sort_values()
        return list(colr[(colr - colr.iloc[0]).abs() <= 5e-4 * max(1.0, abs(colr.iloc[0]))].index)
    tl = [DS_LABEL[ds] for ds in DATASETS if any(m in trees for m in _top(rec[ds]))]
    N["recTreeLeaderList"] = " and ".join(tl) if len(tl) <= 2 else ", ".join(tl[:-1]) + " and " + tl[-1]
    N["nRecTreeLeaders"] = str(len(tl))
    # divergent lagged linear models under recursion
    div = []
    for ds in DATASETS:
        r, o = rec.at["Ridge", ds], one.at["Ridge", ds]
        if np.isfinite(r) and r > 1e3 * o:
            div.append(DS_LABEL[ds])
    linear = {m for m, f in FAMILY_GROUP.items() if f == "linear (lagged)"}
    lw = [(ds, one[ds].dropna().idxmin()) for ds in DATASETS]
    lw = [(ds, m) for ds, m in lw if m in linear]
    N["nLinearWinsOne"] = str(len(lw))
    N["linearWinsOneList"] = " and ".join(DS_LABEL[ds] for ds, _ in lw) or "none"
    N["linearWinsOneModel"] = disp(lw[0][1]) if lw else "none"
    # Kendall tau summary sentence
    taus = {}
    for ds in DATASETS:
        a = piv["one_step"][ds].dropna(); b = piv["recursive"][ds].dropna()
        idx = a.index.intersection(b.index)
        taus[ds] = kendalltau(a[idx].rank(), b[idx].rank())
    t_h, p_h = taus["OPSD_DE_household"]
    if t_h < 0 and p_h < 0.05:
        N["tauOPSDHSentence"] = (f"On the household series, the setting most single-dataset papers study, the correlation is "
                                 f"negative ($\\tau = {t_h:.2f}$, $p = {p_h:.3f}$): the models that lead one step ahead are, "
                                 f"on the whole, the ones that trail once they must consume their own forecasts.")
        N["tauOPSDHClause"] = f"$\\tau$ is negative on the household series ($\\tau = {t_h:.2f}$)"
    elif p_h >= 0.05:
        N["tauOPSDHSentence"] = ("On the household series, the setting most single-dataset papers study, the one-step ranking "
                                 "says little about the recursive one.")
        N["tauOPSDHClause"] = f"$\\tau$ falls to {t_h:.2f} on the household series"
    else:
        N["tauOPSDHSentence"] = ("On the household series the two rankings are only moderately related.")
        N["tauOPSDHClause"] = f"$\\tau$ is {t_h:.2f} on the household series"
    tv = [t for t, _ in taus.values()]
    N["tauMin"] = f"{min(tv):.2f}"
    N["tauMax"] = f"{max(tv):.2f}"
    N["ridgeDivergeList"] = " and ".join(div) if div else "none"
    lab2ds = {v: k for k, v in DS_LABEL.items()}
    if div:
        vals = " and ".join(f"{fnum(rec.at['Ridge', lab2ds[d]])} on {d}" for d in div)
        N["ridgeDivergeSentence"] = (f"The regularised linear models, competitive one step ahead, diverge on {N['ridgeDivergeList']}: "
                                     f"Ridge reaches RMSE {vals}, with Lasso and ElasticNet similar.")
    else:
        N["ridgeDivergeSentence"] = "The regularised linear models, competitive one step ahead, degrade under recursion but do not diverge."
    N["nRidgeDiverge"] = str(len(div))
    N["ridgeRankOne"] = f"{ranks['one_step'].get('Ridge'):.1f}"
    N["etRankOne"] = f"{ranks['one_step'].get('ExtraTrees'):.1f}"
    N["catRankOne"] = f"{ranks['one_step'].get('CatBoost'):.1f}"
    N["uciGapSarimaET"] = f"{100*(one.at['SARIMA','UCI']/one.at['ExtraTrees','UCI']-1):.0f}"
    # Kendall tau between protocols per dataset
    for ds in DATASETS:
        a = piv["one_step"][ds].dropna()
        b = piv["recursive"][ds].dropna()
        idx = a.index.intersection(b.index)
        tau, p = kendalltau(a[idx].rank(), b[idx].rank())
        N[f"tau{ds}"] = f"{tau:.2f}"
        N[f"tauP{ds}"] = f"{p:.3f}"
    # Ridge divergence values
    N["ridgeRecOPSDH"] = fnum(piv["recursive"].at["Ridge", "OPSD_DE_household"])
    N["ridgeRecOPSDN"] = fnum(piv["recursive"].at["Ridge", "OPSD_DE_total"])
    N["etRecUCI"] = fnum(piv["recursive"].at["ExtraTrees", "UCI"])


def tab_family(lb: pd.DataFrame, results: Path, done: set, out: Path, N: dict) -> None:
    one = lb[lb.Protocol == "one_step"].set_index(["Model", "Dataset"])["RMSE"]
    rec = lb[lb.Protocol == "recursive"].set_index(["Model", "Dataset"])["RMSE"]
    ratio = (rec / one).dropna().rename("ratio").reset_index()
    ratio["family"] = ratio.Model.map(FAMILY_GROUP)
    ratio.loc[ratio.family == "persistence", "family"] = "classical"
    diverged = ratio.ratio > 1e3
    N["nDiverged"] = str(int(diverged.sum()))
    ratio = ratio[~diverged]
    med = ratio.groupby(["family", "Dataset"]).ratio.median().unstack("Dataset")
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"\multicolumn{6}{c}{\emph{Degradation ratio, recursive against one-step (median within family)}} \\",
             r"Family & UCI & OPSD-H & OPSD-N & ETTh1 & ETTh2 \\", r"\midrule"]
    fam_rows = ["classical", "lite-DL", "linear (lagged)", "instance", "kernel", "trees / boosting", "CSE (ours)"]
    for f in fam_rows:
        cells = [f if f != "CSE (ours)" else r"\textbf{CSE (ours)}"]
        for ds in DATASETS:
            v = med.at[f, ds] if (f in med.index and ds in med.columns) else np.nan
            s = "--" if np.isnan(v) else f"{v:.3f}$\\times$" if v < 10 else f"{v:.2f}$\\times$"
            if f == "CSE (ours)" and ds not in done and s != "--":
                s = rf"\pending{{{s}}}"
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    tr = med.loc["trees / boosting"]
    N["treeDegMin"] = f"{tr.min():.2f}"
    N["treeDegMinDs"] = DS_LABEL[tr.idxmin()]
    N["treeDegMax"] = f"{tr.max():.2f}"
    N["treeDegMaxDs"] = DS_LABEL[tr.idxmax()]
    sup = ratio[~ratio.family.isin(["classical", "lite-DL"])]
    N["supDegMin"] = f"{sup.ratio.min():.2f}"
    N["supDegMax"] = f"{sup.ratio.max():.2f}"
    lead_rec = {}
    tree_models = {m for m, f in FAMILY_GROUP.items() if f == "trees / boosting"}
    for ds in DATASETS:
        r_ = rec.xs(ds, level="Dataset").dropna().sort_values()
        top = r_[(r_ - r_.iloc[0]).abs() <= 5e-4 * max(1.0, abs(r_.iloc[0]))]
        tm = [m for m in top.index if m in tree_models]
        if tm:
            lead_rec[ds] = r_[tm[0]] / one[(tm[0], ds)]
    if not lead_rec:
        lead_rec = {"none": float("nan")}
    N["treeLeadRecDegMin"] = f"{min(lead_rec.values()):.1f}"
    N["treeLeadRecDegMax"] = f"{max(lead_rec.values()):.1f}"

    # rolling-origin ranks
    long = pd.read_csv(results / "multistep_rolling" / "long_all__rolling.csv")
    piv = long.pivot_table(index=["Model", "Dataset"], columns="horizon", values="RMSE")
    piv = piv.unstack("Dataset")
    models = [m for m in MODEL_ORDER if m in piv.index]
    piv = piv.loc[models].dropna(how="any")
    N["nRollingModels"] = str(len(piv))
    rank_h = {}
    for h in HORIZONS:
        block = piv.xs(h, axis=1, level="horizon")
        rank_h[h] = block.rank(axis=0, method="average").mean(axis=1)
    rk = pd.DataFrame(rank_h)
    lines += [r"\midrule", r"\multicolumn{6}{c}{\emph{Rolling-origin mean rank by horizon, by model}} \\",
              r"Model & $h$=1 & $h$=3 & $h$=6 & $h$=12 & $h$=24 \\", r"\midrule"]
    for m in rk.index:
        cells = [disp(m)] + [f"{rk.at[m, h]:.1f}" for h in HORIZONS]
        lines.append(" & ".join(cells) + r" \\")
    rk["family"] = [FAMILY_GROUP[m] for m in rk.index]
    fam = rk.groupby("family")[HORIZONS].mean()
    lines += [r"\midrule", r"\multicolumn{6}{c}{\emph{\ldots and by family}} \\",
              r"Family & $h$=1 & $h$=3 & $h$=6 & $h$=12 & $h$=24 \\", r"\midrule"]
    for f in FAMILY_ORDER:
        if f not in fam.index:
            continue
        cells = [f if f != "CSE (ours)" else r"\textbf{CSE (ours)}"] + [f"{fam.at[f, h]:.2f}" for h in HORIZONS]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_family.tex").write_text("\n".join(lines) + "\n")
    for f, key in (("trees / boosting", "tree"), ("lite-DL", "lite"), ("linear (lagged)", "lin"), ("CSE (ours)", "cse")):
        for h in (1, 24):
            N[f"rank{key}H{h}"] = f"{fam.at[f, h]:.2f}"
    N["rankcseH1model"] = f"{rk.at[CSE_RAW, 1]:.1f}"
    N["rankcseH24model"] = f"{rk.at[CSE_RAW, 24]:.1f}"
    bestH1 = rk[1].idxmin()
    N["bestRankH1"] = disp(bestH1)
    N["bestRankH1Val"] = f"{rk[1].min():.1f}"
    # per-dataset winners at h=1 and h=24
    tree_models = {m for m, f in FAMILY_GROUP.items() if f == "trees / boosting"}
    for h in (1, 24):
        wins = []
        n_tree = 0
        for ds in DATASETS:
            col = piv.xs(h, axis=1, level="horizon")[ds]
            wins.append(f"{disp(col.idxmin())} on {DS_LABEL[ds]}")
            n_tree += int(col.idxmin() in tree_models)
        N[f"winnersH{h}"] = ", ".join(wins)
        N[f"nTreeWinnersH{h}"] = str(n_tree)


def tab_weights(wdf: pd.DataFrame, masses: dict, out: Path, N: dict) -> None:
    piv = wdf.pivot(index="member", columns="dataset", values="weight")
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Base learner & UCI & OPSD-H & OPSD-N & ETTh1 & ETTh2 \\", r"\midrule"]
    for m in ["sarima", "hw", "lgbm", "catboost", "extratrees"]:
        cells = [MEMBER_LABEL[m]]
        for ds in DATASETS:
            v = piv.at[m, ds]
            if v <= 1e-9:
                cells.append(r"\textbf{0}")
            elif v < 1e-3:
                e = int(np.floor(np.log10(v)))
                cells.append(rf"${v/10**e:.1f}{{\times}}10^{{{e}}}$")
            else:
                cells.append(f"{v:.3f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\midrule", "mass on tree members & " + " & ".join(f"{masses[ds]:.3f}" for ds in DATASETS) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    (out / "tab_weights.tex").write_text("\n".join(lines) + "\n")
    for ds in DATASETS:
        N[f"mass{ds}"] = f"{masses[ds]:.3f}"
        for m in ["sarima", "hw", "lgbm", "catboost", "extratrees"]:
            N[f"w{m}{ds}"] = f"{piv.at[m, ds]:.3f}"
    n_one = sum(1 for ds in DATASETS if round(masses[ds], 3) >= 1.0)
    N["nMassOne"] = str(n_one)
    N["massMin"] = f"{min(masses.values()):.3f}"
    cp = []
    for ds in DATASETS:
        for m in ("sarima", "hw"):
            v = piv.at[m, ds]
            if v > 1e-9:
                e = int(np.floor(np.log10(v)))
                vs = f"{v:.3f}" if v >= 1e-3 else rf"${v/10**e:.1f}\times10^{{{e}}}$"
                cp.append(f"{MEMBER_LABEL[m]} on {DS_LABEL[ds]} ({vs})")
    N["nClassicalPositive"] = str(len(cp))
    N["classicalPositiveList"] = "; ".join(cp) or "none"
    N["massOPSDN"] = f"{masses['OPSD_DE_total']:.4f}"
    v = piv.at["sarima", "OPSD_DE_total"]
    e = int(np.floor(np.log10(v)))
    N["wSarimaOPSDNsci"] = rf"${v/10**e:.1f}\times10^{{{e}}}$"
    N["wHwOPSDH"] = f"{piv.at['hw', 'OPSD_DE_household']:.3f}"


def tab_agree(bdf: pd.DataFrame, out: Path, N: dict) -> None:
    lines = [r"\begin{tabular}{llrrrlrl}", r"\toprule",
             r"Data & Member & $\sigma_i/\sigma_b$ & $\rho_i$ & thr. & Pred. & $w_i$ & Agree \\", r"\midrule"]
    for ds in DATASETS:
        for _, r in bdf[bdf.dataset == ds].iterrows():
            w = r.w
            ws = r"0" if w <= 1e-9 else (f"{w:.3f}" if w >= 1e-3 else rf"${w/10**int(np.floor(np.log10(w))):.1f}{{\times}}10^{{{int(np.floor(np.log10(w)))}}}$")
            ratio = f"{r.ratio:.3f}" if r.ratio < 100 else f"{r.ratio:.1f}"
            rho = f"{r.rho:.3f}".replace("-", "$-$")
            lines.append(f"{DS_LABEL[ds]} & {MEMBER_LABEL[r.member]} & {ratio} & {rho} & {r.thr:.3f} & {r.pred} & {ws} & {r.agree} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_agree.tex").write_text("\n".join(lines) + "\n")
    nl = len(bdf)
    excl = bdf[bdf.pred == "excl."]
    adm = bdf[bdf.pred == "adm."]
    N["nNonLeading"] = str(nl)
    N["nExcluded"] = str(len(excl))
    N["nViolations"] = str(int((excl.w > 1e-9).sum()))
    N["nAdmissible"] = str(len(adm))
    N["nAdmPositive"] = str(int((adm.w > 1e-9).sum()))
    N["nAdmZero"] = str(int((adm.w <= 1e-9).sum()))
    N["nAgree"] = str(nl - int((adm.w <= 1e-9).sum()) - int((excl.w > 1e-9).sum()))
    coll = adm[adm.w <= 1e-9]
    N["collList"] = ", ".join(f"{MEMBER_LABEL[r.member]} on {DS_LABEL[r.dataset]}" for _, r in coll.iterrows())
    N["nKKTok"] = str(int(((bdf.kkt_rel >= -1e-6) | (bdf.w > 1e-9)).sum()))
    s = bdf[(bdf.dataset == "OPSD_DE_total") & (bdf.member == "sarima")].iloc[0]
    N["sarimaOPSDNratio"] = f"{s.ratio:.0f}"
    N["sarimaOPSDNrho"] = f"{s.rho:.3f}".replace("-", "$-$")
    N["sarimaOPSDNthr"] = f"{s.thr:.3f}"
    # smallest margin among collinear cases
    if len(coll):
        N["collMaxGap"] = f"{(coll.thr - coll.rho).max():.3f}"
    hw = bdf[(bdf.dataset == "OPSD_DE_total") & (bdf.member == "hw")].iloc[0]
    N["hwOPSDNrho"] = f"{hw.rho:.3f}"
    N["hwOPSDNthr"] = f"{hw.thr:.3f}"
    N["hwOPSDNkkt"] = f"{hw.kkt_rel:.2f}"


def tab_dm(results: Path, lb: pd.DataFrame, done: set, out: Path, N: dict) -> None:
    lines = [r"\begin{tabular}{lllrrrrcllc}", r"\toprule",
             r"Data & Protocol & Best model & RMSE (best) & RMSE (CSE) & DM stat & $p_{\mathrm{adj}}$ & $d$ / $\delta$ & Verdict & Beats & Ties \\",
             r"\midrule"]
    verdict_counts = {"one_step": {}, "recursive": {}}
    for ds in DATASETS:
        for proto, plabel in (("one_step", "one-step"), ("recursive", "recursive")):
            path = results / "significance" / f"dm_matrix__{ds}__{proto}.csv"
            dm = pd.read_csv(path)
            ref = dm.reference.iloc[0]
            n_models = len(dm)
            beats = int(dm.significant_5pct.sum())
            ties = n_models - beats
            ref_disp = disp(ref.replace("_", " ")) if ref != CSE_SAFE else r"\textbf{CSE (ours)}"
            pend = (lambda s: rf"\pending{{{s}}}") if ds not in done else (lambda s: s)
            twin = None
            if ref == CSE_SAFE:
                # a model whose forecasts equal the CSE's (complete collapse)
                same = dm[(dm.rmse_model - dm.rmse_ref).abs() <= 1e-9 * max(1.0, abs(dm.rmse_ref.iloc[0]))]
                if len(same):
                    twin = same.iloc[0]
            if ref == CSE_SAFE and twin is not None:
                tname = disp(str(twin.model).replace("_", " "))
                verdict_counts[proto][ds] = "identical"
                N[f"identModel{ds}{'One' if proto=='one_step' else 'Rec'}"] = tname
                row = [DS_LABEL[ds], plabel, tname, fnum(twin.rmse_ref), fnum(twin.rmse_model),
                       "$+$0.00", "1.000", "-- / $+$0.000", "identical",
                       f"{beats} / {n_models}", str(ties)]
            elif ref == CSE_SAFE:
                row = [DS_LABEL[ds], plabel, ref_disp, "--", "--", "--", "--", "--", "CSE is best",
                       f"{beats} / {n_models}", str(ties)]
                verdict_counts[proto][ds] = "best"
            else:
                r = dm[dm.model == CSE_SAFE].iloc[0]
                if abs(r.dm_stat) < 1e-9 and abs(r.rmse_model - r.rmse_ref) < 1e-9:
                    verdict = "identical"
                elif r.significant_5pct:
                    verdict = "CSE worse"
                else:
                    verdict = "tie"
                verdict_counts[proto][ds] = verdict
                stat = f"{r.dm_stat:+.2f}".replace("-", "$-$").replace("+", "$+$")
                dd = ("--" if verdict == "identical" else f"{r.cohens_d:.3f}".replace("-", "$-$"))
                dl = f"{r.cliffs_delta:+.3f}".replace("-", "$-$").replace("+", "$+$")
                row = [DS_LABEL[ds], plabel, ref_disp, fnum(r.rmse_ref), pend(fnum(r.rmse_model)),
                       pend(stat), pend(fpcell(r.p_adj)), pend(f"{dd} / {dl}"), pend(verdict),
                       f"{beats} / {n_models}", str(ties)]
                N[f"dm{ds}{'One' if proto=='one_step' else 'Rec'}p"] = fp(r.p_adj)
                # used inside math in the text, so a plain minus sign
                N[f"dm{ds}{'One' if proto=='one_step' else 'Rec'}d"] = f"{r.cohens_d:.3f}"
                N[f"dm{ds}{'One' if proto=='one_step' else 'Rec'}delta"] = f"{r.cliffs_delta:.3f}"
                N[f"dm{ds}{'One' if proto=='one_step' else 'Rec'}gap"] = f"{100*(r.rmse_model/r.rmse_ref-1):.1f}"
            N[f"beats{ds}{'One' if proto=='one_step' else 'Rec'}"] = f"{beats} of {n_models}"
            N[f"nBeats{ds}{'One' if proto=='one_step' else 'Rec'}"] = str(beats)
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_dm.tex").write_text("\n".join(lines) + "\n")
    for proto, key in (("one_step", "One"), ("recursive", "Rec")):
        vc = verdict_counts[proto]
        clauses = {"tie": [], "CSE worse": []}
        info = []
        for ds in DATASETS:
            v = vc.get(ds)
            if v in clauses:
                k2 = f"dm{ds}{key}"
                extra = f"$p_{{\\mathrm{{adj}}}} {N[k2 + 'p']}$, $d = {N[k2 + 'd']}$, $\\delta = {N[k2 + 'delta']}$"
                clauses[v].append(f"{DS_LABEL[ds]} ({extra})")
                info.append((ds, float(N[k2 + "gap"]), v == "CSE worse"))
        N[f"tieClauses{key}"] = "; ".join(clauses["tie"]) or "none"
        N[f"lossClauses{key}"] = "; ".join(clauses["CSE worse"]) or "none"
        note = "Here they do: the larger the RMSE gap, the stronger the test's verdict."
        for ds_a, gap_a, sig_a in info:
            for ds_b, gap_b, sig_b in info:
                if (not sig_a) and sig_b and gap_a > gap_b:
                    note = (f"On {DS_LABEL[ds_a]} the CSE trails the best model by {gap_a:.1f}\\% and cannot be "
                            f"separated from it, while on {DS_LABEL[ds_b]} it trails by {gap_b:.1f}\\% and is "
                            f"separated at the 5\\% level.")
        N[f"orderingNote{key}"] = note
        times = {0: "no times", 1: "once", 2: "twice", 3: "three times", 4: "four times", 5: "five times"}
        N[f"tiesTimes{key}"] = times[sum(v == "tie" for v in vc.values())]
        N[f"lossTimes{key}"] = times[sum(v == "CSE worse" for v in vc.values())]
        N[f"identTimes{key}"] = times[sum(v == "identical" for v in vc.values())]
        N[f"identList{key}"] = ", ".join(DS_LABEL[d] for d, v in vc.items() if v == "identical") or "none"
        N[f"nWins{key}"] = str(sum(v == "best" for v in vc.values()))
        N[f"nTies{key}"] = str(sum(v == "tie" for v in vc.values()))
        N[f"nLoss{key}"] = str(sum(v == "CSE worse" for v in vc.values()))
        N[f"nIdent{key}"] = str(sum(v == "identical" for v in vc.values()))
        N[f"winList{key}"] = ", ".join(DS_LABEL[d] for d, v in vc.items() if v == "best") or "none"
        N[f"tieList{key}"] = ", ".join(DS_LABEL[d] for d, v in vc.items() if v == "tie") or "none"
        N[f"lossList{key}"] = ", ".join(DS_LABEL[d] for d, v in vc.items() if v == "CSE worse") or "none"
    fr = pd.read_csv(results / "significance" / "friedman_summary.csv")
    for _, r in fr.iterrows():
        key = "One" if r.protocol == "one_step" else "Rec"
        N[f"friedChi{key}"] = f"{r.friedman_stat:.2f}"
        N[f"friedP{key}"] = fp(r.p_value)
        N[f"friedK{key}"] = str(int(r.n_models))


def tab_tunedcost(results: Path, lb: pd.DataFrame, done: set, out: Path, N: dict) -> None:
    one = lb[lb.Protocol == "one_step"]
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Data & Model & Lookback & Clip $q$ & Val RMSE & Test RMSE & Test $R^2$ \\", r"\midrule"]
    for ds in DATASETS:
        for m in ("NLinear", "DLinear"):
            p = results / "tuning" / "lite_dl" / f"{ds}__{m}__best.json"
            if not p.exists():
                continue
            j = json.loads(p.read_text())
            row = one[(one.Dataset == ds) & (one.Model == m)].iloc[0]
            r2 = f"{row.R2:.3f}".replace("-", "$-$")
            lines.append(f"{DS_LABEL[ds]} & {m} & {j['best_params']['lookback']} & {j['best_params'].get('clip_quantile', '')} & {fnum(j['best_value'])} & {fnum(row.RMSE)} & {r2} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_tuned.tex").write_text("\n".join(lines) + "\n")
    cost_models = ["SARIMA", "Holt-Winters", "ExtraTrees", "LightGBM", "CatBoost",
                   "HistGradientBoosting", "NLinear", "DLinear", CSE_RAW]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Model & UCI & OPSD-H & OPSD-N & ETTh1 & ETTh2 \\", r"\midrule"]
    tt = one.pivot_table(index="Model", columns="Dataset", values="TrainTime(s)", aggfunc="first")
    for m in cost_models:
        cells = [disp(m)]
        for ds in DATASETS:
            v = tt.at[m, ds] if (m in tt.index and ds in tt.columns) else np.nan
            s = "--" if np.isnan(v) else (f"{v:.3f}" if v < 10 else f"{v:.1f}" if v < 1000 else f"{v:.0f}")
            if m == CSE_RAW and ds not in done and s != "--":
                s = rf"\pending{{{s}}}"
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_cost.tex").write_text("\n".join(lines) + "\n")
    cse = tt.loc[CSE_RAW]
    hgb = tt.loc["HistGradientBoosting"]
    lgb = tt.loc["LightGBM"]
    N["cseTimeMin"] = f"{cse.min():.0f}"
    N["cseTimeMax"] = f"{cse.max():.0f}"
    N["hgbTimeMax"] = f"{hgb.max():.2f}"
    N["costRatioLgbMin"] = f"{(cse/lgb).min():.0f}"
    N["costRatioLgbMax"] = f"{(cse/lgb).max():.0f}"
    N["costRatioHgbMin"] = f"{(cse/hgb).min():.0f}"
    N["costRatioHgbMax"] = f"{(cse/hgb).max():.0f}"
    # refit time saved
    p = results / "tables" / "cse_refit_times.csv"
    if p.exists():
        rt = pd.read_csv(p).set_index("dataset")
        for ds in DATASETS:
            if ds in rt.index:
                N[f"cseFit{ds}"] = f"{rt.at[ds, 'fit_time_s']:.0f}"
                sk = str(rt.at[ds, "skipped_refits"])
                sk = "none" if sk in ("nan", "") else ", ".join(MEMBER_LABEL.get(x, x) for x in sk.split(";"))
                N[f"cseSkipped{ds}"] = sk
    lite = one[one.Model.isin(["NLinear", "DLinear"]) & one.Dataset.isin(["ETTh1", "ETTh2"])]
    N["liteR2ETTmin"] = f"{lite.R2.min():.2f}".replace("-", "$-$")
    N["liteR2ETTmax"] = f"{lite.R2.max():.2f}".replace("-", "$-$")
    N["dlinearUCI"] = fnum(one[(one.Model == "DLinear") & (one.Dataset == "UCI")].RMSE.iloc[0])


def tab_sensitivity(results: Path, out: Path, N: dict) -> None:
    p = results / "tables" / "membership_sensitivity.csv"
    if not p.exists():
        (out / "tab_sensitivity.tex").write_text("% membership_sensitivity.csv not found\n")
        for k in ("sensRows", "sensViol", "sensGapBetter"):
            N[k] = "0"
        for k in ("sensDatasets", "sensTreeMassMin", "sensZeroFracMin", "sensZeroFracMax", "sensAllK",
                  "sensAllActiveMax", "sensAllActiveMin", "sensGapMin", "sensGapMax"):
            N[k] = "n/a"
        return
    df = pd.read_csv(p)
    pools = ["paper5", "paper5+xgb_rf", "paper5+linear", "paper5+lite_dl", "trees8", "all"]
    plabel = {"paper5": "paper five", "paper5+xgb_rf": "+ XGBoost, RF", "paper5+linear": "+ Ridge, Lasso, EN",
              "paper5+lite_dl": "+ NLinear, DLinear", "trees8": "eight tree models", "all": "all standalone"}
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Data & Pool & $K$ & zeroed & tree mass & viol. & RMSE blend & RMSE best member & gap \\", r"\midrule"]
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        for pool in pools:
            r = sub[sub.pool == pool]
            if r.empty:
                continue
            r = r.iloc[0]
            gap = 100 * (r.test_rmse_blend / r.test_rmse_best_active_member - 1)
            gs = f"{gap:+.1f}\\%".replace("-", "$-$").replace("+", "$+$")
            lines.append(f"{DS_LABEL[ds]} & {plabel[pool]} & {int(r.K)} & {int(r.n_zero)} & {r.tree_mass:.3f} & {int(r.necessity_violations)} & {fnum(r.test_rmse_blend)} & {fnum(r.test_rmse_best_active_member)} ({r.best_active_member}) & {gs} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_sensitivity.tex").write_text("\n".join(lines) + "\n")
    N["sensRows"] = str(len(df))
    N["sensDatasets"] = ", ".join(DS_LABEL[d] for d in DATASETS if d in set(df.dataset))
    N["sensViol"] = str(int(df.necessity_violations.sum()))
    N["sensViolTimes"] = _times(int(df.necessity_violations.sum()))
    N["sensTreeMassMin"] = f"{df.tree_mass.min():.3f}"
    N["sensZeroFracMin"] = f"{(df.n_zero / df.K).min()*100:.0f}"
    N["nSensAnyZero"] = str(int((df.n_zero > 0).sum()))
    nz = df[df.n_zero == 0]
    N["sensNoZeroDesc"] = "; ".join(f"{DS_LABEL[r.dataset]} with {r.pool.replace('paper5', 'the paper five').replace('+', ' + ')} ($K = {int(r.K)}$)"
                                    for _, r in nz.iterrows()) or "none"

    N["sensZeroFracMax"] = f"{(df.n_zero / df.K).max()*100:.0f}"
    big = df[df.pool == "all"]
    if len(big):
        N["sensAllZeroMin"] = str(int(big.n_zero.min()))
        N["sensAllZeroMax"] = str(int(big.n_zero.max()))
        N["sensAllKMin"] = str(int(big.K.min()))
    if len(big):
        N["sensAllK"] = str(int(big.K.max()))
        N["sensAllActiveMax"] = str(int(big.n_active.max()))
        N["sensAllActiveMin"] = str(int(big.n_active.min()))
        N["sensAllDesc"] = "; ".join(
            f"{DS_LABEL[r.dataset]}: {int(r.n_active)} active, largest weight {r.max_weight:.2f} on {disp(r["argmax"])}, "
            f"tree mass {r.tree_mass:.2f}" for _, r in big.iterrows())
    else:
        N["sensAllDesc"] = "n/a"
    plabel = {"paper5": "paper five", "paper5+xgb_rf": "+ XGBoost, RF", "paper5+linear": "+ Ridge, Lasso, EN",
              "paper5+lite_dl": "+ NLinear, DLinear", "trees8": "eight tree models", "all": "all standalone models"}
    low = df[df.tree_mass < 0.5]
    N["nSensLowTree"] = str(len(low))
    N["sensLowTreeList"] = "; ".join(
        f"{DS_LABEL[r.dataset]} with {plabel[r.pool]} (tree mass {r.tree_mass:.2f}, largest weight on {disp(r["argmax"])})"
        for _, r in low.iterrows()) or "none"
    N["nSensHighTree"] = str(int((df.tree_mass >= 0.5).sum()))
    N["sensActiveMax"] = str(int(df.n_active.max()))
    gaps = 100 * (df.test_rmse_blend / df.test_rmse_best_active_member - 1)
    N["sensGapMin"] = f"{gaps.min():+.1f}".replace("-", "$-$").replace("+", "$+$")
    N["sensGapMax"] = f"{gaps.max():+.1f}".replace("-", "$-$").replace("+", "$+$")
    N["sensGapBetter"] = str(int((gaps < -0.05).sum()))


def tab_folds(results: Path, out: Path, N: dict) -> None:
    p = results / "tables" / "cse_folds.csv"
    if not p.exists():
        (out / "tab_folds.tex").write_text("% cse_folds.csv not found\n")
        for k in ("foldRows", "foldViol", "foldBetter", "foldWorse"):
            N[k] = "0"
        for k in ("foldDatasets", "foldTreeMassMin", "foldGapMin", "foldGapMax"):
            N[k] = "n/a"
        return
    df = pd.read_csv(p).sort_values(["dataset", "train_end_frac"])
    lines = [r"\begin{tabular}{lrrrrrrrl}", r"\toprule",
             r"Data & train end & $n_{\mathrm{test}}$ & tree mass & zeroed & viol. & RMSE CSE & RMSE best member & best member \\", r"\midrule"]
    for ds in DATASETS:
        for _, r in df[df.dataset == ds].iterrows():
            lines.append(f"{DS_LABEL[ds]} & {int(round(100*r.train_end_frac))}\\% & {int(r.n_test):,} & {r.tree_mass:.3f} & {int(r.n_zero)} & {int(r.necessity_violations)} & {fnum(r.test_rmse_cse)} & {fnum(r.test_rmse_best_member)} & {MEMBER_LABEL[r.best_member]} \\\\".replace(",", r"\,"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_folds.tex").write_text("\n".join(lines) + "\n")
    N["foldRows"] = str(len(df))
    full = {}
    for _, r in df[np.isclose(df.train_end_frac, 0.8)].iterrows():
        N[f"cseFullFit{r.dataset}"] = f"{r.fit_time_s:.0f}"   # all five members refit
        full[r.dataset] = float(r.fit_time_s)
    rp = results / "tables" / "cse_refit_times.csv"
    if rp.exists() and full:
        rt = pd.read_csv(rp).set_index("dataset")
        rows = []
        for ds in DATASETS:
            if ds in rt.index and ds in full:
                save = 100.0 * (1.0 - float(rt.at[ds, "fit_time_s"]) / full[ds])
                rows.append((ds, save, float(rt.at[ds, "fit_time_s"]), full[ds]))
        real = [r for r in rows if r[1] >= 5.0]
        N["nRefitSaved"] = str(len(real))
        N["nRefitSavedWord"] = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "all five"}[len(real)]
        N["refitSaveMin"] = f"{min(r[1] for r in real):.0f}" if real else "0"
        N["refitSaveMax"] = f"{max(r[1] for r in real):.0f}" if real else "0"
        none = [r for r in rows if r[1] < 5.0]
        N["refitNoSaveList"] = ", ".join(DS_LABEL[r[0]] for r in none) or "none"
        lines = [r"\begin{tabular}{lrrrp{3.1cm}}", r"\toprule",
                 r"Data & active (s) & full (s) & saving & members not refit \\", r"\midrule"]
        for ds, sv, a, f in rows:
            sk = str(rt.at[ds, "skipped_refits"])
            sk = "none" if sk in ("nan", "") else ", ".join(MEMBER_LABEL.get(x, x) for x in sk.split(";"))
            svs = f"{sv:.0f}\\%" if sv >= 5.0 else "none"
            lines.append(f"{DS_LABEL[ds]} & {a:.0f} & {f:.0f} & {svs} & {sk} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (out / "tab_refit.tex").write_text("\n".join(lines) + "\n")
    N["foldDatasets"] = ", ".join(DS_LABEL[d] for d in DATASETS if d in set(df.dataset))
    N["foldViol"] = str(int(df.necessity_violations.sum()))
    N["foldViolTimes"] = _times(int(df.necessity_violations.sum()))
    N["foldTreeMassMin"] = f"{df.tree_mass.min():.3f}"
    N["foldGapMin"] = f"{df.cse_minus_best_pct.min():+.1f}".replace("-", "$-$").replace("+", "$+$")
    N["foldGapMax"] = f"{df.cse_minus_best_pct.max():+.1f}".replace("-", "$-$").replace("+", "$+$")
    N["foldBetter"] = str(int((df.cse_minus_best_pct < -0.05).sum()))
    N["foldWorse"] = str(int((df.cse_minus_best_pct > 0.05).sum()))
    # describe each screen violation, if the weight file carries the geometry
    desc = []
    wp = results / "tables" / "cse_folds_weights.csv"
    if wp.exists():
        wdf = pd.read_csv(wp)
        if {"rho_vs_best", "rmse_ratio_vs_best"} <= set(wdf.columns):
            for _, r in wdf[~wdf.is_best.astype(bool)].iterrows():
                thr = 1.0 / r.rmse_ratio_vs_best
                if r.rho_vs_best >= thr and r.weight > 1e-9:
                    wv = r.weight
                    ws = f"{wv:.3f}" if wv >= 1e-3 else f"{wv/10**int(np.floor(np.log10(wv))):.1f}\\times10^{{{int(np.floor(np.log10(wv)))}}}"
                    desc.append(f"{MEMBER_LABEL[r.member]} on {DS_LABEL[r.dataset]} with training ending at "
                                f"{int(round(100*r.train_end_frac))}\\% ($\\rho = {r.rho_vs_best:.3f}$ against "
                                f"{thr:.3f}, $w = {ws}$)")
    N["foldViolDesc"] = "; ".join(desc) if desc else "none"


def tab_app(results: Path, done: set, out: Path) -> None:
    long = pd.read_csv(results / "multistep_rolling" / "long_all__rolling.csv")
    piv = long.pivot_table(index="Model", columns=["Dataset", "horizon"], values="RMSE")
    models = [m for m in MODEL_ORDER if m in piv.index]
    header = [r"\begin{tabular}{l" + "r" * 25 + "}", r"\toprule",
              "& " + " & ".join(rf"\multicolumn{{5}}{{c}}{{{DS_LABEL[ds]}}}" for ds in DATASETS) + r" \\",
              "".join(rf"\cmidrule(lr){{{2+5*i}-{6+5*i}}}" for i in range(5)),
              "Model & " + " & ".join(" & ".join(str(h) for h in HORIZONS) for _ in DATASETS) + r" \\", r"\midrule"]
    best = {(ds, h): piv[(ds, h)].min() for ds in DATASETS for h in HORIZONS}
    lines = []
    for m in models:
        cells = [disp(m)]
        for ds in DATASETS:
            for h in HORIZONS:
                v = piv.at[m, (ds, h)] if (ds, h) in piv.columns else np.nan
                if np.isnan(v):
                    cells.append("--")
                    continue
                s = f"{v:.3f}" if v < 10 else (f"{v:.0f}" if v >= 1000 else f"{v:.2f}")
                if abs(v - best[(ds, h)]) <= 5e-4 * max(1, abs(best[(ds, h)])):
                    s = rf"\textbf{{{s}}}"
                if m == CSE_RAW and ds not in done:
                    s = rf"\pending{{{s}}}"
                cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    (out / "tab_app.tex").write_text("\n".join(header + lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "generated")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lb, done = load(args.results)
    N: dict[str, str] = {}
    N["doneList"] = ", ".join(DS_LABEL[d] for d in DATASETS if d in done) or "none"
    N["pendingList"] = ", ".join(DS_LABEL[d] for d in DATASETS if d not in done) or "none"
    N["nPending"] = str(sum(d not in done for d in DATASETS))
    rr = args.results / "tables" / "OPSD_DE_household_removed_readings.csv"
    if rr.exists():
        rem = pd.read_csv(rr)
        big = rem[rem.reason != "negative step"]
        N["nRemovedNeg"] = str(int((rem.reason == "negative step").sum()))
        N["nRemovedBig"] = str(len(big))
        N["removedBigDesc"] = "; ".join(
            f"{pd.Timestamp(r.timestamp):%d %B %Y}, {r.hourly_step_kwh:.0f}~kWh" for _, r in big.iterrows()) or "none"
        N["opsdhCleaned"] = "1"
    else:
        N["nRemovedNeg"] = N["nRemovedBig"] = "0"
        N["removedBigDesc"] = "none"
        N["opsdhCleaned"] = "0"
    wdf, bdf, masses = diag_tables(args.results)
    tab_main(lb, done, args.out, N)
    tab_family(lb, args.results, done, args.out, N)
    tab_weights(wdf, masses, args.out, N)
    tab_agree(bdf, args.out, N)
    tab_dm(args.results, lb, done, args.out, N)
    tab_tunedcost(args.results, lb, done, args.out, N)
    tab_sensitivity(args.results, args.out, N)
    tab_folds(args.results, args.out, N)
    tab_app(args.results, done, args.out)
    # dataset table numbers
    rows = {"UCI": 8761, "OPSD_DE_household": 35461, "OPSD_DE_total": 50400, "ETTh1": 17420, "ETTh2": 17420}
    for ds in DATASETS:
        n_models = lb[(lb.Dataset == ds) & (lb.Protocol == "one_step")].Model.nunique()
        N[f"nModels{ds}"] = str(n_models)
    with open(args.out / "numbers.tex", "w") as fh:
        fh.write("% generated by scripts/build_paper.py; do not edit by hand\n")
        for k, v in sorted(N.items()):
            safe = "".join(ch for ch in k if ch.isalnum())
            safe = safe.replace("0", "Zero").replace("1", "One").replace("2", "Two").replace("3", "Three").replace("4", "Four").replace("5", "Five").replace("6", "Six").replace("7", "Seven").replace("8", "Eight").replace("9", "Nine")
            fh.write(macro(safe, v))
    print(f"wrote {len(N)} macros and tables to {args.out}")
    print("pending (old solver) datasets:", N["pendingList"])


if __name__ == "__main__":
    main()
