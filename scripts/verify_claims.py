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

"""Assert headline paper claims against files under ``results/``.

Fails loudly if a number drifts. Run::

    python scripts/verify_claims.py
    make verify
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
TREES = {"lgbm", "catboost", "extratrees"}
CSE_NAMES = {"FAHE (Ours)", "CSE (Ours)", "FAHE_(Ours)", "CSE_(Ours)"}


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def check_tree_masses(weights: pd.DataFrame) -> None:
    masses = {}
    for ds, g in weights.groupby("dataset"):
        masses[ds] = float(g.loc[g.member.isin(TREES), "weight"].sum())
    # Exact simplex solver (v1.1): four series at 1.000 to three decimals
    # (OPSD-N keeps SARIMA at about 5.5e-5), OPSD-H at 0.965.
    near_one = [ds for ds, m in masses.items() if round(m, 3) == 1.0]
    household = masses.get("OPSD_DE_household")
    if len(near_one) != 4:
        _fail(f"expected 4 datasets with tree mass ≈1.000, got {masses}")
    if household is None or abs(household - 0.965) > 5e-3:
        _fail(f"OPSD household tree mass expected ≈0.965, got {household}")
    # Exact single-member reproduce on ETTh2
    etth2 = weights[weights.dataset == "ETTh2"].set_index("member")["weight"]
    if abs(float(etth2.get("lgbm", 0.0)) - 1.0) > 1e-9:
        _fail(f"ETTh2 should collapse to LightGBM weight 1.0, got {dict(etth2)}")
    print(f"[ok] tree masses: { {k: round(v, 6) for k, v in masses.items()} }")


def check_boundary(weights: pd.DataFrame) -> None:
    exact = 0
    admissible = []
    positive_admissible = []
    for ds, g in weights.groupby("dataset"):
        best = g[g.is_best].iloc[0]
        sig_b = float(best.sigma)
        for _, r in g.iterrows():
            if bool(r.is_best):
                continue
            thr = sig_b / float(r.sigma)
            rho = float(r.rho_vs_best)
            predicted_zero = rho >= thr
            actual_zero = abs(float(r.weight)) < 1e-12
            if predicted_zero == actual_zero:
                exact += 1
            if not predicted_zero:
                admissible.append((ds, r.member))
                if float(r.weight) > 1e-12:
                    positive_admissible.append((ds, r.member, float(r.weight)))
    if exact != 17:
        _fail(f"boundary exact agreement expected 17/20, got {exact}/20")
    if len(admissible) != 11:
        _fail(f"admissible non-leaders expected 11, got {len(admissible)}")
    if len(positive_admissible) != 8:
        _fail(
            f"positive-weight admissible expected 8, got {len(positive_admissible)}"
        )
    print("[ok] boundary 17/20 exact; 11 admissible; 8 positive")


def check_opsd_sarima_ratio(weights: pd.DataFrame) -> None:
    # sigma is the uncentred validation RMSE since v1.1 (was the centred std, ratio 243)
    g = weights[weights.dataset == "OPSD_DE_total"].set_index("member")
    ratio = float(g.loc["sarima", "sigma"]) / float(g.loc["lgbm", "sigma"])
    if abs(ratio - 508) > 2.0:
        _fail(f"OPSD-N SARIMA/leader sigma ratio expected ≈508, got {ratio}")
    print(f"[ok] OPSD-N SARIMA sigma ratio = {ratio:.3f}")


def check_friedman(path: Path) -> None:
    """The exact statistics move slightly with the CSE rows; the paper reads
    them from the file through scripts/build_paper.py. Here we check only
    the qualitative claim: rejected one step ahead, not rejected recursively."""
    fr = pd.read_csv(path)
    one = fr[fr.protocol == "one_step"].iloc[0]
    rec = fr[fr.protocol == "recursive"].iloc[0]
    if float(one.p_value) >= 1e-3:
        _fail(f"Friedman one_step p expected <1e-3, got {one.p_value}")
    if float(rec.p_value) <= 0.05:
        _fail(f"Friedman recursive p expected >0.05, got {rec.p_value}")
    print(f"[ok] Friedman one_step chi2={one.friedman_stat:.2f} (p={one.p_value:.1e}); "
          f"recursive chi2={rec.friedman_stat:.2f} (p={rec.p_value:.3f})")


def check_mean_rank(lb: pd.DataFrame) -> None:
    one = lb[lb.Protocol == "one_step"]
    counts = one.groupby("Model")["Dataset"].nunique()
    common = sorted(counts[counts == 5].index.tolist())
    if len(common) != 19:
        _fail(f"expected 19 shared models, got {len(common)}")
    ranks = []
    for _, g in one.groupby("Dataset"):
        g = g[g.Model.isin(common)].set_index("Model").loc[common]
        r = rankdata(g["RMSE"].to_numpy(dtype=float), method="average")
        ranks.append(dict(zip(common, r))["FAHE (Ours)"])
    mean_rank = float(np.mean(ranks))
    others = []
    for _, g in one.groupby("Dataset"):
        g = g[g.Model.isin(common)].set_index("Model").loc[common]
        others.append(rankdata(g["RMSE"].to_numpy(dtype=float), method="average"))
    all_mean = np.mean(np.vstack(others), axis=0)
    if mean_rank > all_mean.min() + 1e-9:
        _fail(f"CSE expected the best one-step mean rank, got {mean_rank} vs {all_mean.min()}")
    print(f"[ok] CSE has the best one-step mean rank = {mean_rank:.2f} (ranks={ranks})")


def check_no_transformers(lb: pd.DataFrame) -> None:
    banned = (
        "autoformer", "fedformer", "informer", "itransformer", "patchtst",
        "nhits", "nbeats", "chronos", "timesnet", "tft",
    )
    hit = [
        m for m in lb["Model"].unique()
        if any(b in str(m).lower() for b in banned)
    ]
    if hit:
        _fail(f"Transformer/optional-DL rows must not appear in results: {hit}")
    print("[ok] no Transformer / optional-DL rows in leaderboards")


def check_opsd_model_count(lb: pd.DataFrame) -> None:
    one = lb[lb.Protocol == "one_step"]
    n = one[one.Dataset == "OPSD_DE_total"]["Model"].nunique()
    if n != 19:
        _fail(f"OPSD_DE_total expected 19 models, got {n}")
    print("[ok] OPSD_DE_total has 19 models")


def main() -> None:
    weights = pd.read_csv(RESULTS / "cse_weights_by_dataset.csv")
    lb = pd.read_csv(RESULTS / "leaderboards.csv")
    check_tree_masses(weights)
    check_boundary(weights)
    check_opsd_sarima_ratio(weights)
    check_friedman(RESULTS / "significance" / "friedman_summary.csv")
    check_mean_rank(lb)
    check_no_transformers(lb)
    check_opsd_model_count(lb)
    # Horizon tables must exist under multistep_rolling only for paper use
    rolling = RESULTS / "multistep_rolling" / "long_all__rolling.csv"
    if not rolling.exists():
        _fail(f"missing paper horizon table {rolling}")
    print(f"[ok] rolling-origin table present: {rolling}")
    print("\nAll claims verified.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"VERIFY FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
