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

"""Re-solve the stacking weights exactly on the simplex from stored errors.

Reads the validation errors saved in ``results/cse_diagnostics/*.pkl`` and,
without refitting any model, reports for every dataset and member:

* the weight stored with the artefacts,
* the exact simplex-constrained weight,
* the uncentred error-scale ratio and correlation against the leader,
* the Proposition 1 verdict (admissible / excluded) and whether it agrees
  with the exact weight,
* the full KKT residual ``((Σw)_i - λ) / λ`` at the exact optimum.

Run::

    python scripts/recheck_simplex_kkt.py
    python scripts/recheck_simplex_kkt.py --out results/tables/kkt_simplex_recheck.csv

This script needs no raw data. It does NOT update test-set scores: those
require ``scripts/rerun_cse.py`` and ``scripts/run_significance_analysis.py``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.kkt_analysis import compute_cse_diagnostics  # noqa: E402
from src.models.novel import solve_simplex_qp  # noqa: E402


def recheck(diag_dir: Path, tol: float = 1e-6) -> pd.DataFrame:
    rows = []
    for path in sorted(diag_dir.glob("*.pkl")):
        with open(path, "rb") as fh:
            old = pickle.load(fh)
        names = list(old.member_names)
        E = np.asarray(old.validation_errors, dtype=float)      # (K, n)
        S = (E @ E.T) / E.shape[1]
        w = solve_simplex_qp(S)
        new = compute_cse_diagnostics(
            {n: E[i] for i, n in enumerate(names)}, w, dataset=old.dataset
        )
        b = new.best_member_idx
        lam = new.kkt_lambda
        for i, n in enumerate(names):
            ratio = new.sigma[i] / new.sigma[b]
            rho = new.rho[i, b]
            thr = 1.0 / ratio
            if i == b:
                pred, agree = "leader", ""
            else:
                pred = "adm." if rho < thr else "excl."
                active = w[i] > tol
                if pred == "excl.":
                    agree = "yes" if not active else "VIOLATION"
                else:
                    agree = "yes" if active else "coll."
            rows.append({
                "dataset": old.dataset,
                "member": n,
                "weight_stored": float(old.weights[i]),
                "weight_simplex": float(w[i]),
                "rmse_ratio_vs_best": float(ratio),
                "rho_vs_best": float(rho),
                "threshold": float(thr),
                "prop1": pred,
                "agree": agree,
                "kkt_residual_rel": float(new.kkt_residuals[i] / lam),
                "kkt_ok": bool(
                    abs(new.kkt_residuals[i] / lam) <= 1e-6 if w[i] > tol
                    else new.kkt_residuals[i] / lam >= -1e-6
                ),
                "val_rmse_stored": float(np.sqrt(old.weights @ S @ old.weights)),
                "val_rmse_simplex": float(np.sqrt(lam)),
            })
    return pd.DataFrame(rows)


def update_artefacts(diag_dir: Path) -> None:
    """Rewrite the diagnostics pickles and the weight CSV from stored errors."""
    rows = []
    for path in sorted(diag_dir.glob("*.pkl")):
        with open(path, "rb") as fh:
            old = pickle.load(fh)
        names = list(old.member_names)
        E = np.asarray(old.validation_errors, dtype=float)
        S = (E @ E.T) / E.shape[1]
        w = solve_simplex_qp(S)
        new = compute_cse_diagnostics({n: E[i] for i, n in enumerate(names)}, w,
                                      dataset=old.dataset, extra=dict(old.extra))
        with open(path, "wb") as fh:
            pickle.dump(new, fh)
        rows.append(new.to_frame()[["dataset", "member", "weight", "sigma", "rho_vs_best",
                                    "is_best", "kkt_residual"]])
        print(f"rewrote {path}")
    csv = diag_dir.parent / "cse_weights_by_dataset.csv"
    pd.concat(rows, ignore_index=True).to_csv(csv, index=False)
    print(f"rewrote {csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inp", default=str(ROOT / "results" / "cse_diagnostics"))
    ap.add_argument("--out", default=None, help="optional CSV path")
    ap.add_argument("--update-artefacts", action="store_true",
                    help="rewrite results/cse_diagnostics/*.pkl and "
                         "results/cse_weights_by_dataset.csv with the exact weights and "
                         "uncentred moments (test-set rows are untouched)")
    args = ap.parse_args()

    if args.update_artefacts:
        update_artefacts(Path(args.inp))
    df = recheck(Path(args.inp))
    non_lead = df[df.prop1 != "leader"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(df.drop(columns=["val_rmse_stored", "val_rmse_simplex"]).to_string(index=False))
    print()
    print(f"non-leading pairs      : {len(non_lead)}")
    print(f"excluded, weight zero  : {(non_lead.prop1 == 'excl.').sum()} "
          f"(violations: {(non_lead.agree == 'VIOLATION').sum()})")
    print(f"admissible, weight > 0 : {((non_lead.prop1 == 'adm.') & (non_lead.agree == 'yes')).sum()} "
          f"of {(non_lead.prop1 == 'adm.').sum()}")
    print(f"admissible, zeroed     : {(non_lead.agree == 'coll.').sum()}")
    print(f"full KKT system holds  : {int(df.kkt_ok.sum())} of {len(df)} members")
    trees = {"lgbm", "catboost", "extratrees"}
    mass = df[df.member.isin(trees)].groupby("dataset").weight_simplex.sum()
    print("tree mass (simplex)    :", {k: round(v, 4) for k, v in mass.items()})
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
