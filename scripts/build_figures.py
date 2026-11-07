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

"""Regenerate manuscript figures into ``figures/`` as PNG (300 dpi) and PDF.

Writes ``fig2_kkt_boundary`` and ``fig3_cd_diagrams_*`` from the CSVs and
pickles under ``results/``, keeps the rendered ``fig1_architecture``, and copies
the DM heat maps (``fig5_dm_heatmap__*``, not used by the paper). The files
``fig4_dm_pvalues``, ``fig5_error_drift_uci`` and ``fig6_horizon_ranks`` come
from ``scripts/build_paper_figures.py``. See ``docs/figure_map.md``.

Run::

    python scripts/build_figures.py
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.kkt_analysis import plot_kkt_boundary
from src.evaluation.cd_diagram import plot_cd_diagram
from src.seed import set_global_seed


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {pdf.name} / {png.name}")


def build_kkt(results: Path, out_dir: Path) -> None:
    """Recompute every diagnostic from the stored validation errors so the
    figure always reflects the exact simplex solver and uncentred moments,
    even for pickles written by an older release."""
    from src.evaluation.kkt_analysis import compute_cse_diagnostics
    from src.models.novel import solve_simplex_qp

    diagnostics = []
    order = ["UCI", "OPSD_DE_household", "OPSD_DE_total", "ETTh1", "ETTh2"]
    files = {f.stem: f for f in (results / "cse_diagnostics").glob("*.pkl")}
    for ds in [d for d in order if d in files] + sorted(set(files) - set(order)):
        with files[ds].open("rb") as fh:
            old = pickle.load(fh)
        E = np.asarray(old.validation_errors, dtype=float)
        S = (E @ E.T) / E.shape[1]
        w = solve_simplex_qp(S)
        diagnostics.append(compute_cse_diagnostics(
            {n: E[i] for i, n in enumerate(old.member_names)}, w, dataset=old.dataset))
    fig = plot_kkt_boundary(diagnostics, title=None, annotate=True)
    _save(fig, out_dir / "fig2_kkt_boundary")


def build_cd(results: Path, out_dir: Path) -> None:
    import pandas as pd

    lb = pd.read_csv(results / "leaderboards.csv")
    for protocol in ("one_step", "recursive"):
        sub = lb[lb.Protocol == protocol]
        counts = sub.groupby("Model")["Dataset"].nunique()
        common = counts[counts == sub["Dataset"].nunique()].index.tolist()
        pivot = (
            sub[sub.Model.isin(common)]
            .pivot_table(index="Dataset", columns="Model", values="RMSE")
            [common]
        )
        ranks = pivot.rank(axis=1, ascending=True, method="average")
        ranks = ranks.rename(columns={"FAHE (Ours)": "CSE (ours)"})
        label = "One-step (teacher forced)" if protocol == "one_step" else "Recursive (full test window)"
        fig, _cd = plot_cd_diagram(
            ranks, title=f"{label}: RMSE ranks over five datasets"
        )
        # Combined fig3 uses both; also write per-protocol stems for clarity
        stem = (
            out_dir / "fig3_cd_diagrams_one_step"
            if protocol == "one_step"
            else out_dir / "fig3_cd_diagrams_recursive"
        )
        _save(fig, stem)


def build_architecture(out_dir: Path) -> None:
    """Ensure fig1 comes from TikZ (``paper/fig1_architecture.tex``).

    Prefers an existing shipped pair under ``out_dir``. Otherwise requires
    ``pdflatex`` (+ PNG rasterisation). Never invents a matplotlib substitute.
    """
    tex = ROOT / "paper" / "fig1_architecture.tex"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_out = out_dir / "fig1_architecture.pdf"
    png_out = out_dir / "fig1_architecture.png"

    if pdf_out.exists() and png_out.exists():
        print(
            f"keeping shipped TikZ assets {pdf_out.name} / {png_out.name} "
            f"(re-render with pdflatex if you change {tex.name})"
        )
        return

    if not shutil.which("pdflatex"):
        raise RuntimeError(
            "fig1_architecture.pdf/.png are missing and pdflatex is not on PATH. "
            "Render paper/fig1_architecture.tex with pdflatex (and export a 300 dpi PNG), "
            "or place both files under figures/. Matplotlib fallback is disabled."
        )
    if not tex.exists():
        raise RuntimeError(f"TikZ source missing: {tex}")

    try:
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory",
                str(out_dir),
                str(tex),
            ],
            check=True,
            capture_output=True,
            cwd=str(ROOT / "paper"),
        )
    except Exception as exc:
        raise RuntimeError(
            f"pdflatex failed for {tex.name}: {exc}. "
            "Matplotlib fallback is disabled — fix TeX or ship pre-rendered assets."
        ) from exc

    if not pdf_out.exists():
        raise RuntimeError(f"pdflatex did not produce {pdf_out}")

    if shutil.which("pdftoppm"):
        subprocess.run(
            ["pdftoppm", "-png", "-r", "300", "-singlefile",
             str(pdf_out), str(pdf_out.with_suffix(""))],
            check=True,
        )
    elif shutil.which("convert"):
        subprocess.run(
            ["convert", "-density", "300", str(pdf_out), str(png_out)],
            check=True,
        )
    else:
        raise RuntimeError(
            f"Wrote {pdf_out.name} but cannot rasterise PNG "
            "(need pdftoppm or ImageMagick convert)."
        )
    if not png_out.exists():
        raise RuntimeError(f"PNG rasterisation failed for {png_out}")
    print(f"wrote {pdf_out.name} / {png_out.name} from TikZ")


def copy_error_drift(results: Path, out_dir: Path) -> None:
    """Promote error-drift PDFs to numbered fig4_* copies (PNG via matplotlib read not needed)."""
    src_dir = results / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Also regenerate a simple index figure listing available drift plots
    drifts = sorted(src_dir.glob("error_drift__*.pdf"))
    for i, src in enumerate(drifts, start=1):
        dest = out_dir / f"fig4_error_drift_{src.stem.replace('error_drift__', '')}.pdf"
        shutil.copy2(src, dest)
        # PNG companion: rasterize via pdftoppm if present else skip
        png = dest.with_suffix(".png")
        if shutil.which("pdftoppm"):
            subprocess.run(
                ["pdftoppm", "-png", "-r", "300", "-singlefile",
                 str(dest), str(dest.with_suffix(""))],
                check=False,
            )
        elif not png.exists():
            # placeholder: copy note by rendering blank is worse; leave PDF only
            pass
    print(f"copied {len(drifts)} error-drift PDFs as fig4_*")


def copy_dm_heatmaps(results: Path, out_dir: Path) -> None:
    src_dir = results / "figures"
    heats = sorted(src_dir.glob("dm_heatmap__*.pdf"))
    for src in heats:
        dest = out_dir / f"fig5_{src.name}"
        shutil.copy2(src, dest)
        if shutil.which("pdftoppm"):
            subprocess.run(
                ["pdftoppm", "-png", "-r", "300", "-singlefile",
                 str(dest), str(dest.with_suffix(""))],
                check=False,
            )
    print(f"copied {len(heats)} DM heatmaps as fig5_*")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--out", type=Path, default=ROOT / "figures")
    args = p.parse_args()
    set_global_seed(42)
    build_architecture(args.out)
    build_kkt(args.results, args.out)
    build_cd(args.results, args.out)
    copy_error_drift(args.results, args.out)
    copy_dm_heatmaps(args.results, args.out)
    # Keep results/figures in sync for scripts that still read there
    for f in args.out.glob("fig2_kkt_boundary.*"):
        shutil.copy2(f, args.results / "figures" / f.name.replace("fig2_", ""))
    print("figure build complete")


if __name__ == "__main__":
    main()
