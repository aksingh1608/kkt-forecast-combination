"""Streamlit frontend — Advanced Time-Series Forecasting Benchmark.

Run locally with::

    streamlit run extras/streamlit_app.py

Load the bundled UCI dataset (or upload a CSV), explore the data, run the
model zoo, and view accuracy on screen. Nothing is saved to disk automatically.
"""

from __future__ import annotations

import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.data import (
    clean_dataset,
    infer_schema,
    load_generic_csv,
    load_uci_household_power,
    resample_dataset,
)
from src.eda.analysis import (
    plot_acf_pacf,
    plot_correlation_heatmap,
    plot_decomposition,
    plot_overview,
    plot_seasonal_profiles,
    summary_table,
)
from src.eda.extra_plots import (
    plot_box_by, plot_calibration, plot_error_heatmap, plot_fan_chart,
    plot_hour_dow_heatmap, plot_residuals_over_time, plot_target_histogram,
)
from src.eda.stationarity import stationarity_report
from src.evaluation import (
    adaptive_conformal_intervals, bootstrap_metric_ci,
    evaluate_probabilistic, missing_injection_curve,
    noise_injection_curve, per_horizon_for_models,
    split_conformal_intervals,
)
from src.evaluation.statistical_tests import diebold_mariano
from src.features import add_holiday_features, list_supported_countries
from src.features.engineering import build_supervised_frame
from src.compute import accuracy_per_second
from src.interpret import (
    permutation_importance_tabular, shap_summary,
)
from src.models import QuantileLightGBMForecaster
from src.pipeline import ExperimentResult, run_experiment
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("streamlit_app")


# ---------------------------------------------------------------------------
# Page configuration & global styles
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Energy Forecasting Benchmark",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }
        .stMetric { background: linear-gradient(135deg,#f8fafc 0%,#eef2ff 100%);
                    padding: 12px 16px; border-radius: 12px;
                    border: 1px solid #e2e8f0; }
        .winner-card {
            background: linear-gradient(135deg,#0ea5e9 0%,#6366f1 100%);
            color: white; padding: 22px 28px; border-radius: 16px;
            box-shadow: 0 8px 24px rgba(99,102,241,0.25);
        }
        .winner-card h1 { color: white; margin: 0 0 6px 0; font-size: 28px; }
        .winner-card p  { margin: 0; opacity: 0.92; }
        .pill {
            display: inline-block; padding: 3px 10px; border-radius: 999px;
            font-size: 12px; font-weight: 600; background: #eef2ff;
            color: #4338ca; margin-right: 6px;
        }
        .lib-card {
            background:#f8fafc; padding:10px 14px; border-radius:10px;
            border:1px solid #e2e8f0; margin-bottom:6px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached data helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_bundled_uci(nrows: int | None) -> pd.DataFrame:
    return load_uci_household_power(use_cache=True, nrows=nrows)


@st.cache_data(show_spinner=False)
def _process_uploaded(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, str]:
    buf = io.BytesIO(file_bytes)
    df, target = load_generic_csv(buf)
    return df, target


@st.cache_data(show_spinner=False)
def _prepare_frame(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    cleaned = clean_dataset(df)
    return resample_dataset(cleaned, freq=freq)


# ---------------------------------------------------------------------------
# Top-level header
# ---------------------------------------------------------------------------
st.title("⚡  Time-Series Forecasting Benchmark")
st.caption(
    "Compare 20+ forecasting algorithms on the UCI household dataset — "
    "EDA, benchmark, and metrics in the browser (no auto-save to disk)."
)


# ---------------------------------------------------------------------------
# Sidebar — mode selector
# ---------------------------------------------------------------------------
st.sidebar.title("⚡ Forecasting Lab")
st.sidebar.caption("BSBI MSc AI · MLF Final Assignment")

# --- Compute device badge ---------------------------------------------------
try:
    from src.compute import detect_device
    _dev = detect_device()
    if _dev.is_gpu():
        _badge = (f"🟢 **GPU** &nbsp; <code>{_dev.name}</code> &nbsp; "
                  f"({_dev.vram_gb} GB)")
    else:
        _badge = "⚪ **CPU** &nbsp; (install CUDA torch to enable GPU)"
    st.sidebar.markdown(_badge, unsafe_allow_html=True)
    st.sidebar.divider()
except Exception:  # pragma: no cover
    pass

# =================================================================================
# SHARED HELPER — render a leaderboard / predictions block from a result/snapshot
# =================================================================================
def _render_winner_card(best_row: pd.Series | None, metric: str) -> None:
    if best_row is None:
        return
    badge_color = "#16a34a" if best_row.get("R2", 0) > 0 else "#f59e0b"
    st.markdown(
        f"""
        <div class="winner-card">
            <span class="pill" style="background:rgba(255,255,255,0.18);color:white;">
                {best_row.get('Family', '—')}
            </span>
            <h1>🥇  Best model · {best_row.name}</h1>
            <p>
                <b>RMSE</b> {best_row.get('RMSE', float('nan')):.4f} &nbsp;·&nbsp;
                <b>MAE</b> {best_row.get('MAE', float('nan')):.4f} &nbsp;·&nbsp;
                <b>MAPE</b> {best_row.get('MAPE', float('nan')):.2f}% &nbsp;·&nbsp;
                <b>R²</b> <span style="color:{badge_color};font-weight:700;">
                {best_row.get('R2', float('nan')):.4f}</span> &nbsp;·&nbsp;
                <b>MASE</b> {best_row.get('MASE', float('nan')):.3f}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =================================================================================
# MODE 1: SINGLE DATASET BENCHMARK
# =================================================================================
def render_single_dataset_mode() -> None:
    st.sidebar.subheader("1️⃣  Data source")
    data_source = st.sidebar.radio(
        "Choose a source",
        options=["Bundled UCI dataset", "Upload my own CSV"],
        index=0,
    )

    uploaded_file = None
    nrows_subset: int | None = None
    if data_source == "Bundled UCI dataset":
        use_subset = st.sidebar.checkbox(
            "Use a subset (faster)", value=True,
            help="The full dataset has ~2M minute-level rows."
        )
        if use_subset:
            nrows_subset = st.sidebar.select_slider(
                "Subset rows",
                options=[50_000, 100_000, 250_000, 500_000, 1_000_000, 2_075_259],
                value=250_000,
            )
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload a CSV with a datetime column",
            type=["csv", "txt"],
        )

    st.sidebar.subheader("2️⃣  Preprocessing")
    freq_label = st.sidebar.selectbox(
        "Resampling frequency",
        options=[("Hourly (h)", "h"), ("4-hourly (4h)", "4h"),
                 ("Daily (D)", "D"), ("Original — no resample", None)],
        format_func=lambda x: x[0],
        index=0,
    )
    resample_freq = freq_label[1]
    st.sidebar.caption(
        "**Why hourly?** Minute-level UCI data is dominated by measurement "
        "noise; hourly aggregation preserves the full daily/weekly cycle "
        "(period m=24) while reducing runtime ~60×. Daily is more aggressive "
        "and only retains weekly seasonality."
    )

    st.sidebar.subheader("3️⃣  Modelling")
    test_size = st.sidebar.slider("Test fraction (chronological)", 0.05, 0.4, 0.2, 0.05)
    include_classical = st.sidebar.checkbox(
        "Include heavy classical models (ARIMA/SARIMA/HW)", value=True
    )
    # Only show the DL toggle if neuralforecast is importable.
    include_dl = False
    dl_context = 168
    dl_horizon = 24
    dl_max_steps = 200
    try:
        from src.models.optional_dl import neuralforecast_available
        if neuralforecast_available():
            try:
                _gpu_dev = detect_device()
                _dl_help = (
                    "Trains NHITS / NBEATS / PatchTST / LSTM / TFT / iTransformer / VanillaTransformer + Chronos. "
                    + ("**GPU detected** — bf16-mixed precision used."
                       if _gpu_dev.is_gpu()
                       else "Will run on CPU (slower).")
                )
            except Exception:
                _dl_help = "Trains the deep-learning + foundation zoo."
            include_dl = st.sidebar.checkbox(
                "Include deep-learning models",
                value=False, help=_dl_help,
            )
            if include_dl:
                with st.sidebar.expander("DL hyperparameters", expanded=False):
                    dl_context = st.select_slider(
                        "Context length (input_size)",
                        options=[24, 48, 96, 168, 336, 512, 1024, 2048],
                        value=168,
                        help="How many past steps the DL models see. "
                             "Longer = stronger seasonal capture but more VRAM.",
                    )
                    dl_horizon = st.select_slider(
                        "Rolling forecast chunk (horizon h)",
                        options=[1, 6, 12, 24, 48, 96, 168, 336],
                        value=24,
                        help="Per-call horizon. Test predictions are rolled in chunks of this size.",
                    )
                    dl_max_steps = st.slider(
                        "Training steps per model",
                        min_value=50, max_value=2000, value=300, step=50,
                    )
    except ImportError:
        pass
    seasonal_period_override = st.sidebar.number_input(
        "Seasonal period override (0 = auto)", min_value=0, max_value=8760, value=0
    )
    add_holidays = st.sidebar.checkbox(
        "Add holiday features", value=True,
        help="Adds is_holiday, is_holiday_eve, days_to_holiday columns. "
             "UCI dataset = France (FR).",
    )
    holiday_country = "FR"
    if add_holidays:
        holiday_country = st.sidebar.selectbox(
            "Holiday country", options=list_supported_countries(),
            index=0,
        )

    # Optional run label — important for the Library.
    default_label = (
        f"UCI ({nrows_subset:,} rows)" if data_source == "Bundled UCI dataset"
        else (uploaded_file.name if uploaded_file else "Uploaded dataset")
    )
    snapshot_label = st.sidebar.text_input(
        "Run label (for your notes only)", value=default_label,
        help="Not saved to disk — just displayed in the session.",
    )

    run_button = st.sidebar.button(
        "🚀 Run benchmark", type="primary", use_container_width=True
    )

    # ----------------------------------------------------- load data -----
    try:
        if data_source == "Upload my own CSV":
            if uploaded_file is None:
                st.info("⬅️ Please upload a CSV using the sidebar.")
                return
            raw_df, target = _process_uploaded(uploaded_file.getvalue(), uploaded_file.name)
        else:
            raw_df = _load_bundled_uci(nrows_subset)
            target = config.TARGET_COLUMN
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        return

    with st.spinner("Cleaning & resampling…"):
        df_proc = clean_dataset(raw_df) if resample_freq is None \
            else _prepare_frame(raw_df, resample_freq)
        if add_holidays:
            try:
                df_proc = add_holiday_features(df_proc, country=holiday_country)
            except Exception as exc:
                st.sidebar.warning(f"Holiday features skipped: {exc}")

    numeric_cols = df_proc.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns to forecast.")
        return
    default_target = target if target in numeric_cols else numeric_cols[0]
    target = st.sidebar.selectbox(
        "Target column", options=numeric_cols,
        index=numeric_cols.index(default_target) if default_target in numeric_cols else 0,
    )

    schema = infer_schema(df_proc, target)

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{schema.n_rows:,}")
    c2.metric("Frequency", str(schema.freq))
    c3.metric("Date range", f"{schema.start.date()} → {schema.end.date()}")
    c4.metric("Missing %", f"{schema.missing_pct:.2f}%")

    with st.expander("📋  Preview & summary statistics", expanded=False):
        st.dataframe(df_proc.head(50), use_container_width=True, height=240)
        st.markdown("**Summary statistics**")
        st.dataframe(summary_table(df_proc, target), use_container_width=True)

    # ---------------------------------------------------- EDA tabs ------
    st.subheader("📈  Exploratory Data Analysis")
    eda_tabs = st.tabs([
        "Overview", "Seasonal profiles", "STL decomposition", "ACF / PACF",
        "Correlations", "Box plots", "Hour×DOW heatmap", "Distribution",
        "Stationarity",
    ])
    with eda_tabs[0]:
        st.plotly_chart(plot_overview(df_proc[target]), use_container_width=True)
    with eda_tabs[1]:
        try:
            st.plotly_chart(plot_seasonal_profiles(df_proc[target]),
                            use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not build seasonal profile: {exc}")
    with eda_tabs[2]:
        try:
            st.plotly_chart(
                plot_decomposition(df_proc[target],
                                   period=int(seasonal_period_override) or 24),
                use_container_width=True,
            )
        except Exception as exc:
            st.info(f"STL decomposition skipped: {exc}")
    with eda_tabs[3]:
        try:
            st.plotly_chart(plot_acf_pacf(df_proc[target], lags=48),
                            use_container_width=True)
        except Exception as exc:
            st.info(f"Autocorrelation plot skipped: {exc}")
    with eda_tabs[4]:
        try:
            st.plotly_chart(plot_correlation_heatmap(df_proc),
                            use_container_width=True)
        except Exception as exc:
            st.info(f"Correlation heatmap skipped: {exc}")
    with eda_tabs[5]:
        bcols = st.columns(3)
        for i, group in enumerate(("hour", "day_of_week", "month")):
            with bcols[i]:
                try:
                    st.plotly_chart(
                        plot_box_by(df_proc[target], group=group),
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.info(f"Box ({group}) skipped: {exc}")
    with eda_tabs[6]:
        try:
            st.plotly_chart(plot_hour_dow_heatmap(df_proc[target]),
                            use_container_width=True)
        except Exception as exc:
            st.info(f"Hour×DOW heatmap skipped: {exc}")
    with eda_tabs[7]:
        try:
            st.plotly_chart(plot_target_histogram(df_proc[target]),
                            use_container_width=True)
        except Exception as exc:
            st.info(f"Histogram skipped: {exc}")
    with eda_tabs[8]:
        with st.spinner("Running ADF + KPSS tests…"):
            try:
                rep = stationarity_report(df_proc[target])
                st.success(rep.verdict)
                cA, cB, cC, cD = st.columns(4)
                cA.metric("ADF statistic", f"{rep.adf_stat:.3f}")
                cB.metric("ADF p-value", f"{rep.adf_pvalue:.4f}")
                cC.metric("KPSS statistic", f"{rep.kpss_stat:.3f}")
                cD.metric("KPSS p-value", f"{rep.kpss_pvalue:.4f}")
                st.caption(
                    "**ADF**: H0 = unit root → small p ⇒ stationary. "
                    "**KPSS**: H0 = stationarity → large p ⇒ stationary. "
                    "Use both, then differencing if both reject stationarity."
                )
            except Exception as exc:
                st.info(f"Stationarity tests skipped: {exc}")

    # --------------------------------------------------- run benchmark --
    st.markdown("---")
    st.subheader("🏁  Benchmark — every model under a common protocol")

    if "result" not in st.session_state:
        st.session_state["result"] = None
        st.session_state["result_meta"] = {}

    if run_button:
        progress = st.progress(0.0)
        status = st.empty()

        def _cb(msg: str, pct: float) -> None:
            progress.progress(min(max(pct, 0.0), 1.0))
            status.write(f"⏳ {msg}")

        t0 = time.perf_counter()
        try:
            result = run_experiment(
                df_proc, target=target,
                seasonal_period=int(seasonal_period_override) or None,
                test_size=test_size,
                include_heavy_classical=include_classical,
                include_optional_dl=include_dl,
                optional_dl_horizon=dl_horizon if include_dl else None,
                optional_dl_max_steps=dl_max_steps if include_dl else 200,
                optional_dl_input_size=dl_context if include_dl else None,
                progress_callback=_cb,
            )
            elapsed = time.perf_counter() - t0
            st.session_state["result"] = result
            st.session_state["result_meta"] = {
                "label": snapshot_label,
                "elapsed": elapsed,
                "schema": schema,
                "nrows_raw": nrows_subset,
                "nrows_processed": len(df_proc),
                "config": {
                    "freq": str(resample_freq),
                    "test_size": test_size,
                    "include_classical": include_classical,
                    "include_dl": include_dl,
                    "add_holidays": add_holidays,
                    "holiday_country": holiday_country if add_holidays else None,
                },
            }
            status.success(f"✅ Benchmark complete in {elapsed:.1f}s.")
        except Exception as exc:
            st.error(f"❌ Benchmark failed: {exc}")
            logger.exception("Benchmark failed.")
            progress.empty()

    result: ExperimentResult | None = st.session_state.get("result")
    if result is None:
        st.info("Configure the run on the left and click **🚀 Run benchmark**.")
        return

    meta = st.session_state.get("result_meta", {})
    n_ok = sum(1 for r in result.runs if r.error is None)
    n_fail = sum(1 for r in result.runs if r.error is not None)
    with st.expander("📌  Run protocol (what these numbers mean)", expanded=True):
        cfg = meta.get("config", {})
        _raw = meta.get("nrows_raw")
        raw_rows_s = f"{_raw:,}" if _raw is not None else "all (full file)"
        st.markdown(
            f"""
            | Setting | Value |
            |---------|--------|
            | Raw UCI rows loaded | **{raw_rows_s}** minute rows |
            | After clean + resample | **{meta.get('nrows_processed', len(result.y_train) + len(result.y_test)):,}** rows ({cfg.get('freq', 'h')}) |
            | Train / test (chronological) | **{len(result.y_train):,}** / **{len(result.y_test):,}** |
            | Test fraction | **{cfg.get('test_size', 0.2):.0%}** |
            | Seasonal period *m* | **{result.seasonal_period}** |
            | Heavy classical (ARIMA/SARIMA/HW) | **{'yes' if cfg.get('include_classical', True) else 'no'}** |
            | Deep learning | **{'yes' if cfg.get('include_dl') else 'no'}** |
            | Holiday features | **{'yes (' + str(cfg.get('holiday_country', '')) + ')' if cfg.get('add_holidays') else 'no'}** |
            | Models finished / failed | **{n_ok}** / **{n_fail}** |
            | Wall time | **{meta.get('elapsed', 0):.1f} s** |
            """
        )
        st.caption(
            "Metrics (RMSE, MAE, R²) are computed on the **held-out test period only** — "
            "not hardcoded. Use the same settings in your report."
        )
        if n_fail:
            failed = [r.name for r in result.runs if r.error]
            st.warning(f"Failed models: {', '.join(failed)}")

    sort_metric = st.selectbox(
        "Sort leaderboard by",
        options=["RMSE", "MAE", "MAPE", "sMAPE", "MASE", "R2"], index=0,
    )
    leaderboard = result.leaderboard(sort_by=sort_metric)
    best = result.best_model(sort_metric)
    if best is not None:
        best_row = leaderboard.set_index("Model").loc[best.name]
        _render_winner_card(best_row, sort_metric)

    st.markdown(" ")
    st.markdown("**Full leaderboard**")
    st.dataframe(
        leaderboard.style.background_gradient(
            subset=[sort_metric] if sort_metric in leaderboard.columns else None,
            cmap="RdYlGn_r" if sort_metric != "R2" else "RdYlGn",
        ).format(precision=4),
        use_container_width=True, height=460,
    )

    csv_bytes = leaderboard.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️  Download leaderboard CSV (optional)", data=csv_bytes,
        file_name="leaderboard.csv", mime="text/csv", use_container_width=True,
    )

    # --------------------------------------------- predictions plot ----
    st.subheader("🔎  Predicted vs Actual — pick model(s) to compare")
    model_names = [r.name for r in result.runs if r.error is None]
    default_selection = [best.name] if best else []
    for fallback in ("SeasonalNaive", "SARIMA"):
        if fallback in model_names and fallback not in default_selection:
            default_selection.append(fallback)
    selected_models = st.multiselect(
        "Models on the plot", options=model_names,
        default=default_selection or model_names[: min(3, len(model_names))],
    )
    if selected_models:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result.test_index, y=result.y_test.values, mode="lines",
            name="actual",
            line=dict(color="rgba(15,23,42,0.95)", width=2.5),
        ))
        palette = ["#ef4444", "#0ea5e9", "#22c55e", "#f59e0b",
                   "#a855f7", "#06b6d4", "#f43f5e", "#84cc16"]
        for i, name in enumerate(selected_models):
            run = next((r for r in result.runs if r.name == name), None)
            if run is None or run.error is not None:
                continue
            fig.add_trace(go.Scatter(
                x=result.test_index, y=run.predictions, mode="lines",
                name=name, line=dict(color=palette[i % len(palette)], width=1.6),
                opacity=0.9,
            ))
        fig.update_layout(
            template="plotly_white", height=480,
            title="Test-set predictions vs ground truth",
            xaxis_title="Time", yaxis_title=target,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            margin=dict(l=40, r=20, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # =====================================================================
    # ADVANCED ANALYTICS — Q1-paper add-ons (each expander is lazy)
    # =====================================================================
    st.markdown("---")
    st.subheader("🔬  Advanced Analytics (paper-grade)")
    st.caption(
        "Each panel below runs on demand. Heavier ones (SHAP, probabilistic, "
        "robustness) are gated behind a button to keep the UI responsive."
    )

    _render_advanced_sections(
        df_proc=df_proc,
        target=target,
        result=result,
        best=best,
        seasonal_period=int(seasonal_period_override) or 24,
        test_size=test_size,
    )


# =====================================================================
# Advanced analytics renderer (called from render_single_dataset_mode)
# =====================================================================
def _render_advanced_sections(
    *, df_proc, target, result, best, seasonal_period, test_size,
) -> None:
    preds_dict = {r.name: r.predictions for r in result.runs if r.error is None}

    # ---------------------------- residuals & error heatmap (best model) --
    with st.expander("📉  Residual diagnostics (best model)", expanded=False):
        if best is None:
            st.info("No best model available.")
        else:
            try:
                preds = preds_dict.get(best.name, np.full(len(result.y_test), np.nan))
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_residuals_over_time(
                        result.test_index, result.y_test.values, preds, model_name=best.name,
                    ), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_error_heatmap(
                        result.test_index, result.y_test.values, preds,
                    ), use_container_width=True)
            except Exception as exc:
                st.info(f"Residual diagnostics skipped: {exc}")

    # ---------------------------- per-horizon breakdown -------------------
    with st.expander("⏱️  Per-horizon error breakdown (1h / 6h / 24h / 168h)", expanded=False):
        try:
            per_h = per_horizon_for_models(
                result.y_test, preds_dict, horizons=(1, 6, 24, 168),
                y_train=result.y_train, seasonality=seasonal_period, metric="RMSE",
            )
            st.dataframe(
                per_h.style.background_gradient(cmap="RdYlGn_r"),
                use_container_width=True, height=420,
            )
            # Long-form bar plot for the top 6 models on the worst horizon.
            longest = per_h.columns[-1]
            top6 = per_h.sort_values(longest).head(6).reset_index()
            fig = px.bar(
                top6.melt(id_vars="model", var_name="horizon", value_name="RMSE"),
                x="horizon", y="RMSE", color="model", barmode="group",
            )
            fig.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.info(f"Per-horizon breakdown skipped: {exc}")

    # ---------------------------- bootstrap CIs + Diebold-Mariano ---------
    with st.expander("📐  Bootstrap CIs + Diebold–Mariano matrix", expanded=False):
        sel_models = st.multiselect(
            "Models to evaluate (top-RMSE selected by default)",
            options=list(preds_dict.keys()),
            default=[r.name for r in result.runs if r.error is None][:5],
            key="adv_dm_models",
        )
        n_boot = st.slider("Bootstrap iterations", 100, 1000, 300, 100,
                           key="adv_n_boot")
        if st.button("Run bootstrap + DM tests", key="adv_run_dm"):
            with st.spinner("Bootstrapping…"):
                ci_rows = []
                for name in sel_models:
                    p = preds_dict.get(name)
                    if p is None:
                        continue
                    for metric in ("RMSE", "MAE", "MAPE"):
                        ci = bootstrap_metric_ci(
                            result.y_test.values, p, metric=metric,
                            block_size=24, n_bootstrap=n_boot,
                            y_train=result.y_train.values,
                            seasonality=seasonal_period,
                        )
                        ci_rows.append({"Model": name, **ci})
                st.dataframe(pd.DataFrame(ci_rows), use_container_width=True)
            with st.spinner("Diebold–Mariano pairwise…"):
                dm_rows = []
                for i, a in enumerate(sel_models):
                    for b in sel_models[i + 1:]:
                        dm = diebold_mariano(
                            result.y_test.values, preds_dict[a], preds_dict[b], h=24,
                        )
                        dm_rows.append({"a": a, "b": b,
                                        "DM_stat": round(dm.statistic, 3),
                                        "p_value": round(dm.p_value, 4),
                                        "winner": a if dm.statistic < 0 else b})
                st.dataframe(pd.DataFrame(dm_rows), use_container_width=True)

    # ---------------------------- probabilistic + conformal ---------------
    with st.expander("🎲  Probabilistic forecasting (quantile + conformal)", expanded=False):
        st.caption(
            "Trains a Quantile-LightGBM on the engineered features, computes "
            "CRPS, Pinball, coverage and sharpness, and constructs split & "
            "adaptive conformal intervals around the median."
        )
        if st.button("Run probabilistic stack", key="adv_run_prob"):
            with st.spinner("Training quantile models + computing intervals…"):
                try:
                    X, y, _ = build_supervised_frame(df_proc, target)
                    n = len(X); n_test = max(1, int(round(n * test_size)))
                    n_cal = max(50, int(n_test * 0.5))
                    Xtr, ytr = X.iloc[:n - n_test - n_cal], y.iloc[:n - n_test - n_cal]
                    Xc,  yc  = X.iloc[n - n_test - n_cal:n - n_test], y.iloc[n - n_test - n_cal:n - n_test]
                    Xte, yte = X.iloc[n - n_test:], y.iloc[n - n_test:]
                    qmodel = QuantileLightGBMForecaster(
                        quantiles=(0.05, 0.1, 0.5, 0.9, 0.95))
                    qmodel.fit(ytr, X_train=Xtr)
                    qte = qmodel.predict_quantiles(len(Xte), X_test=Xte)
                    qcal = qmodel.predict_quantiles(len(Xc),  X_test=Xc)

                    prob = evaluate_probabilistic(
                        yte.values, qte,
                        intervals=((0.05, 0.95), (0.1, 0.9)),
                    )
                    cA, cB, cC, cD = st.columns(4)
                    cA.metric("CRPS", f"{prob.crps:.4f}")
                    cB.metric("Mean Pinball", f"{prob.mean_pinball:.4f}")
                    if prob.coverage_per_alpha:
                        first = next(iter(prob.coverage_per_alpha.items()))
                        cC.metric(f"Coverage @ {first[0]:.2f}", f"{first[1]:.3f}")
                    if prob.width_per_alpha:
                        first = next(iter(prob.width_per_alpha.items()))
                        cD.metric(f"Width @ {first[0]:.2f}", f"{first[1]:.3f}")

                    st.markdown("**Pinball loss per quantile**")
                    pq = pd.DataFrame(
                        list(prob.pinball_per_q.items()), columns=["q", "pinball"])
                    st.dataframe(pq, use_container_width=True, hide_index=True)

                    fan = plot_fan_chart(
                        yte.index, yte.values, qte[0.5],
                        qte[0.05], qte[0.95], coverage=0.90,
                        name="QuantileLightGBM",
                    )
                    st.plotly_chart(fan, use_container_width=True)

                    cov_grid = {}
                    for alpha in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
                        qa = max(0.01, (1 - alpha) / 2.0)
                        qb = 1 - qa
                        if qa in qte and qb in qte:
                            cov_grid[1 - alpha] = float(np.mean(
                                (yte.values >= qte[qa]) & (yte.values <= qte[qb])
                            ))
                    if cov_grid:
                        st.plotly_chart(plot_calibration(cov_grid),
                                        use_container_width=True)

                    cal_residuals = yc.values - qcal[0.5]
                    lo, hi, q_hat = split_conformal_intervals(
                        cal_residuals, qte[0.5], alpha=0.1)
                    lo_a, hi_a, _state = adaptive_conformal_intervals(
                        cal_residuals, qte[0.5],
                        test_actual=yte.values, alpha=0.1, gamma=0.005,
                    )
                    cov_split = float(np.mean((yte.values >= lo) & (yte.values <= hi)))
                    cov_aci = float(np.mean((yte.values >= lo_a) & (yte.values <= hi_a)))
                    width_split = float(np.mean(hi - lo))
                    width_aci = float(np.mean(hi_a - lo_a))
                    st.markdown(
                        f"**Conformal (target 90%)** — "
                        f"split coverage **{cov_split:.3f}** (width {width_split:.3f}); "
                        f"ACI coverage **{cov_aci:.3f}** (width {width_aci:.3f})."
                    )
                    fan2 = plot_fan_chart(
                        yte.index, yte.values, qte[0.5], lo_a, hi_a,
                        coverage=0.90, name="ACI (90% target)",
                    )
                    st.plotly_chart(fan2, use_container_width=True)
                except Exception as exc:
                    st.error(f"Probabilistic stack failed: {exc}")

    # ---------------------------- robustness suite ------------------------
    with st.expander("🛡️  Robustness (noise & missingness injection)", expanded=False):
        if st.button("Run robustness suite", key="adv_run_robust"):
            best_preds = preds_dict.get(best.name) if best else None
            if best_preds is None:
                st.info("No best model.")
            else:
                with st.spinner("Injecting perturbations…"):
                    nz = noise_injection_curve(result.y_test.values, best_preds)
                    miss = missing_injection_curve(result.y_test.values, best_preds)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Noise injection (σ vs RMSE)**")
                        fig = px.line(nz, x="sigma", y="RMSE", markers=True)
                        fig.update_layout(template="plotly_white", height=320)
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(nz, use_container_width=True, hide_index=True)
                    with c2:
                        st.markdown("**Missing injection (fraction vs RMSE)**")
                        fig = px.line(miss, x="missing_frac", y="RMSE", markers=True)
                        fig.update_layout(template="plotly_white", height=320)
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(miss, use_container_width=True, hide_index=True)

    # ---------------------------- compute & sustainability ----------------
    with st.expander("⚡  Compute profile + Pareto plot", expanded=False):
        try:
            rows = []
            for r in result.runs:
                if r.error is not None:
                    continue
                rows.append({
                    "Model": r.name, "Family": r.family,
                    "TrainTime(s)": round(r.train_time_s, 3),
                    "PredictTime(s)": round(r.predict_time_s, 3),
                    "RMSE": r.metrics.get("RMSE"),
                    "MAE": r.metrics.get("MAE"),
                })
            comp = accuracy_per_second(pd.DataFrame(rows))
            st.dataframe(
                comp.sort_values("acc_per_sec", ascending=False)
                    .style.format(precision=4),
                use_container_width=True, height=420,
            )
            fig = go.Figure()
            for _, row in comp.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["total_time_s"]], y=[row["accuracy_score"]],
                    mode="markers+text", text=[row["Model"]],
                    textposition="top center",
                    marker=dict(size=12),
                    name=row["Model"], showlegend=False,
                ))
            fig.update_layout(
                template="plotly_white", height=460,
                title="Pareto front: accuracy vs total time (lower-right is dominated)",
                xaxis_title="total time (s, log)", yaxis_title="relative accuracy",
                xaxis_type="log",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.info(f"Compute profile skipped: {exc}")

    # ---------------------------- interpretability ------------------------
    with st.expander("🧠  Interpretability (SHAP + permutation importance)", expanded=False):
        st.caption(
            "Re-trains the chosen tree-based model on the train split and "
            "computes SHAP values + permutation importance on a sample of "
            "the test split."
        )
        tree_models = [r.name for r in result.runs
                       if r.error is None
                       and r.family in {"boosting", "tree-ensemble"}]
        if not tree_models:
            st.info("No tree-based model in the leaderboard.")
        else:
            picked = st.selectbox(
                "Pick a tree-based model", options=tree_models, key="adv_interp_model"
            )
            if st.button("Run SHAP + permutation importance", key="adv_run_interp"):
                with st.spinner("Re-training and explaining…"):
                    try:
                        from src.models import build_registry as _br
                        spec = next(s for s in _br() if s.name == picked)
                        inst = spec.factory()
                        X, y, _ = build_supervised_frame(df_proc, target)
                        n = len(X); n_test = max(1, int(round(n * test_size)))
                        Xtr, ytr = X.iloc[:-n_test], y.iloc[:-n_test]
                        Xte, yte = X.iloc[-n_test:], y.iloc[-n_test:]
                        inst.fit(ytr, X_train=Xtr)
                        underlying = inst.estimator
                        shap_table = shap_summary(underlying, Xte, max_samples=400, top_k=15)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Top features by mean(|SHAP|)**")
                            fig = px.bar(
                                shap_table.head(15),
                                x="mean_abs_shap", y="feature",
                                orientation="h",
                            )
                            fig.update_layout(template="plotly_white", height=420,
                                              yaxis=dict(autorange="reversed"))
                            st.plotly_chart(fig, use_container_width=True)
                        with c2:
                            pi_table = permutation_importance_tabular(
                                underlying, Xte, yte, n_repeats=5, top_k=15)
                            st.markdown("**Permutation importance**")
                            fig = px.bar(
                                pi_table.head(15),
                                x="importance_mean", y="feature",
                                orientation="h",
                                error_x="importance_std",
                            )
                            fig.update_layout(template="plotly_white", height=420,
                                              yaxis=dict(autorange="reversed"))
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as exc:
                        st.error(f"Interpretability failed: {exc}")


render_single_dataset_mode()

st.markdown("---")
st.caption(
    "Built for the BSBI MSc AI Machine Learning Fundamentals module · "
    "Source: ``Code/``"
)
