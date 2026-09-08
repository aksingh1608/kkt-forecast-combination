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

"""Publication-quality EDA functions.

Each function returns either a Plotly figure (interactive — used in the
Streamlit frontend) or a Matplotlib figure (static — used when writing
artefacts to ``results/figures`` for the paper).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Tabular summaries
# ---------------------------------------------------------------------------
def summary_table(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """Compact, paper-ready summary statistics table."""
    desc = df.describe(include="number").T
    desc["missing_%"] = (df.isna().mean() * 100).reindex(desc.index)
    desc["dtype"] = df.dtypes.reindex(desc.index).astype(str)
    if target is not None and target in desc.index:
        desc.loc[target, "is_target"] = True
    return desc.round(4)


def seasonal_profile(series: pd.Series) -> dict[str, pd.Series]:
    """Mean target value grouped by hour, day-of-week and month."""
    s = series.dropna()
    idx = s.index
    return {
        "by_hour": s.groupby(idx.hour).mean().rename("mean"),
        "by_dow": s.groupby(idx.dayofweek).mean().rename("mean"),
        "by_month": s.groupby(idx.month).mean().rename("mean"),
    }


# ---------------------------------------------------------------------------
# Plotly figures (used by the Streamlit frontend)
# ---------------------------------------------------------------------------
def plot_overview(series: pd.Series, *, max_points: int = 5000) -> go.Figure:
    """Big-picture line plot with rolling mean overlay."""
    s = series.dropna()
    if len(s) > max_points:
        # Subsample uniformly so the plot stays responsive.
        step = max(1, len(s) // max_points)
        s_plot = s.iloc[::step]
    else:
        s_plot = s

    rolling = s.rolling(max(24, len(s) // 100), min_periods=1).mean()
    rolling = rolling.loc[s_plot.index]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s_plot.index, y=s_plot.values, mode="lines",
        line=dict(width=1, color="rgba(60,120,200,0.55)"),
        name=series.name or "value",
    ))
    fig.add_trace(go.Scatter(
        x=rolling.index, y=rolling.values, mode="lines",
        line=dict(width=2.5, color="rgba(220,80,40,0.95)"),
        name="rolling mean",
    ))
    fig.update_layout(
        template="plotly_white",
        title=f"Time-series overview · {series.name}",
        xaxis_title="Time",
        yaxis_title=series.name or "value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=420,
    )
    return fig


def plot_decomposition(series: pd.Series, period: int = 24) -> go.Figure:
    """STL decomposition: trend, seasonal, residual."""
    from statsmodels.tsa.seasonal import STL

    s = series.dropna()
    if len(s) <= 2 * period:
        raise ValueError("Series too short for STL.")
    # Subsample if huge — STL on >50k points is slow and visually identical.
    if len(s) > 20_000:
        step = max(1, len(s) // 20_000)
        s = s.iloc[::step]
        period = max(2, period)

    stl = STL(s, period=period, robust=True).fit()
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        vertical_spacing=0.04,
    )
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name="observed",
                             line=dict(color="#3776ab")), row=1, col=1)
    fig.add_trace(go.Scatter(x=stl.trend.index, y=stl.trend.values, name="trend",
                             line=dict(color="#d35400")), row=2, col=1)
    fig.add_trace(go.Scatter(x=stl.seasonal.index, y=stl.seasonal.values,
                             name="seasonal", line=dict(color="#2e8b57")), row=3, col=1)
    fig.add_trace(go.Scatter(x=stl.resid.index, y=stl.resid.values, name="residual",
                             line=dict(color="#7f7f7f", width=1)), row=4, col=1)
    fig.update_layout(
        template="plotly_white", height=720, showlegend=False,
        title="STL decomposition", margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Numeric-column correlation heatmap."""
    corr = df.select_dtypes(include="number").corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        title="Feature correlation matrix",
    )
    fig.update_layout(template="plotly_white", height=500,
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_seasonal_profiles(series: pd.Series) -> go.Figure:
    """Three-panel mean-by-(hour, DOW, month) plot."""
    profiles = seasonal_profile(series)
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("By hour of day", "By day of week", "By month"),
    )
    colors = ["#3776ab", "#d35400", "#2e8b57"]
    for i, (label, prof) in enumerate(profiles.items(), start=1):
        fig.add_trace(
            go.Bar(x=prof.index, y=prof.values, marker_color=colors[i - 1],
                   showlegend=False),
            row=1, col=i,
        )
    fig.update_layout(
        template="plotly_white",
        title=f"Seasonal profiles · {series.name}",
        height=380, margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


def plot_acf_pacf(series: pd.Series, lags: int = 48) -> go.Figure:
    """ACF and PACF as side-by-side bar plots with confidence bounds."""
    from statsmodels.tsa.stattools import acf, pacf

    s = series.dropna().astype(float)
    if len(s) > 20_000:
        s = s.iloc[-20_000:]
    nlags = min(lags, max(2, len(s) // 5 - 1))
    acf_vals, acf_ci = acf(s.values, nlags=nlags, alpha=0.05, fft=True)
    pacf_vals, pacf_ci = pacf(s.values, nlags=nlags, alpha=0.05, method="ywm")
    bound = 1.96 / np.sqrt(len(s))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
    fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals,
                         marker_color="#3776ab", showlegend=False), row=1, col=1)
    fig.add_hline(y=bound, line_dash="dash", line_color="#d35400", row=1, col=1)
    fig.add_hline(y=-bound, line_dash="dash", line_color="#d35400", row=1, col=1)
    fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals,
                         marker_color="#2e8b57", showlegend=False), row=1, col=2)
    fig.add_hline(y=bound, line_dash="dash", line_color="#d35400", row=1, col=2)
    fig.add_hline(y=-bound, line_dash="dash", line_color="#d35400", row=1, col=2)
    fig.update_layout(template="plotly_white", height=380,
                      margin=dict(l=40, r=20, t=60, b=40),
                      title="Autocorrelation diagnostics")
    return fig
