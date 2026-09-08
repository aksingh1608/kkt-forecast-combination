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

"""Extra publication-quality plots: box plots, hour×DOW heatmap, histograms,
plus a multi-model cumulative-error plot for the paper's Section VII-C.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_box_by(
    series: pd.Series, group: str = "hour", max_n: int = 200_000
) -> go.Figure:
    """Box plot of ``series`` grouped by an index attribute."""
    s = series.dropna()
    if len(s) > max_n:
        s = s.sample(max_n, random_state=0).sort_index()
    idx = s.index
    if group == "hour":
        g = idx.hour
        title = "Distribution by hour of day"
    elif group in ("day_of_week", "dow"):
        g = idx.dayofweek
        title = "Distribution by day of week"
    elif group == "month":
        g = idx.month
        title = "Distribution by month"
    else:
        raise ValueError(f"Unknown group {group!r}")
    df = pd.DataFrame({"y": s.values, group: g})
    fig = px.box(df, x=group, y="y", color=group,
                 color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_layout(template="plotly_white", height=380, showlegend=False,
                      title=title, xaxis_title=group, yaxis_title=series.name,
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_hour_dow_heatmap(series: pd.Series) -> go.Figure:
    """Mean target value as a (hour × day-of-week) heatmap."""
    s = series.dropna()
    df = pd.DataFrame({
        "y": s.values,
        "hour": s.index.hour,
        "dow": s.index.dayofweek,
    })
    pivot = df.pivot_table(index="dow", columns="hour", values="y", aggfunc="mean")
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(pivot)]
    fig = px.imshow(
        pivot, aspect="auto",
        text_auto=".2f",
        color_continuous_scale="Viridis",
        labels={"x": "hour of day", "y": "day of week", "color": series.name},
    )
    fig.update_layout(template="plotly_white", height=420,
                      title=f"Mean {series.name} by hour × day-of-week",
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_target_histogram(series: pd.Series, bins: int = 60) -> go.Figure:
    s = series.dropna()
    fig = go.Figure(go.Histogram(x=s.values, nbinsx=bins,
                                 marker_color="#6366f1", opacity=0.85))
    fig.update_layout(template="plotly_white", height=340,
                      title=f"Distribution of {series.name}",
                      xaxis_title=series.name, yaxis_title="count",
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_residuals_over_time(
    test_index: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """Residual = actual − predicted, plotted over the test window."""
    residuals = np.asarray(actual) - np.asarray(predicted)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_index, y=residuals, mode="lines",
                             line=dict(color="#ef4444", width=1.4),
                             name="residual"))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(template="plotly_white", height=320,
                      title=f"Residuals over time — {model_name}",
                      xaxis_title="Time", yaxis_title="actual − predicted",
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_error_heatmap(
    test_index: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    metric: str = "abs_error",
) -> go.Figure:
    """|residual| heatmap by hour × day-of-week — shows *when* a model fails."""
    err = np.abs(np.asarray(actual) - np.asarray(predicted))
    df = pd.DataFrame({
        "err": err,
        "hour": test_index.hour,
        "dow": test_index.dayofweek,
    })
    pivot = df.pivot_table(index="dow", columns="hour", values="err", aggfunc="mean")
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(pivot)]
    fig = px.imshow(
        pivot, aspect="auto", text_auto=".2f",
        color_continuous_scale="Reds",
        labels={"x": "hour of day", "y": "day of week", "color": metric},
    )
    fig.update_layout(template="plotly_white", height=380,
                      title=f"Mean {metric} by hour × DOW",
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_fan_chart(
    test_index: pd.DatetimeIndex,
    actual: np.ndarray,
    median: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    coverage: float = 0.80,
    name: str = "FAHE-Q",
) -> go.Figure:
    """Probabilistic forecast fan chart (lower / median / upper + actual)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([test_index, test_index[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself", fillcolor="rgba(99,102,241,0.18)",
        line=dict(color="rgba(99,102,241,0)"),
        showlegend=True, name=f"{int(coverage*100)}% interval",
    ))
    fig.add_trace(go.Scatter(
        x=test_index, y=median, mode="lines",
        line=dict(color="#6366f1", width=2.0),
        name=f"{name} median",
    ))
    fig.add_trace(go.Scatter(
        x=test_index, y=actual, mode="lines",
        line=dict(color="black", width=2.0),
        name="actual",
    ))
    fig.update_layout(template="plotly_white", height=420,
                      title=f"Probabilistic forecast — {name}",
                      xaxis_title="Time", yaxis_title="value",
                      legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_calibration(
    coverage_actual: dict[float, float],
) -> go.Figure:
    """Reliability diagram: nominal vs actual coverage. Diagonal = perfect."""
    nominal = sorted(coverage_actual.keys())
    actual = [coverage_actual[n] for n in nominal]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nominal, y=actual, mode="lines+markers",
                             line=dict(color="#0ea5e9", width=2.5),
                             marker=dict(size=10), name="empirical"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color="grey", dash="dash"),
                             name="ideal"))
    fig.update_layout(template="plotly_white", height=380,
                      title="Reliability diagram (calibration)",
                      xaxis_title="nominal coverage",
                      yaxis_title="empirical coverage",
                      yaxis=dict(range=[0, 1.05]), xaxis=dict(range=[0, 1]),
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


# ---------------------------------------------------------------------------
# Multi-model cumulative-error / drift plot (paper Fig. drift)
# ---------------------------------------------------------------------------
def plot_cumulative_error(
    y_true: pd.Series,
    predictions_by_model: Mapping[str, np.ndarray],
    *,
    metric: str = "MAE",
    resample: str = "1D",
    title: str | None = None,
) -> go.Figure:
    """Rolling / cumulative absolute-error trace for many models.

    Answers the reviewer's request for a figure that shows **when** the
    classical models drift versus the tree learners. Two panels are drawn
    in a single Plotly figure:

    * Left  — daily-mean absolute error over the test window.
    * Right — cumulative absolute error (integral over time).

    Parameters
    ----------
    y_true
        Test-set actuals, indexed by timestamp.
    predictions_by_model
        Mapping ``{model_name: predictions}`` aligned with ``y_true.index``.
    metric
        ``"MAE"`` (default) or ``"MAPE"``. MAPE is clipped when |y| < 0.05
        to avoid blow-up on the residential load's overnight troughs.
    resample
        Pandas offset for the smoothing panel. ``"1D"`` (daily) is the
        paper's default.
    """
    if not isinstance(y_true.index, pd.DatetimeIndex):
        raise TypeError("y_true must have a DatetimeIndex.")

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2, shared_xaxes=False,
        subplot_titles=(f"{resample.capitalize()}-mean absolute error",
                        "Cumulative absolute error"),
    )
    palette = px.colors.qualitative.Bold + px.colors.qualitative.Set2
    yt = np.asarray(y_true, dtype=float)

    for i, (name, pred) in enumerate(predictions_by_model.items()):
        yp = np.asarray(pred, dtype=float).ravel()
        n = min(len(yt), len(yp))
        idx = y_true.index[:n]
        err = np.abs(yt[:n] - yp[:n])
        if metric.upper() == "MAPE":
            denom = np.where(np.abs(yt[:n]) < 0.05, 0.05, np.abs(yt[:n]))
            err = 100.0 * err / denom
        s = pd.Series(err, index=idx, name=name)
        # Panel 1 — daily-mean.
        daily = s.resample(resample).mean()
        colour = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(x=daily.index, y=daily.values,
                       mode="lines", name=name, line=dict(color=colour, width=2),
                       legendgroup=name),
            row=1, col=1,
        )
        # Panel 2 — cumulative.
        cum = s.cumsum()
        fig.add_trace(
            go.Scatter(x=cum.index, y=cum.values,
                       mode="lines", name=name, line=dict(color=colour, width=2),
                       legendgroup=name, showlegend=False),
            row=1, col=2,
        )

    y_label = "daily MAPE (%)" if metric.upper() == "MAPE" else "daily MAE"
    fig.update_yaxes(title_text=y_label, row=1, col=1)
    fig.update_yaxes(title_text="cumulative |error|", row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        height=440, width=None,
        title=title or "Error drift over the test window",
        margin=dict(l=50, r=20, t=70, b=40),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


__all__ = [
    "plot_box_by", "plot_hour_dow_heatmap", "plot_target_histogram",
    "plot_residuals_over_time", "plot_error_heatmap", "plot_fan_chart",
    "plot_calibration", "plot_cumulative_error",
]
