"""Plotly figures for model accuracy, weekly analysis, backtests and Monte Carlo.

Colours follow the entity, never its rank: each model and strategy keeps one hue everywhere.
Pure functions (data in, figure out) so they can be tested without Streamlit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Fixed categorical order (validated reference palette, light mode).
MODEL_COLORS = {"naive": "#2a78d6", "historical_average": "#eb6834", "linear_regression": "#1baf7a",
                "hist_gradient_boosting": "#6250d6", "zero": "#2a78d6"}
STRATEGY_COLORS = {"A": "#2a78d6", "B": "#eb6834", "H": "#8a8984"}
TRADABLE, NOT_TRADABLE = "#16a34a", "#b4b2a9"
ACTUAL = "#2b2b2b"
LAYOUT = dict(margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified", legend=dict(orientation="h", y=-0.15))


def _fig(title: str, y_title: str, x_title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, yaxis_title=y_title, xaxis_title=x_title, **LAYOUT)
    return fig


def predicted_vs_actual(df: pd.DataFrame, labels: dict[str, str]) -> go.Figure:
    """Actual gap line and one predicted line per model, by date (df: date, model, actual, pred)."""
    fig = _fig("Predicted vs actual opening gap", "Gap (%)")
    first = df[df["model"] == df["model"].iloc[0]].sort_values("date")
    fig.add_trace(go.Scatter(x=first["date"], y=first["actual"] * 100, name="Actual gap", mode="lines",
                             line=dict(color=ACTUAL, width=1.5)))
    for m, g in df.groupby("model", sort=False):
        g = g.sort_values("date")
        fig.add_trace(go.Scatter(x=g["date"], y=g["pred"] * 100, name=labels.get(m, m), mode="lines",
                                 line=dict(color=MODEL_COLORS.get(m), width=2)))
    return fig


def scatter_actual_vs_pred(df: pd.DataFrame, label: str, model: str) -> go.Figure:
    lim = float(np.nanpercentile(np.abs(np.r_[df["actual"], df["pred"]]) * 100, 99.5)) if len(df) else 1
    fig = _fig(f"Actual vs predicted gap: {label}", "Actual gap (%)", "Predicted gap (%)")
    fig.add_trace(go.Scattergl(x=df["pred"] * 100, y=df["actual"] * 100, mode="markers", name=label,
                               marker=dict(size=8, color=MODEL_COLORS.get(model), opacity=0.45,
                                           line=dict(width=1, color="white")),
                               hovertemplate="Predicted %{x:.2f}%<br>Actual %{y:.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", name="Perfect forecast",
                             line=dict(color="#8a8984", dash="dash", width=1)))
    fig.update_layout(hovermode="closest")
    return fig


def daily_error(df: pd.DataFrame, label: str, model: str) -> go.Figure:
    d = df.sort_values("date")
    fig = _fig(f"Daily forecast error (forecast - actual): {label}", "Error (% points)")
    fig.add_trace(go.Bar(x=d["date"], y=d["error"] * 100, name="Error", marker_color=MODEL_COLORS.get(model)))
    fig.add_hline(y=0, line=dict(color="#8a8984", width=1))
    return fig


def rolling_lines(roll: dict[str, pd.DataFrame], column: str, title: str, y_title: str,
                  labels: dict[str, str]) -> go.Figure:
    fig = _fig(title, y_title)
    for m, r in roll.items():
        fig.add_trace(go.Scatter(x=r["date"], y=r[column], name=labels.get(m, m), mode="lines",
                                 line=dict(color=MODEL_COLORS.get(m), width=2)))
    return fig


def metric_bars(table: pd.DataFrame, metric: str, title: str, labels: dict[str, str]) -> go.Figure:
    fig = _fig(title, "")
    t = table.sort_values(metric)
    fig.add_trace(go.Bar(y=t["model"].map(labels), x=t[metric], orientation="h",
                         marker_color=[MODEL_COLORS.get(m) for m in t["model"]],
                         text=t[metric].round(3), textposition="outside", hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(hovermode="closest", showlegend=False, height=60 * len(t) + 120)
    return fig


def weekly_delta(weeks: pd.DataFrame, symbol: str, cost_pct: float) -> go.Figure:
    """Weekly return bars, green above the round-trip cost line, grey otherwise."""
    w = weeks.sort_values("week_start")
    r = w["weekly_return"] * 100
    colors = np.where(w["weekly_return"] > cost_pct, TRADABLE, NOT_TRADABLE)
    fig = _fig(f"{symbol}: weekly return (first open to last close)", "Weekly return (%)", "Trading week")
    fig.add_trace(go.Bar(x=w["week_start"], y=r, marker_color=colors, name="Weekly return",
                         customdata=np.c_[w["entry_date"].dt.strftime("%a %d %b %Y"), w["exit_date"].dt.strftime("%a %d %b"),
                                          w["sessions"]],
                         hovertemplate="Entry %{customdata[0]} - exit %{customdata[1]} (%{customdata[2]} sessions)"
                                       "<br>Return %{y:.2f}%<extra></extra>"))
    fig.add_hline(y=cost_pct * 100, line=dict(color="#c0392b", dash="dash", width=1.5),
                  annotation_text=f"Round-trip cost {cost_pct * 100:.2f}%", annotation_position="top left")
    fig.update_layout(hovermode="closest", showlegend=False, bargap=0.1)
    return fig


def equity_curves(weeks: dict[str, pd.DataFrame], names: dict[str, str], capital: float) -> go.Figure:
    fig = _fig("Equity curves (net of all costs)", "Capital (₹)")
    for k, w in weeks.items():
        if len(w):
            fig.add_trace(go.Scatter(x=w["week_start"], y=w["equity"], name=names[k], mode="lines",
                                     line=dict(color=STRATEGY_COLORS.get(k), width=2)))
    fig.add_hline(y=capital, line=dict(color="#8a8984", dash="dot", width=1), annotation_text="Starting capital")
    return fig


def drawdown_curves(weeks: dict[str, pd.DataFrame], names: dict[str, str], capital: float) -> go.Figure:
    fig = _fig("Drawdown from previous peak", "Drawdown (%)")
    for k, w in weeks.items():
        if len(w):
            eq = np.r_[capital, w["equity"].to_numpy()]
            dd = (eq / np.maximum.accumulate(eq) - 1)[1:] * 100
            fig.add_trace(go.Scatter(x=w["week_start"], y=dd, name=names[k], mode="lines",
                                     line=dict(color=STRATEGY_COLORS.get(k), width=2)))
    return fig


def histogram(values, title: str, x_title: str, color: str, marks: dict[str, float] | None = None) -> go.Figure:
    fig = _fig(title, "Simulations", x_title)
    fig.add_trace(go.Histogram(x=values, nbinsx=60, marker_color=color, name=x_title,
                               marker_line=dict(width=1, color="white")))
    for label, v in (marks or {}).items():
        fig.add_vline(x=v, line=dict(color="#2b2b2b", dash="dash", width=1), annotation_text=label)
    fig.update_layout(hovermode="closest", showlegend=False, bargap=0.02)
    return fig
