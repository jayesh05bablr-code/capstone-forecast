"""Plotly figures for model accuracy, weekly analysis, backtests and Monte Carlo.

Colours follow the entity, never its rank: each model and strategy keeps one hue everywhere.
Pure functions (data in, figure out) so they can be tested without Streamlit. All figures use
charts.style() for the shared look (crosshair, compact hover, legend on top).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from capstone.charts import DOWN, MUTED, UP, style

# Fixed categorical order (validated reference palette, light mode).
MODEL_COLORS = {"naive": "#2a78d6", "historical_average": "#eb6834", "linear_regression": "#1baf7a",
                "hist_gradient_boosting": "#6250d6", "zero": "#2a78d6"}
STRATEGY_COLORS = {"A": "#2a78d6", "B": "#eb6834", "H": "#8a8984"}
TRADABLE, NOT_TRADABLE = "#16a34a", "#b4b2a9"
ACTUAL = "#2b2b2b"
OVER, UNDER = "#eb6834", "#2a78d6"   # forecast too high / too low


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _fig(title: str, y_title: str, x_title: str = "", height: int = 420, hover: str = "x unified",
         legend_top: bool = True) -> go.Figure:
    fig = go.Figure()
    style(fig, height=height, title=title, hover=hover, legend_top=legend_top)
    fig.update_layout(yaxis_title=y_title, xaxis_title=x_title)
    return fig


def predicted_vs_actual(df: pd.DataFrame, labels: dict[str, str]) -> go.Figure:
    """Actual gap as bars (green up, red down) and one predicted line per model (df: date, model, actual, pred)."""
    fig = _fig("Predicted vs actual opening gap", "Gap (%)", height=460)
    first = df[df["model"] == df["model"].iloc[0]].sort_values("date")
    a = first["actual"] * 100
    fig.add_trace(go.Bar(x=first["date"], y=a, name="Actual gap", marker_line_width=0,
                         marker_color=np.where(a >= 0, _rgba(UP, 0.45), _rgba(DOWN, 0.45)),
                         hovertemplate="Actual %{y:+.2f}%<extra></extra>"))
    for m, g in df.groupby("model", sort=False):
        g = g.sort_values("date")
        fig.add_trace(go.Scatter(x=g["date"], y=g["pred"] * 100, name=labels.get(m, m), mode="lines",
                                 line=dict(color=MODEL_COLORS.get(m), width=1.8),
                                 visible=True if m in ("hist_gradient_boosting", "linear_regression") else "legendonly",
                                 hovertemplate=f"{labels.get(m, m)} %{{y:+.2f}}%<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(bargap=0.05)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.07))
    fig.update_yaxes(ticksuffix="%")
    return fig


def scatter_actual_vs_pred(df: pd.DataFrame, label: str, model: str) -> go.Figure:
    """Each dot is one stock-day; shaded quadrants are where the direction was right."""
    lim = float(np.nanpercentile(np.abs(np.r_[df["actual"], df["pred"]]) * 100, 99.5)) if len(df) else 1
    lim = max(lim, 0.1)
    fig = _fig(f"Actual vs predicted: {label}", "Actual gap (%)", "Predicted gap (%)", hover="closest",
               legend_top=False, height=460)
    for x0, x1, y0, y1 in ((0, lim, 0, lim), (-lim, 0, -lim, 0)):
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=_rgba(UP, 0.07), line_width=0, layer="below")
    fig.add_trace(go.Scattergl(x=df["pred"] * 100, y=df["actual"] * 100, mode="markers", name=label,
                               marker=dict(size=7, color=MODEL_COLORS.get(model), opacity=0.45,
                                           line=dict(width=0.5, color="white")),
                               hovertemplate="Predicted %{x:+.2f}%<br>Actual %{y:+.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", name="Perfect forecast",
                             line=dict(color=MUTED, dash="dash", width=1), hoverinfo="skip"))
    if len(df):
        hit = (np.sign(df["pred"]) == np.sign(df["actual"])).mean() * 100
        fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top", showarrow=False,
                           text=f"Direction right on <b>{hit:.1f}%</b> of {len(df):,} days<br>"
                                f"<span style='font-size:11px'>shaded = right direction</span>",
                           bgcolor="rgba(255,255,255,0.8)", font=dict(size=12, color="#2b2b2b"))
    fig.update_xaxes(range=[-lim, lim], ticksuffix="%", zeroline=True, zerolinecolor=MUTED)
    fig.update_yaxes(range=[-lim, lim], ticksuffix="%", zeroline=True, zerolinecolor=MUTED,
                     scaleanchor="x", scaleratio=1)
    return fig


def daily_error(df: pd.DataFrame, label: str, model: str) -> go.Figure:
    """Error bars coloured by side (forecast too high / too low) with a 20-day mean absolute error line."""
    d = df.sort_values("date")
    e = d["error"] * 100
    fig = _fig(f"Daily error: {label}", "Forecast - actual (% points)", legend_top=False, height=460)
    fig.add_trace(go.Bar(x=d["date"], y=e, name="Error", marker_line_width=0, showlegend=False,
                         marker_color=np.where(e >= 0, _rgba(OVER, 0.7), _rgba(UNDER, 0.7)),
                         hovertemplate="Error %{y:+.2f} pts<extra></extra>"))
    fig.add_trace(go.Scatter(x=d["date"], y=e.abs().rolling(20, min_periods=5).mean(), name="20-day avg |error|",
                             mode="lines", line=dict(color="#2b2b2b", width=1.6),
                             hovertemplate="20-day avg |error| %{y:.2f}<extra></extra>"))
    for name, color in (("Forecast too high", OVER), ("Forecast too low", UNDER)):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(symbol="square", size=10, color=color)))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(bargap=0.05)
    return fig


def rolling_lines(roll: dict[str, pd.DataFrame], column: str, title: str, y_title: str,
                  labels: dict[str, str]) -> go.Figure:
    fig = _fig(title, y_title, legend_top=False, height=460)
    for m, r in roll.items():
        fig.add_trace(go.Scatter(x=r["date"], y=r[column], name=labels.get(m, m), mode="lines",
                                 line=dict(color=MODEL_COLORS.get(m), width=2 if m == "hist_gradient_boosting" else 1.5),
                                 hovertemplate=f"{labels.get(m, m)} %{{y:.2f}}<extra></extra>"))
    if "direction" in column:
        fig.add_hline(y=50, line=dict(color=MUTED, dash="dash", width=1), annotation_text="Coin flip 50%",
                      annotation_position="bottom right")
    return fig


def metric_bars(table: pd.DataFrame, metric: str, title: str, labels: dict[str, str]) -> go.Figure:
    """Horizontal bars, best model on top and labelled."""
    lower_better = "mae" in metric or "rmse" in metric
    t = table.sort_values(metric, ascending=not lower_better)
    best = t[metric].iloc[-1]
    fig = _fig(title, "", hover="closest", height=70 * len(t) + 120)
    fig.add_trace(go.Bar(y=t["model"].map(labels), x=t[metric], orientation="h", width=0.55,
                         marker_color=[MODEL_COLORS.get(m) for m in t["model"]],
                         text=[f"<b>{v:.3f}</b> ★ best" if v == best else f"{v:.3f}" for v in t[metric]],
                         textposition="outside", cliponaxis=False,
                         hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(showlegend=False, margin=dict(l=10, r=70, t=70, b=10))
    fig.update_xaxes(showspikes=False, range=[0, float(t[metric].max()) * 1.25])
    fig.update_yaxes(showspikes=False, showgrid=False)
    return fig


def weekly_delta(weeks: pd.DataFrame, symbol: str, cost_pct: float) -> go.Figure:
    """Weekly return bars, green above the round-trip cost line, grey otherwise, plus a 13-week average."""
    w = weeks.sort_values("week_start")
    r = w["weekly_return"] * 100
    colors = np.where(w["weekly_return"] > cost_pct, TRADABLE, NOT_TRADABLE)
    fig = _fig(f"{symbol}: weekly return (first open to last close)", "Weekly return (%)", height=440)
    fig.add_trace(go.Bar(x=w["week_start"], y=r, marker_color=colors, name="Weekly return", marker_line_width=0,
                         showlegend=False,
                         customdata=np.c_[w["entry_date"].dt.strftime("%a %d %b %Y"), w["exit_date"].dt.strftime("%a %d %b"),
                                          w["sessions"]],
                         hovertemplate="Entry %{customdata[0]} → exit %{customdata[1]} (%{customdata[2]} sessions)"
                                       "<br>Return <b>%{y:+.2f}%</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=w["week_start"], y=r.rolling(13, min_periods=4).mean(), name="13-week average",
                             mode="lines", line=dict(color="#2a78d6", width=2),
                             hovertemplate="13-week avg %{y:+.2f}%<extra></extra>"))
    for name, color in ((f"Beat the {cost_pct * 100:.2f}% cost", TRADABLE), ("Did not", NOT_TRADABLE)):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(symbol="square", size=10, color=color)))
    fig.add_hline(y=cost_pct * 100, line=dict(color="#c0392b", dash="dash", width=1.5),
                  annotation_text=f"Round-trip cost {cost_pct * 100:.2f}%", annotation_position="top left",
                  annotation_font=dict(color="#c0392b", size=11), annotation_bgcolor="rgba(255,255,255,0.85)")
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(hovermode="x", bargap=0.08)
    fig.update_yaxes(ticksuffix="%")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.07))
    return fig


def equity_curves(weeks: dict[str, pd.DataFrame], names: dict[str, str], capital: float) -> go.Figure:
    fig = _fig("Equity curves (net of all costs)", "Capital (₹)")
    for k, w in weeks.items():
        if len(w):
            fig.add_trace(go.Scatter(x=w["week_start"], y=w["equity"], name=names[k], mode="lines",
                                     line=dict(color=STRATEGY_COLORS.get(k), width=2),
                                     hovertemplate=f"{names[k]} ₹%{{y:,.0f}}<extra></extra>"))
    fig.add_hline(y=capital, line=dict(color=MUTED, dash="dot", width=1), annotation_text="Starting capital")
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
    return fig


def drawdown_curves(weeks: dict[str, pd.DataFrame], names: dict[str, str], capital: float) -> go.Figure:
    fig = _fig("Drawdown from previous peak", "Drawdown (%)")
    for k, w in weeks.items():
        if len(w):
            fig.add_trace(go.Scatter(x=w["week_start"], y=_drawdown(w, capital), name=names[k], mode="lines",
                                     fill="tozeroy", fillcolor=_rgba(STRATEGY_COLORS.get(k, MUTED), 0.10),
                                     line=dict(color=STRATEGY_COLORS.get(k), width=1.6),
                                     hovertemplate=f"{names[k]} %{{y:.1f}}%<extra></extra>"))
    fig.update_yaxes(ticksuffix="%")
    return fig


def _drawdown(w: pd.DataFrame, capital: float) -> np.ndarray:
    eq = np.r_[capital, w["equity"].to_numpy()]
    return (eq / np.maximum.accumulate(eq) - 1)[1:] * 100


def equity_and_drawdown(weeks: dict[str, pd.DataFrame], names: dict[str, str], capital: float) -> go.Figure:
    """Equity (top) and drawdown (bottom) on one shared time axis, so a fall lines up with its drawdown."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.04)
    for k, w in weeks.items():
        if not len(w):
            continue
        color = STRATEGY_COLORS.get(k, MUTED)
        fig.add_trace(go.Scatter(x=w["week_start"], y=w["equity"], name=names[k], mode="lines", legendgroup=k,
                                 line=dict(color=color, width=2),
                                 hovertemplate=f"{names[k]} ₹%{{y:,.0f}}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=w["week_start"], y=_drawdown(w, capital), name=names[k], mode="lines",
                                 legendgroup=k, showlegend=False, line=dict(color=color, width=1.5),
                                 hovertemplate=f"{names[k]} drawdown %{{y:.1f}}%<extra></extra>"), row=2, col=1)
    fig.add_hline(y=capital, line=dict(color=MUTED, dash="dot", width=1), row=1, col=1,
                  annotation_text=f"Start ₹{capital:,.0f}", annotation_position="top right",
                  annotation_bgcolor="rgba(255,255,255,0.85)", annotation_font=dict(size=11, color=MUTED))
    style(fig, height=640, title="Equity curve and drawdown (net of all costs)")
    fig.update_yaxes(title_text="Capital (₹)", tickprefix="₹", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", ticksuffix="%", row=2, col=1)
    return fig


def histogram(values, title: str, x_title: str, color: str, marks: dict[str, float] | None = None) -> go.Figure:
    """Distribution with the 5th-95th percentile range shaded and the median marked."""
    v = pd.Series(values).dropna()
    fig = _fig(f"{title} <span style='font-size:12px;font-weight:normal'>· shaded = 5th–95th percentile</span>",
               "Simulations", x_title, hover="closest")
    if len(v):
        p5, p50, p95 = np.percentile(v, [5, 50, 95])
        fig.add_vrect(x0=p5, x1=p95, fillcolor=_rgba(color, 0.08), line_width=0, layer="below")
        fig.add_vline(x=p50, line=dict(color="#2b2b2b", width=2),
                      annotation_text=f"Median {p50:,.1f}" if abs(p50) < 1e4 else f"Median {p50:,.0f}",
                      annotation_position="top left", annotation_bgcolor="rgba(255,255,255,0.85)",
                      annotation_font=dict(color="#2b2b2b", size=11))
    fig.add_trace(go.Histogram(x=v, nbinsx=60, marker_color=_rgba(color, 0.85), name=x_title,
                               marker_line=dict(width=0.5, color="white"),
                               hovertemplate=f"{x_title} %{{x}}<br>%{{y}} simulations<extra></extra>"))
    for label, mv in (marks or {}).items():
        fig.add_vline(x=mv, line=dict(color="#c0392b", dash="dash", width=1.4), annotation_text=label,
                      annotation_position="top right", annotation_bgcolor="rgba(255,255,255,0.85)",
                      annotation_font=dict(color="#c0392b", size=11))
    fig.update_layout(showlegend=False, bargap=0.02)
    fig.update_xaxes(showspikes=False)
    fig.update_yaxes(showspikes=False)
    return fig
