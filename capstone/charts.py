"""Trading-style charts for the dashboard (Plotly).

`price_chart` draws a TradingView-like price chart: candles (or Heikin-Ashi, OHLC bars, line) on
bonus/split-adjusted prices, moving averages, Bollinger bands, 52-week and pivot levels, the
expected open with its 10-90% range, and optional Volume / RSI / MACD panels. The visible window
is sliced in Python and the price axis is fitted to it, so every period is readable (Plotly's own
range buttons do not rescale the price axis, which made earlier charts look flat).

`style()` and `CONFIG` give every chart in the app the same look: crosshair, compact hover,
scroll-to-zoom, drag-to-pan and drawing tools for trend lines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from capstone import indicators as ind

UP, DOWN = "#089981", "#f23645"          # candle colours (teal-green up, red down)
UP_FADE, DOWN_FADE = "rgba(8,153,129,0.45)", "rgba(242,54,69,0.45)"
FORECAST = "#6250d6"
GRID = "rgba(128,128,128,0.16)"
MUTED = "#8a8984"
FONT = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

OVERLAY_COLORS = {"SMA 20": "#2a78d6", "SMA 50": "#eda100", "SMA 200": "#6250d6",
                  "EMA 9": "#1baf7a", "EMA 21": "#e87ba4"}
OVERLAYS = ["SMA 20", "SMA 50", "SMA 200", "EMA 9", "EMA 21", "Bollinger (20, 2)",
            "52-week high/low", "Pivot levels"]
PANELS = ["Volume", "RSI 14", "MACD (12, 26, 9)"]
STYLES = ["Candles", "Heikin-Ashi", "OHLC bars", "Line"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100"]

# Passed to st.plotly_chart(config=...): scroll zoom, drawing tools, sharp PNG export.
CONFIG = {
    "scrollZoom": True, "displaylogo": False,          # toolbar appears on hover, so it never hides titles
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "eraseshape"],
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def style(fig: go.Figure, height: int = 420, title: str | None = None, hover: str = "x unified",
          legend_top: bool = True) -> go.Figure:
    """Common look: readable fonts, light grid, crosshair spikes, compact hover, legend on top."""
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>" if title else None, x=0, xanchor="left", font=dict(size=16)),
        height=height, hovermode=hover, font=dict(family=FONT, size=13),
        margin=dict(l=10, r=10, t=70 if title else 30, b=10),
        hoverlabel=dict(font=dict(family=FONT, size=12), namelength=-1),
        legend=(dict(orientation="h", y=1.0, yanchor="bottom", x=0, xanchor="left", bgcolor="rgba(0,0,0,0)")
                if legend_top else dict(orientation="h", y=-0.18)),
        newshape=dict(line=dict(color=FORECAST, width=2)), dragmode="zoom", bargap=0.15,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, showspikes=True, spikemode="across",
                     spikesnap="cursor", spikethickness=1, spikedash="dot", spikecolor=MUTED, showline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, showspikes=True, spikemode="across",
                     spikesnap="cursor", spikethickness=1, spikedash="dot", spikecolor=MUTED,
                     automargin=True)
    return fig


def _breaks(dates: pd.Series, extra_end: pd.Timestamp | None = None) -> list[dict]:
    """Hide days with no session (weekends, NSE holidays) so candles sit side by side.

    Special weekend sessions (e.g. a Sunday budget or Muhurat session) stay visible: weekends are
    only hidden wholesale when the window has no weekend session.
    """
    if dates.empty:
        return []
    keep = set(pd.DatetimeIndex(dates).normalize())
    if extra_end is not None:
        keep.add(pd.Timestamp(extra_end).normalize())
    start, end = min(keep), max(keep)
    if any(d.weekday() >= 5 for d in keep):
        missing = [d for d in pd.date_range(start, end) if d not in keep]
        return [dict(values=[d.strftime("%Y-%m-%d") for d in missing])] if missing else []
    missing = [d for d in pd.bdate_range(start, end) if d not in keep]
    out = [dict(bounds=["sat", "mon"])]
    if missing:
        out.append(dict(values=[d.strftime("%Y-%m-%d") for d in missing]))
    return out


def _fmt_vol(v: float) -> str:
    if v != v:
        return "–"
    for unit, div in (("Cr", 1e7), ("L", 1e5), ("K", 1e3)):
        if abs(v) >= div:
            return f"{v / div:,.2f} {unit}"
    return f"{v:,.0f}"


def price_chart(df: pd.DataFrame, title: str, *, period: str = "6M", chart_style: str = "Candles",
                overlays=("SMA 20", "SMA 50"), panels=("Volume", "RSI 14"), log: bool = False,
                forecast: dict | None = None, levels: dict | None = None) -> go.Figure:
    """df: date, open, high, low, close[, volume], sorted or not, ideally already adjusted.

    forecast: {date, open, low, high} for the expected next open. levels: {label: (value, colour)}.
    Indicators are computed on the full history, then the chart shows the last `period` sessions.
    """
    full = df.sort_values("date").reset_index(drop=True)
    c = full["close"]
    calc = {"SMA 20": ind.sma(c, 20), "SMA 50": ind.sma(c, 50), "SMA 200": ind.sma(c, 200),
            "EMA 9": ind.ema(c, 9), "EMA 21": ind.ema(c, 21)}
    bb_lo, bb_mid, bb_hi = ind.bollinger(c)
    rsi = ind.rsi(c)
    mac = ind.macd(c)
    has_vol = "volume" in full and full["volume"].notna().any()
    panels = [p for p in panels if p != "Volume" or has_vol]

    n = ind.PERIODS.get(period)
    idx = full.index if n is None else full.index[-n:]
    h = full.loc[idx]
    if chart_style == "Heikin-Ashi":
        h = ind.heikin_ashi(full).loc[idx]
    x = h["date"]

    rows = 1 + len(panels)
    price_h = 520
    panel_h = 130
    heights = [price_h] + [panel_h] * len(panels)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[v / sum(heights) for v in heights])

    # ---- price ----
    prev = full["close"].shift(1).loc[idx]
    chg = (full.loc[idx, "close"] / prev - 1) * 100
    vol_txt = [_fmt_vol(v) for v in h["volume"]] if has_vol else [""] * len(h)
    hover = [f"O {o:,.2f}  H {hi:,.2f}  L {lo:,.2f}  C {cl:,.2f}  ({ch:+.2f}%)" + (f"  Vol {v}" if v else "")
             for o, hi, lo, cl, ch, v in zip(h["open"], h["high"], h["low"], h["close"], chg.fillna(0), vol_txt)]
    if chart_style in ("Candles", "Heikin-Ashi"):
        fig.add_trace(go.Candlestick(
            x=x, open=h["open"], high=h["high"], low=h["low"], close=h["close"], name="Price",
            increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
            decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN),
            whiskerwidth=0, hovertext=hover, hoverinfo="text", showlegend=False), row=1, col=1)
    elif chart_style == "OHLC bars":
        fig.add_trace(go.Ohlc(x=x, open=h["open"], high=h["high"], low=h["low"], close=h["close"], name="Price",
                              increasing_line_color=UP, decreasing_line_color=DOWN, tickwidth=0.3,
                              hovertext=hover, hoverinfo="text", showlegend=False), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=x, y=h["close"], name="Close", mode="lines", line=dict(color="#2a78d6", width=2),
                                 fill="tozeroy" if not log else None, fillcolor="rgba(42,120,214,0.08)",
                                 hovertext=hover, hoverinfo="text", showlegend=False), row=1, col=1)

    y_vals = [h["low"].min(), h["high"].max()] if chart_style != "Line" else [h["close"].min(), h["close"].max()]
    for name in overlays:
        if name in calc:
            s = calc[name].loc[idx]
            if s.notna().any():
                fig.add_trace(go.Scatter(x=x, y=s, name=name, mode="lines", connectgaps=False,
                                         line=dict(color=OVERLAY_COLORS[name], width=1.6),
                                         hovertemplate=f"{name} %{{y:,.2f}}<extra></extra>"), row=1, col=1)
        elif name.startswith("Bollinger"):
            lo, hi = bb_lo.loc[idx], bb_hi.loc[idx]
            fig.add_trace(go.Scatter(x=x, y=hi, name="BB upper", mode="lines", line=dict(color=MUTED, width=1),
                                     hovertemplate="BB upper %{y:,.2f}<extra></extra>", legendgroup="bb",
                                     showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=lo, name="Bollinger (20, 2)", mode="lines", fill="tonexty",
                                     fillcolor="rgba(138,137,132,0.10)", line=dict(color=MUTED, width=1),
                                     hovertemplate="BB lower %{y:,.2f}<extra></extra>", legendgroup="bb"),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=bb_mid.loc[idx], name="BB mid", mode="lines", showlegend=False,
                                     line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip"), row=1, col=1)
            y_vals += [lo.min(), hi.max()]

    lv = dict(levels or {})
    last = full.iloc[-1]
    if "52-week high/low" in overlays:
        yr = full.tail(252)
        lv["52W high"] = (yr["high"].max(), UP)
        lv["52W low"] = (yr["low"].min(), DOWN)
    if "Pivot levels" in overlays:
        for k, v in ind.pivots(last).items():
            lv[k] = (v, "#6250d6" if k == "Pivot" else (UP if k.startswith("S") else DOWN))
    near_lo, near_hi = float(np.nanmin(y_vals)) * 0.92, float(np.nanmax(y_vals)) * 1.08
    for label, (value, color) in lv.items():
        if not near_lo <= value <= near_hi:   # far-away level (e.g. an old 52-week high): note it, don't squash
            continue
        fig.add_hline(y=value, line=dict(color=color, width=1, dash="dash"), row=1, col=1,
                      annotation=dict(text=f"{label} {value:,.2f}", font=dict(size=11, color=color),
                                      bgcolor="rgba(255,255,255,0.75)"),
                      annotation_position="top left")
        y_vals.append(value)
    far = {k: v for k, (v, _) in lv.items() if not near_lo <= v <= near_hi}
    if far:
        fig.add_annotation(xref="paper", yref="paper", x=0, y=1, xanchor="left", yanchor="top", showarrow=False,
                           text="Off-screen: " + " · ".join(f"{k} {v:,.2f}" for k, v in far.items()),
                           font=dict(size=11, color=MUTED), bgcolor="rgba(255,255,255,0.7)")

    end_x = x.max()
    if forecast:
        fd = pd.Timestamp(forecast["date"])
        fig.add_trace(go.Scatter(
            x=[fd], y=[forecast["open"]], mode="markers", name="Expected open (10-90% range)",
            marker=dict(symbol="diamond", size=13, color=FORECAST, line=dict(width=1.5, color="white")),
            error_y=dict(type="data", symmetric=False, array=[forecast["high"] - forecast["open"]],
                         arrayminus=[forecast["open"] - forecast["low"]], color=FORECAST, thickness=2, width=8),
            hovertemplate=(f"Expected open ₹%{{y:,.2f}}<br>Likely range ₹{forecast['low']:,.2f} – "
                           f"₹{forecast['high']:,.2f}<extra></extra>")), row=1, col=1)
        fig.add_annotation(x=fd, y=forecast["high"], text=f"<b>₹{forecast['open']:,.2f}</b>", showarrow=False,
                           yshift=14, font=dict(color=FORECAST, size=12), row=1, col=1)
        y_vals += [forecast["low"], forecast["high"]]
        end_x = max(end_x, fd)

    # last-price line and tag on the right edge
    lc = float(full.loc[idx[-1], "close"])
    up_day = (chg.iloc[-1] if len(chg) else 0) >= 0
    tag = UP if up_day else DOWN
    fig.add_hline(y=lc, line=dict(color=tag, width=1, dash="dot"), row=1, col=1)
    fig.add_annotation(xref="paper", x=1, xanchor="left", y=np.log10(lc) if log else lc, yref="y",
                       text=f"<b>{lc:,.2f}</b>",
                       showarrow=False, font=dict(color="white", size=11), bgcolor=tag, borderpad=3)

    lo_y, hi_y = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
    if log:
        lo_l, hi_l = np.log10(lo_y), np.log10(hi_y)
        pad = (hi_l - lo_l) * 0.06 or 0.01
        fig.update_yaxes(type="log", range=[lo_l - pad, hi_l + pad], row=1, col=1)
    else:
        pad = (hi_y - lo_y) * 0.06 or hi_y * 0.01
        fig.update_yaxes(range=[lo_y - pad, hi_y + pad], row=1, col=1)
    fig.update_yaxes(side="right", tickformat=",.0f" if hi_y > 100 else ",.2f", tickprefix="₹", row=1, col=1)

    # ---- panels ----
    for r, p in enumerate(panels, start=2):
        if p == "Volume":
            colors = np.where(full.loc[idx, "close"] >= full.loc[idx, "open"], UP_FADE, DOWN_FADE)
            fig.add_trace(go.Bar(x=x, y=h["volume"], name="Volume", marker_color=colors, marker_line_width=0,
                                 showlegend=False, hovertemplate="Volume %{y:,.3s}<extra></extra>"), row=r, col=1)
            fig.add_trace(go.Scatter(x=x, y=full["volume"].rolling(20).mean().loc[idx], name="Avg volume (20d)",
                                     mode="lines", line=dict(color="#2a78d6", width=1.3), showlegend=False,
                                     hovertemplate="Avg vol 20d %{y:,.3s}<extra></extra>"), row=r, col=1)
            fig.update_yaxes(title_text="Volume", tickformat=".2s", side="right", row=r, col=1)
        elif p.startswith("RSI"):
            fig.add_hrect(y0=30, y1=70, fillcolor="rgba(98,80,214,0.07)", line_width=0, row=r, col=1)
            for lvl in (30, 70):
                fig.add_hline(y=lvl, line=dict(color=MUTED, width=1, dash="dash"), row=r, col=1)
            fig.add_trace(go.Scatter(x=x, y=rsi.loc[idx], name="RSI 14", mode="lines", showlegend=False,
                                     line=dict(color="#6250d6", width=1.5),
                                     hovertemplate="RSI %{y:.1f}<extra></extra>"), row=r, col=1)
            fig.update_yaxes(title_text="RSI 14", range=[0, 100], tickvals=[30, 50, 70], side="right", row=r, col=1)
        elif p.startswith("MACD"):
            m = mac.loc[idx]
            fig.add_trace(go.Bar(x=x, y=m["hist"], name="MACD histogram", showlegend=False, marker_line_width=0,
                                 marker_color=np.where(m["hist"] >= 0, UP_FADE, DOWN_FADE),
                                 hovertemplate="Hist %{y:.2f}<extra></extra>"), row=r, col=1)
            fig.add_trace(go.Scatter(x=x, y=m["macd"], name="MACD", mode="lines", showlegend=False,
                                     line=dict(color="#2a78d6", width=1.4),
                                     hovertemplate="MACD %{y:.2f}<extra></extra>"), row=r, col=1)
            fig.add_trace(go.Scatter(x=x, y=m["signal"], name="Signal", mode="lines", showlegend=False,
                                     line=dict(color="#eb6834", width=1.4),
                                     hovertemplate="Signal %{y:.2f}<extra></extra>"), row=r, col=1)
            fig.update_yaxes(title_text="MACD", side="right", row=r, col=1)

    style(fig, height=sum(heights) + 90, title=title)
    fig.update_layout(xaxis_rangeslider_visible=False, dragmode="pan", margin=dict(l=10, r=78, t=70, b=10))
    span = pd.Timedelta(days=2 if forecast else 1)
    fig.update_xaxes(rangebreaks=_breaks(x, pd.Timestamp(forecast["date"]) if forecast else None),
                     range=[x.min() - pd.Timedelta(hours=18), end_x + span])
    for r in range(1, rows):
        fig.update_xaxes(showticklabels=False, row=r, col=1)
    return fig


def candles(hist: pd.DataFrame, title: str, mas=(20, 50), forecast: dict | None = None,
            levels: dict | None = None, show_days: int = 126) -> go.Figure:
    """Backward-compatible wrapper around price_chart (6 months, SMAs, volume)."""
    period = next((k for k, v in ind.PERIODS.items() if v and v >= show_days), "All")
    return price_chart(hist, title, period=period, overlays=[f"SMA {m}" for m in mas], panels=["Volume"],
                       forecast=forecast, levels=levels)


def line(df: pd.DataFrame, x: str, ys: dict[str, str], title: str, y_title: str,
         show_days: int | None = None, period: str | None = None, area: bool | None = None,
         prefix: str = "") -> go.Figure:
    """Clean line chart: the shown window is sliced so the y axis fits it; last value tagged.

    ys maps column -> legend name. `period` (1M..All) or `show_days` picks the window.
    """
    d = df.sort_values(x)
    n = ind.PERIODS.get(period) if period else show_days
    if n:
        cutoff = d[x].iloc[-min(n, len(d))] if len(d) else None
        d = d[d[x] >= cutoff] if cutoff is not None else d
    fig = go.Figure()
    single = len(ys) == 1 if area is None else area
    for (col, name), color in zip(ys.items(), PALETTE):
        fig.add_trace(go.Scatter(x=d[x], y=d[col], name=name, mode="lines", line=dict(width=2, color=color),
                                 fill="tozeroy" if single else None,
                                 fillcolor="rgba(42,120,214,0.08)" if single else None,
                                 hovertemplate=f"{name} {prefix}%{{y:,.2f}}<extra></extra>"))
        last = d[col].dropna()
        if len(last):
            fig.add_annotation(xref="paper", x=1, xanchor="left", y=float(last.iloc[-1]), text=f"<b>{last.iloc[-1]:,.2f}</b>",
                               showarrow=False, font=dict(color="white", size=11), bgcolor=color, borderpad=3)
    vals = pd.concat([d[c] for c in ys]).dropna()
    if len(vals):
        lo, hi = float(vals.min()), float(vals.max())
        pad = (hi - lo) * 0.08 or abs(hi) * 0.01 or 1
        fig.update_yaxes(range=[lo - pad, hi + pad])
    style(fig, height=420, title=title, legend_top=len(ys) > 1)
    fig.update_yaxes(title_text=y_title, side="right", tickprefix=prefix)
    fig.update_layout(showlegend=len(ys) > 1, margin=dict(l=10, r=78, t=70, b=10))
    return fig
