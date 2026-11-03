"""Company profile tab: profile, fundamentals, results, ownership and major investors."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from capstone import company, overnight, tracxn, uploads

NAN = float("nan")
LEGEND = dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom")
MARGIN = dict(l=10, r=10, t=50, b=10)
RESULT_SERIES = [("revenue_cr", "Revenue", "#2a78d6"), ("ebitda_cr", "EBITDA", "#1baf7a"),
                 ("net_profit_cr", "Net profit", "#eb6834")]
HOLDER_SERIES = [("promoter_pct", "Promoters", "#2a78d6"), ("fii_pct", "FII/FPI (foreign)", "#eb6834"),
                 ("dii_pct", "DII (domestic institutions)", "#1baf7a"),
                 ("public_pct", "Public / retail", "#eda100"), ("govt_pct", "Government", "#e87ba4")]


def _pct(v) -> str:
    return f"{v:.2f}%" if pd.notna(v) else "–"


def _header(comp: str) -> None:
    smap = overnight.stock_map().set_index("symbol")
    info = smap.loc[comp] if comp in smap.index else None
    prof = company.for_symbol(uploads.load("profile"), comp)
    st.subheader(comp)
    st.markdown(" · ".join(company.profile_facts(info, prof)))
    if len(prof) and "description" in prof and pd.notna(prof.iloc[-1].get("description")):
        st.write(prof.iloc[-1]["description"])


def _fundamentals(fund: pd.DataFrame) -> None:
    st.markdown("#### Fundamentals")
    if fund.empty:
        st.info("No fundamentals yet. Add them on the Upload company data tab.")
        return
    latest = fund.ffill().iloc[-1]
    mc = latest.get("market_cap_cr")
    big = pd.notna(mc) and mc >= 1e5
    if big:
        latest["market_cap_cr"] = mc / 1e5  # show lakh crore so the tile fits
    tiles = [("market_cap_cr", "Market cap", "₹{:,.2f}L Cr" if big else "₹{:,.0f} Cr"),
             ("pe", "P/E", "{:.1f}"), ("pb", "P/B", "{:.1f}"), ("roe_pct", "ROE", "{:.1f}%"),
             ("debt_to_equity", "Debt/Equity", "{:.2f}"), ("eps_ttm", "EPS (TTM)", "₹{:.1f}")]
    for col, (k, label, fmt) in zip(st.columns(6), tiles):
        v = latest.get(k)
        col.metric(label, fmt.format(v) if pd.notna(v) else "–")
    st.caption(f"Latest values as of {latest['as_of']}.")
    hist_cols = [c for c in company.FUNDAMENTAL_COLUMNS if c in fund]
    shown = fund[["as_of", *hist_cols]].dropna(how="all", subset=hist_cols)
    st.dataframe(shown.iloc[::-1], use_container_width=True, hide_index=True)


def _financials(comp: str) -> None:
    st.markdown("#### Financials")
    fin = company.for_symbol(uploads.load("financials"), comp).sort_values("period_end")
    if fin.empty:
        st.info("No results yet. Upload a financials sheet (period_end, revenue_cr, ebitda_cr, net_profit_cr...).")
        return
    fig = go.Figure()
    for col, name, color in RESULT_SERIES:
        if col in fin and fin[col].notna().any():
            fig.add_bar(x=fin["period_end"], y=fin[col], name=name, marker_color=color)
    fig.update_xaxes(type="category")
    fig.update_layout(barmode="group", height=380, title="Results by period (₹ crore)",
                      hovermode="x unified", margin=MARGIN, legend=LEGEND)
    st.plotly_chart(fig, use_container_width=True)
    if {"revenue_cr", "net_profit_cr"} <= set(fin.columns):
        fin = fin.assign(net_margin_pct=(fin["net_profit_cr"] / fin["revenue_cr"] * 100).round(1))
    st.dataframe(fin.drop(columns=["symbol"]).iloc[::-1], use_container_width=True, hide_index=True)


def _ownership(fund: pd.DataFrame) -> None:
    st.markdown("#### Ownership: foreign (FII/FPI) vs domestic")
    sh = company.shareholding(fund)
    tx_own = company.tracxn_ownership(fund)
    if len(tx_own):
        o = tx_own.iloc[-1]
        e1, e2 = st.columns(2)
        e1.metric("Insiders (Tracxn)", _pct(o["insiders_pct"]))
        e2.metric("All institutions (Tracxn)", _pct(o["institutions_pct"]))
        st.caption(f"From the Tracxn export as of {o['as_of']}. Tracxn does not split institutions into "
                   "foreign and domestic; the NSE shareholding pattern below does.")
    if sh.empty or len(sh.columns) <= 1:
        st.info("No shareholding pattern yet. Upload a shareholding sheet (as_of, promoter_pct, fii_pct, "
                "dii_pct, public_pct) from the company's quarterly filing on NSE/BSE.")
        return
    last = sh.iloc[-1]
    prev = sh.iloc[-2] if len(sh) > 1 else None
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Foreign institutions (FII/FPI)", _pct(last.get("fii_pct", NAN)),
              company.change_vs_previous(last, prev, "fii_pct"))
    d2.metric("Domestic institutions (DII)", f"{last.get('dii_pct', NAN):.2f}%",
              company.change_vs_previous(last, prev, "dii_pct"))
    d3.metric("Promoters", f"{last.get('promoter_pct', NAN):.2f}%",
              company.change_vs_previous(last, prev, "promoter_pct"))
    d4.metric("All domestic holders", f"{company.domestic_total(last):.2f}%")
    fig = go.Figure()
    for col, name, color in HOLDER_SERIES:
        if col in sh and sh[col].notna().any():
            fig.add_bar(x=sh["as_of"], y=sh[col], name=name, marker_color=color,
                        hovertemplate=name + ": %{y:.2f}%<extra></extra>")
    fig.update_xaxes(type="category")
    fig.update_layout(barmode="stack", height=380, title="Shareholding pattern by filing (%)",
                      yaxis=dict(range=[0, 100], ticksuffix="%"), margin=MARGIN, legend=LEGEND)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Rising FII share usually means foreign money is buying in; rising DII share means "
               "Indian mutual funds and insurers are. The model uses these percentages as features.")


def _investors(comp: str) -> None:
    st.markdown("#### Major investors")
    inv = company.for_symbol(uploads.load("investors"), comp)
    if inv.empty:
        st.info("No investor list yet. Upload an investors sheet (investor, investor_type, country, stake_pct).")
        return
    foreign, domestic, unknown = company.classify_investors(inv)
    groups = [(foreign, "Foreign"), (domestic, "Domestic"), (unknown, "Country not given (e.g. Tracxn lists)")]
    for col, (mask, title) in zip(st.columns(3 if unknown.any() else 2), groups):
        with col:
            st.markdown(f"**{title}**")
            part = inv[mask].drop(columns=["symbol"]).dropna(axis=1, how="all")
            if "stake_pct" in part:
                part = part.sort_values("stake_pct", ascending=False)
            st.dataframe(part, hide_index=True, use_container_width=True)


def render(preds: pd.DataFrame) -> None:
    comp = st.selectbox("Company", preds["symbol"], key="company_symbol")
    _header(comp)
    fund = company.for_symbol(uploads.load("fundamentals"), comp).sort_values("as_of")
    _fundamentals(fund)
    _financials(comp)
    _ownership(fund)
    _investors(comp)
    if tracxn.PROFILES_FILE.exists():
        t = company.for_symbol(pd.read_csv(tracxn.PROFILES_FILE), comp)
        if len(t):
            with st.expander("From uploaded Tracxn PDFs"):
                st.dataframe(t, hide_index=True, use_container_width=True)
