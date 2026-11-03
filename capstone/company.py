"""Company profile calculations: profile facts, shareholding and investor classification.

Pure pandas, no Streamlit, so each piece can be unit tested.
"""
from __future__ import annotations

import pandas as pd

FUNDAMENTAL_COLUMNS = ["market_cap_cr", "pe", "pb", "eps_ttm", "roe_pct", "debt_to_equity"]
SHAREHOLDING_COLUMNS = ["promoter_pct", "fii_pct", "dii_pct", "public_pct", "govt_pct"]
DOMESTIC_HOLDERS = ["promoter_pct", "dii_pct", "public_pct", "govt_pct"]
PROFILE_FACTS = [("founded", "Founded"), ("headquarters", "HQ"), ("employees", "Employees"),
                 ("ceo", "CEO"), ("website", "Website")]


def for_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Rows of an uploaded table that belong to one stock (symbols matched case-insensitively)."""
    return df[df["symbol"].astype(str).str.upper() == symbol]


def profile_facts(info: pd.Series | None, profile: pd.DataFrame) -> list[str]:
    """Short 'Label: **value**' facts from the stock map row and the latest uploaded profile."""
    facts = []
    if info is not None:
        facts.append(f"Sector: **{info['india_sector']}**")
        facts.append(f"US twin: **{info['us_sector']} ({info['us_etf']})**")
        if info["adr"]:
            facts.append(f"US-listed ADR: **{info['adr']}**")
    if len(profile):
        p0 = profile.iloc[-1]
        for col, label in PROFILE_FACTS:
            if col in p0 and pd.notna(p0[col]):
                v = p0[col]
                if col in ("founded", "employees") and isinstance(v, (int, float)):
                    v = f"{int(v)}" if col == "founded" else f"{int(v):,}"
                facts.append(f"{label}: **{v}**")
    return facts


def shareholding(fund: pd.DataFrame) -> pd.DataFrame:
    """Shareholding pattern by filing date (only filings with at least one holder %)."""
    if fund.empty:
        return pd.DataFrame()
    cols = [c for c in SHAREHOLDING_COLUMNS if c in fund]
    return fund[["as_of", *cols]].dropna(how="all", subset=cols)


def tracxn_ownership(fund: pd.DataFrame) -> pd.DataFrame:
    """Insider vs institution split from Tracxn exports, which do not split foreign/domestic."""
    if not {"insiders_pct", "institutions_pct"} <= set(fund.columns):
        return pd.DataFrame()
    return fund[["as_of", "insiders_pct", "institutions_pct"]].dropna(
        subset=["insiders_pct", "institutions_pct"], how="all")


def domestic_total(filing: pd.Series) -> float:
    """Sum of every domestic holder group in one filing (missing groups count as 0)."""
    return sum(filing.get(c, 0) or 0 for c in DOMESTIC_HOLDERS if c in filing)


def change_vs_previous(last: pd.Series, prev: pd.Series | None, col: str) -> str | None:
    """'+0.70 pp vs previous filing', or None when either filing lacks the number."""
    if prev is None or pd.isna(prev.get(col)) or pd.isna(last.get(col)):
        return None
    return f"{last[col] - prev[col]:+.2f} pp vs previous filing"


def classify_investors(inv: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Boolean masks (foreign, domestic, unknown) over an investors table.

    Foreign: type says FII/FPI/Foreign, or a country other than India is given.
    Domestic: not foreign, and either a country (India) is given or the type is a domestic kind.
    Unknown: neither, e.g. Tracxn investor lists without a country.
    """
    itype = inv.get("investor_type", pd.Series("", index=inv.index)).astype(str)
    country = inv.get("country", pd.Series("", index=inv.index)).astype(str).str.strip().str.lower()
    no_country = country.isin(["", "nan"])
    foreign = itype.str.contains("FII|FPI|Foreign", case=False) | (~no_country & (country != "india"))
    domestic = ~foreign & (~no_country | itype.str.contains(
        "DII|Promoter|Mutual|Insurance|Government|Retail", case=False))
    unknown = ~foreign & ~domestic
    return foreign, domestic, unknown
