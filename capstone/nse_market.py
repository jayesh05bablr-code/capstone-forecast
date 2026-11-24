"""Market-wide daily inputs for the opening-gap model, from NSE's official MCP server (no key).

Saved to data/processed/nse_market_daily.csv, one row per NSE session:
  nifty_open, nifty_close   NIFTY 50 index (get_index_history; NSE keeps it from Nov 2024)
  vix_close                 India VIX (same tool and span)
  advances, declines        whole-market breadth over every NSE equity (get_market_breadth; back to 2021)

`python -m capstone.fetch_data` calls refresh() after the prices, so the file covers every price date.
The first build takes about two minutes (one breadth call per session of the last three years,
eight at a time); after that
each run fetches only the new sessions. If NSE cannot be reached the old file is kept, and the model
simply sees these inputs as missing for the newest rows (its trees handle missing values).

FEATURES are what the model gets for a day t, all known at t's close (no look-ahead):
  nifty_ret_1       NIFTY 50 close-to-close return
  nifty_gap_today   NIFTY 50 open vs previous close (how the index itself gapped that morning)
  vix_level         India VIX close
  vix_chg_1         India VIX change (relative)
  breadth_log_ad    log(advances / declines): above 0 = more stocks rose than fell
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import nse_mcp
from .paths import PROCESSED

FILE = PROCESSED / "nse_market_daily.csv"
FEATURES = ["nifty_ret_1", "nifty_gap_today", "vix_level", "vix_chg_1", "breadth_log_ad"]
COLUMNS = ["date", "nifty_open", "nifty_close", "vix_close", "advances", "declines"]
WORKERS = 8
BREADTH_YEARS = 3  # breadth for older sessions adds minutes to a first start for little gain
_local = threading.local()


def _call(tool: str, args: dict):
    """One tool call on this thread's own session (the shared one serialises), None on any failure."""
    try:
        if not hasattr(_local, "s"):
            _local.s = nse_mcp._Session(nse_mcp.BHAVCOPY_URL, timeout=30)
        res = _local.s.request("tools/call", {"name": tool, "arguments": args}, retries=2)
        if res.get("isError"):
            return None
        text = "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        return json.loads(text)
    except Exception:
        return None


def load() -> pd.DataFrame:
    if not FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(FILE, parse_dates=["date"])


def index_history(name: str, since: date | None = None) -> pd.DataFrame:
    """Daily open/close of one NSE index, walking back in 12-month chunks to `since` or NSE's first day."""
    rows, end = {}, date.today()
    for _ in range(6):
        r = _call("get_index_history", {"indexName": name, "months": 12, "endDate": end.isoformat()})
        data = (r or {}).get("data") if isinstance(r, dict) else None
        if not data:
            break
        for d in data:
            rows[d["date"]] = (d.get("open"), d.get("close"))
        first = date.fromisoformat(min(x["date"] for x in data))
        if since is not None and first <= since:
            break
        end = first - timedelta(days=1)
    df = pd.DataFrame([(k, *v) for k, v in rows.items()], columns=["date", "open", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _breadth(days: list[date]) -> pd.DataFrame:
    with ThreadPoolExecutor(WORKERS) as ex:
        found = list(ex.map(lambda d: _call("get_market_breadth", {"date": d.isoformat()}), days))
    rows = [(r["date"], r.get("advances"), r.get("declines")) for r in found if isinstance(r, dict) and r.get("date")]
    df = pd.DataFrame(rows, columns=["date", "advances", "declines"])
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date")


def refresh(dates) -> pd.DataFrame:
    """Bring the file up to date for these session dates (the price history's). Returns the saved table."""
    have = load()
    want = sorted({pd.Timestamp(d).normalize() for d in dates})
    if not want:
        return have
    last_ix = have.dropna(subset=["nifty_close"])["date"].max() if len(have) else None
    since = (last_ix - pd.Timedelta(days=10)).date() if last_ix is not None and pd.notna(last_ix) else None
    nifty, vix = index_history("NIFTY 50", since), index_history("INDIA VIX", since)
    have_b = set(have.dropna(subset=["advances"])["date"]) if len(have) else set()
    oldest = pd.Timestamp.today().normalize() - pd.DateOffset(years=BREADTH_YEARS)
    todo = [d.date() for d in want if d not in have_b and d >= oldest]
    breadth = _breadth(todo) if todo else pd.DataFrame(columns=["date", "advances", "declines"])

    new =pd.DataFrame(index=pd.DatetimeIndex(sorted(set(nifty["date"]) | set(vix["date"]) | set(breadth["date"])),
                                              name="date"), columns=COLUMNS[1:], dtype=float)
    if len(nifty):
        new.loc[nifty["date"], ["nifty_open", "nifty_close"]] = nifty[["open", "close"]].to_numpy()
    if len(vix):
        new.loc[vix["date"], "vix_close"] = vix["close"].to_numpy()
    if len(breadth):
        new.loc[breadth["date"], ["advances", "declines"]] = breadth[["advances", "declines"]].to_numpy(dtype=float)
    old = have.set_index("date")[COLUMNS[1:]].astype(float) if len(have) else None
    out = new.combine_first(old) if old is not None else new  # NSE's newer answer wins where both have one
    out = out.sort_index().reset_index()
    if len(out):
        FILE.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(FILE, index=False)
    return out


def features(daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-date FEATURES (date + one column each), from the saved table. Empty frame if none saved."""
    d = load() if daily is None else daily.copy()
    if d.empty:
        return pd.DataFrame(columns=["date", *FEATURES])
    d = d.sort_values("date").reset_index(drop=True)
    ix = d.dropna(subset=["nifty_close"])
    prev = ix["nifty_close"].shift(1)
    d.loc[ix.index, "nifty_ret_1"] = ix["nifty_close"] / prev - 1
    d.loc[ix.index, "nifty_gap_today"] = ix["nifty_open"] / prev - 1
    vx = d.dropna(subset=["vix_close"])
    d.loc[vx.index, "vix_level"] = vx["vix_close"]
    d.loc[vx.index, "vix_chg_1"] = vx["vix_close"] / vx["vix_close"].shift(1) - 1
    adv, dec = d["advances"].astype(float), d["declines"].astype(float)
    d["breadth_log_ad"] = np.log((adv + 1) / (dec + 1))
    for c in FEATURES:
        if c not in d:
            d[c] = np.nan
    return d[["date", *FEATURES]]
