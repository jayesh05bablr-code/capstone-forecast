"""Sector intelligence: which sectors lead, which are gaining or losing strength against the NIFTY 50.

Two kinds of data, kept apart on the page:

* NSE's official sector indices (bhavcopy `get_index_history`: NIFTY BANK, NIFTY IT, ...). Their returns,
  momentum, relative strength and volume come straight from NSE's published index values.
* Our own sector groups of the chosen stock universe (data/nifty500_sectors.csv) from the saved daily
  stock history (flow_history): breadth (share of stocks up, above their 20/50-day average), relative
  volume and each sector's contribution to the universe's average move. These are equal-weighted (every
  stock counts once) because free NSE feeds do not publish index weights; the page says so.

Definitions
-----------
return N         : close today / close N sessions ago - 1
relative strength: sector return over 20 sessions minus NIFTY 50 return over 20 sessions (points)
RS momentum      : relative strength today minus relative strength 5 sessions ago (points); > 0 means
                   the sector is gaining strength against the NIFTY 50, < 0 losing it
rotation quadrant: Leading (RS > 0, momentum > 0), Weakening (RS > 0, momentum < 0),
                   Lagging (RS < 0, momentum < 0), Improving (RS < 0, momentum > 0)
relative volume  : average volume of the last 5 sessions / average of the 20 before them
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from capstone import nse_mcp
from capstone.paths import PROCESSED

BENCH = "NIFTY 50"
SECTOR_INDICES = ["NIFTY BANK", "NIFTY PRIVATE BANK", "NIFTY PSU BANK", "NIFTY FINANCIAL SERVICES", "NIFTY CAPITAL MARKETS",
                  "NIFTY IT", "NIFTY AUTO", "NIFTY FMCG", "NIFTY PHARMA", "NIFTY HEALTHCARE INDEX", "NIFTY METAL",
                  "NIFTY ENERGY", "NIFTY OIL & GAS", "NIFTY REALTY", "NIFTY MEDIA", "NIFTY CONSUMER DURABLES",
                  "NIFTY CHEMICALS", "NIFTY CEMENT", "NIFTY PSE", "NIFTY INDIA DEFENCE"]
INDEX_FILE = PROCESSED / "sector_index_history.csv"
QUADRANTS = {(True, True): "Leading", (True, False): "Weakening", (False, False): "Lagging", (False, True): "Improving"}

_state = {"busy": False, "at": 0.0, "error": ""}
_lock = threading.Lock()


# ---------- official sector indices ----------

def _fetch_one(name: str) -> list[dict]:
    s = nse_mcp._Session(nse_mcp.BHAVCOPY_URL, timeout=30)
    res = s.request("tools/call", {"name": "get_index_history", "arguments": {"indexName": name, "months": 6}}, retries=2)
    text = "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
    data = json.loads(text).get("data") or [] if not res.get("isError") else []
    return [{"index": name, "date": r["date"], "open": r.get("open"), "high": r.get("high"), "low": r.get("low"),
             "close": r.get("close"), "volume": r.get("volume"), "turnover_cr": r.get("turnoverCr"),
             "pe": r.get("pe")} for r in data]


def load_indices() -> pd.DataFrame:
    if not INDEX_FILE.exists():
        return pd.DataFrame(columns=["index", "date", "close"])
    return pd.read_csv(INDEX_FILE, parse_dates=["date"])


def refresh_indices(block: bool = False, max_age: float = 6 * 3600) -> None:
    """Download ~6 months of every sector index (and the NIFTY 50) in the background, a few times a day."""
    have = load_indices()
    fresh = INDEX_FILE.exists() and time.time() - INDEX_FILE.stat().st_mtime < max_age
    with _lock:
        if _state["busy"] or (fresh and len(have)):
            return
        _state.update(busy=True, error="")

    def run():
        try:
            rows = []
            with ThreadPoolExecutor(4) as ex:
                for part in ex.map(lambda n: _safe(_fetch_one, n), [BENCH, *SECTOR_INDICES]):
                    rows += part
            if not rows:
                _state["error"] = "NSE's bhavcopy server returned no index history"
                return
            df = pd.DataFrame(rows)
            INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = INDEX_FILE.with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(INDEX_FILE)
            _state["at"] = time.time()
        except Exception as e:
            _state["error"] = str(e)
        finally:
            _state["busy"] = False

    if block:
        run()
    else:
        threading.Thread(target=run, daemon=True).start()


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return []


def status() -> dict:
    return dict(_state)


def _ret(c: pd.Series, n: int) -> float | None:
    return float(100 * (c.iloc[-1] / c.iloc[-1 - n] - 1)) if len(c) > n else None


def leaderboard(ix: pd.DataFrame, bench: str = BENCH) -> pd.DataFrame:
    """One row per official sector index: returns, relative strength, momentum, quadrant, relative volume."""
    if ix.empty or bench not in set(ix["index"]):
        return pd.DataFrame()
    wide = ix.pivot_table(index="date", columns="index", values="close").sort_index()
    vol = ix.pivot_table(index="date", columns="index", values="volume").sort_index()
    b = wide[bench].dropna()
    rows = []
    for name in wide.columns:
        if name == bench:
            continue
        c = wide[name].dropna()
        if len(c) < 30:
            continue
        c, bb = c.align(b, join="inner")
        rs_now = _ret(c, 20) - _ret(bb, 20)
        rs_then = (100 * (c.iloc[-6] / c.iloc[-26] - 1)) - (100 * (bb.iloc[-6] / bb.iloc[-26] - 1))
        v = vol[name].dropna()
        rvol = float(v.tail(5).mean() / v.iloc[-25:-5].mean()) if len(v) >= 25 and v.iloc[-25:-5].mean() > 0 else None
        mom = rs_now - rs_then
        rows.append({"index": name, "close": float(c.iloc[-1]), "ret_1d": _ret(c, 1), "ret_5d": _ret(c, 5),
                     "ret_20d": _ret(c, 20), "ret_60d": _ret(c, 60), "rs_20d": rs_now, "rs_mom": mom,
                     "quadrant": QUADRANTS[(rs_now > 0, mom > 0)], "rvol": rvol, "as_of": c.index[-1]})
    out = pd.DataFrame(rows).sort_values("rs_20d", ascending=False).reset_index(drop=True)
    out["trend"] = np.where(out["rs_mom"] > 0, "Gaining strength", "Losing strength")
    return out


def rotation_trails(ix: pd.DataFrame, names: list[str], points: int = 6, bench: str = BENCH) -> pd.DataFrame:
    """For the rotation chart: each sector's (RS, momentum) on each of the last `points` sessions."""
    wide = ix.pivot_table(index="date", columns="index", values="close").sort_index()
    if bench not in wide:
        return pd.DataFrame()
    r20 = 100 * (wide / wide.shift(20) - 1)
    rs = r20.sub(r20[bench], axis=0)
    mom = rs - rs.shift(5)
    rows = []
    for n in names:
        if n not in rs:
            continue
        t = pd.DataFrame({"rs": rs[n], "mom": mom[n]}).dropna().tail(points)
        t["index"], t["step"] = n, range(len(t))
        rows.append(t.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------- our own sector groups, from stock history ----------

def constituent_stats(hist: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    """Per sector of the universe (equal-weighted): breadth, % above 20/50-day average, relative volume and
    contribution to the universe's average 1-day and 5-day move (contributions add up to that average)."""
    if hist.empty:
        return pd.DataFrame()
    h = hist[hist["symbol"].isin(uni["symbol"])].sort_values(["symbol", "date"])
    last_day = h["date"].max()
    rows = []
    for sym, g in h.groupby("symbol"):
        c = g["close"].astype(float).reset_index(drop=True)
        if len(c) < 6 or g["date"].iloc[-1] != last_day:
            continue
        v = g["volume"].astype(float).reset_index(drop=True)
        rows.append({"symbol": sym, "ret_1d": 100 * (c.iloc[-1] / c.iloc[-2] - 1), "ret_5d": 100 * (c.iloc[-1] / c.iloc[-6] - 1),
                     "above_20": c.iloc[-1] > c.tail(20).mean() if len(c) >= 20 else np.nan,
                     "above_50": c.iloc[-1] > c.tail(50).mean() if len(c) >= 50 else np.nan,
                     "rvol": v.tail(5).mean() / v.iloc[-25:-5].mean() if len(v) >= 25 and v.iloc[-25:-5].mean() > 0 else np.nan})
    s = pd.DataFrame(rows)
    if s.empty:
        return s
    s = s.merge(uni[["symbol", "sector"]], on="symbol")
    n = len(s)
    g = s.groupby("sector")
    out = pd.DataFrame({
        "stocks": g["symbol"].count(), "up_pct": g["ret_1d"].apply(lambda r: 100 * (r > 0).mean()),
        "ret_1d": g["ret_1d"].mean(), "ret_5d": g["ret_5d"].mean(),
        "above_20_pct": g["above_20"].apply(lambda x: 100 * x.dropna().astype(float).mean() if x.notna().any() else np.nan),
        "above_50_pct": g["above_50"].apply(lambda x: 100 * x.dropna().astype(float).mean() if x.notna().any() else np.nan),
        "rvol": g["rvol"].median(),
        "contrib_1d": g["ret_1d"].sum() / n, "contrib_5d": g["ret_5d"].sum() / n}).reset_index()
    out["as_of"] = last_day
    return out.sort_values("contrib_1d", ascending=False).reset_index(drop=True)
