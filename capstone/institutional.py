"""Institutional activity: FII/FPI and DII cash-market flows, delivery %, bulk and block deals.

These are REPORTED figures, published by NSE after the session, unlike the estimated buyer/seller
pressure on the Market flow page. None of them is in NSE's MCP feeds, so they come from NSE's own
website files:

* FII/DII      : www.nseindia.com/api/fiidiiTradeReact (the table on nseindia.com/reports/fii-dii;
                 provisional, NSE capital market segment)
* Bulk / block : nsearchives.nseindia.com/content/equities/bulk.csv and block.csv (latest session)
* Delivery     : nsearchives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv (full bhavcopy
                 with DELIV_QTY and DELIV_PER)

NSE's site sometimes refuses requests from cloud servers. Every fetch is best effort: on failure the
page says the data is unavailable and offers an upload of the same CSV that NSE's website lets anyone
download. Whatever is fetched or uploaded is saved under data/processed/ so history builds up day by
day (Streamlit Cloud clears these files when the app reboots).
"""
from __future__ import annotations

import http.cookiejar
import io
import re
import threading
import time
import urllib.request
from datetime import date

import pandas as pd

from capstone import market_calendar
from capstone.paths import PROCESSED

HOME = "https://www.nseindia.com/"
FII_API = "https://www.nseindia.com/api/fiidiiTradeReact"
BULK_CSV = "https://nsearchives.nseindia.com/content/equities/bulk.csv"
BLOCK_CSV = "https://nsearchives.nseindia.com/content/equities/block.csv"
DELIVERY_CSV = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{:%d%m%Y}.csv"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
FILES = {"fii_dii": PROCESSED / "fii_dii.csv", "deals": PROCESSED / "bulk_block_deals.csv",
         "delivery": PROCESSED / "delivery.csv"}
SOURCES = {"fii_dii": "NSE FII/FPI & DII trading activity (provisional), nseindia.com/reports/fii-dii",
           "deals": "NSE bulk and block deals, nsearchives.nseindia.com",
           "delivery": "NSE full bhavcopy with delivery, nsearchives.nseindia.com"}

_lock = threading.Lock()
_state: dict = {}  # name -> {"at": time of last try, "error": str}
RETRY_AFTER = 30 * 60  # after a failed live fetch, wait before trying NSE again


class NseBlocked(Exception):
    pass


# ---------- parsing (pure; also used for uploads) ----------

def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def _category(v: str) -> str | None:
    v = str(v).upper()
    if "FII" in v or "FPI" in v:
        return "FII/FPI"
    if "DII" in v:
        return "DII"
    return None


def parse_fii_dii(obj) -> pd.DataFrame:
    """NSE's JSON (list of {category, date, buyValue, sellValue, netValue}) or the page's CSV download."""
    df = pd.DataFrame(obj) if isinstance(obj, list) else pd.read_csv(io.StringIO(obj) if isinstance(obj, str) else obj)
    cols = {_norm(c): c for c in df.columns}
    pick = lambda *names: next((cols[n] for n in names if n in cols), None)  # noqa: E731
    cat, dt, buy, sell, net = (pick("category"), pick("date"), pick("buyvalue", "buyvaluecrores", "buyvaluecrore"),
                               pick("sellvalue", "sellvaluecrores", "sellvaluecrore"),
                               pick("netvalue", "netvaluecrores", "netvaluecrore"))
    if not all([cat, dt, buy, sell]):
        raise ValueError("Not an NSE FII/DII file: expected CATEGORY, DATE, BUY VALUE, SELL VALUE columns.")
    out = pd.DataFrame({"category": df[cat].map(_category), "date": pd.to_datetime(df[dt], format="mixed", dayfirst=True),
                        "buy_cr": _num(df[buy]), "sell_cr": _num(df[sell])})
    out["net_cr"] = _num(df[net]) if net else out["buy_cr"] - out["sell_cr"]
    return out.dropna(subset=["category", "date", "buy_cr"]).reset_index(drop=True)


def parse_deals(text_or_file, kind: str) -> pd.DataFrame:
    """bulk.csv / block.csv: Date, Symbol, Security Name, Client Name, Buy/Sell, Quantity Traded, Trade Price."""
    df = pd.read_csv(io.StringIO(text_or_file) if isinstance(text_or_file, str) else text_or_file)
    cols = {_norm(c): c for c in df.columns}
    need = {"date": "date", "symbol": "symbol", "securityname": "name", "clientname": "client", "buysell": "side",
            "quantitytraded": "qty"}
    price = next((cols[c] for c in cols if c.startswith("tradeprice") or c.startswith("wghtavgprice")), None)
    missing = [k for k in need if k not in cols]
    if missing or price is None:
        raise ValueError(f"Not an NSE {kind} deals file (missing {', '.join(missing) or 'Trade Price'}).")
    out = pd.DataFrame({v: df[cols[k]] for k, v in need.items()})
    out["price"] = _num(df[price])
    out["qty"] = _num(out["qty"])
    out["date"] = pd.to_datetime(out["date"], format="mixed", dayfirst=True)
    out["side"] = out["side"].astype(str).str.strip().str.upper().str[0].map({"B": "Buy", "S": "Sell"})
    out["value_cr"] = out["qty"] * out["price"] / 1e7
    out["kind"] = kind
    for c in ("symbol", "name", "client"):
        out[c] = out[c].astype(str).str.strip()
    return out.dropna(subset=["date", "qty", "price"]).reset_index(drop=True)


def parse_delivery(text_or_file) -> pd.DataFrame:
    """sec_bhavdata_full: SYMBOL, SERIES, DATE1, ..., CLOSE_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, DELIV_QTY, DELIV_PER."""
    df = pd.read_csv(io.StringIO(text_or_file) if isinstance(text_or_file, str) else text_or_file, skipinitialspace=True)
    df.columns = [str(c).strip().upper() for c in df.columns]
    need = ["SYMBOL", "SERIES", "DATE1", "CLOSE_PRICE", "TTL_TRD_QNTY", "DELIV_QTY", "DELIV_PER"]
    if any(c not in df.columns for c in need):
        raise ValueError("Not an NSE full bhavcopy: expected SYMBOL, SERIES, DATE1, DELIV_QTY, DELIV_PER columns.")
    df = df[df["SERIES"].astype(str).str.strip().isin(["EQ", "BE"])]
    out = pd.DataFrame({"symbol": df["SYMBOL"].astype(str).str.strip(),
                        "date": pd.to_datetime(df["DATE1"].astype(str).str.strip(), format="mixed", dayfirst=True),
                        "close": _num(df["CLOSE_PRICE"]), "volume": _num(df["TTL_TRD_QNTY"]),
                        "deliv_qty": _num(df["DELIV_QTY"]), "deliv_pct": _num(df["DELIV_PER"])})
    if "TURNOVER_LACS" in df.columns:
        out["value_cr"] = _num(df["TURNOVER_LACS"]) / 100
    out["deliv_value_cr"] = out["deliv_qty"] * out["close"] / 1e7
    return out.dropna(subset=["deliv_pct"]).reset_index(drop=True)


# ---------- storage ----------

def load(name: str) -> pd.DataFrame:
    f = FILES[name]
    return pd.read_csv(f, parse_dates=["date"]) if f.exists() else pd.DataFrame()


def save(name: str, new: pd.DataFrame, keys: list[str], keep_days: int = 120) -> pd.DataFrame:
    old = load(name)
    both = pd.concat([old, new], ignore_index=True) if len(old) else new.copy()
    both["date"] = pd.to_datetime(both["date"])
    both = both.drop_duplicates(keys, keep="last").sort_values(["date"] + [k for k in keys if k != "date"])
    both = both[both["date"] >= both["date"].max() - pd.Timedelta(days=keep_days)]
    f = FILES[name]
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    both.to_csv(tmp, index=False)
    tmp.replace(f)
    return both


KEYS = {"fii_dii": ["date", "category"], "deals": ["date", "kind", "symbol", "client", "side", "qty", "price"],
        "delivery": ["date", "symbol"]}


def store_upload(name: str, file) -> pd.DataFrame:
    """Parse an uploaded NSE CSV (raises ValueError with a plain message if it is the wrong file)."""
    if name == "fii_dii":
        df = parse_fii_dii(file)
    elif name == "delivery":
        df = parse_delivery(file)
    else:
        text = file.read().decode("utf-8", "replace") if hasattr(file, "read") else file
        kind = "block" if "block" in str(getattr(file, "name", "")).lower() else "bulk"
        df = parse_deals(text, kind)
    return save(name, df, KEYS[name])


# ---------- live fetch from NSE (best effort) ----------

def _opener():
    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    op.addheaders = [("User-Agent", UA), ("Accept", "*/*"), ("Accept-Language", "en-US,en;q=0.9"),
                     ("Referer", "https://www.nseindia.com/reports/fii-dii")]
    return op


def _get(op, url: str, timeout: float = 8) -> bytes:
    try:
        with op.open(url, timeout=timeout) as r:
            return r.read()
    except Exception as e:  # HTTP 401/403 from NSE's bot protection, timeouts, DNS, ...
        raise NseBlocked(f"{url.split('/')[2]} did not answer ({getattr(e, 'code', '') or type(e).__name__})") from None


def fetch_fii_dii() -> pd.DataFrame:
    import json
    op = _opener()
    try:
        _get(op, HOME, 6)  # NSE's API needs the cookies its home page sets
    except NseBlocked:
        pass
    return parse_fii_dii(json.loads(_get(op, FII_API).decode("utf-8")))


def fetch_deals() -> pd.DataFrame:
    op = _opener()
    parts = [parse_deals(_get(op, url).decode("utf-8", "replace"), kind)
             for kind, url in (("bulk", BULK_CSV), ("block", BLOCK_CSV))]
    return pd.concat(parts, ignore_index=True)


def fetch_delivery(day: date | None = None) -> pd.DataFrame:
    from capstone.flow_history import expected_last_session
    day = day or expected_last_session()
    return parse_delivery(_get(_opener(), DELIVERY_CSV.format(day)).decode("utf-8", "replace"))


FETCHERS = {"fii_dii": fetch_fii_dii, "deals": fetch_deals, "delivery": fetch_delivery}


def latest_date(name: str):
    d = load(name)
    return None if d.empty else pd.Timestamp(d["date"].max())


def refresh(name: str, force: bool = False) -> dict:
    """Try NSE once (at most every 30 minutes after a failure); returns {"ok", "error", "rows"}."""
    from capstone.flow_history import expected_last_session
    have = latest_date(name)
    if not force and have is not None and have.date() >= expected_last_session():
        return {"ok": True, "error": "", "rows": 0}
    with _lock:
        st = _state.setdefault(name, {"at": 0.0, "error": "", "busy": False})
        if st["busy"] or (not force and st["error"] and time.time() - st["at"] < RETRY_AFTER):
            return {"ok": False, "error": st["error"], "rows": 0}
        st["busy"] = True
    try:
        df = FETCHERS[name]()
        save(name, df, KEYS[name])
        st.update(at=time.time(), error="")
        return {"ok": True, "error": "", "rows": len(df)}
    except (NseBlocked, ValueError) as e:
        st.update(at=time.time(), error=str(e))
        return {"ok": False, "error": str(e), "rows": 0}
    finally:
        st["busy"] = False


# ---------- summaries (pure) ----------

def fii_summary(df: pd.DataFrame) -> dict:
    """Latest day's net FII/FPI and DII flows plus 5- and 20-session sums when that many days are saved."""
    if df.empty:
        return {}
    w = df.pivot_table(index="date", columns="category", values="net_cr", aggfunc="last").sort_index()
    out = {"as_of": w.index[-1], "days": len(w)}
    for c in ("FII/FPI", "DII"):
        if c in w:
            s = w[c].dropna()
            out[c] = {"net": float(s.iloc[-1]), "sum5": float(s.tail(5).sum()) if len(s) >= 5 else None,
                      "sum20": float(s.tail(20).sum()) if len(s) >= 20 else None}
    return out


def delivery_view(dl: pd.DataFrame, uni: pd.DataFrame, avg_days: int = 20) -> pd.DataFrame:
    """Latest session's delivery % per universe stock, with its own average when enough days are saved."""
    if dl.empty:
        return pd.DataFrame()
    d = dl[dl["symbol"].isin(uni["symbol"])].sort_values(["symbol", "date"])
    last = d["date"].max()
    today = d[d["date"] == last].set_index("symbol")
    prior = d[d["date"] < last].groupby("symbol").tail(avg_days)
    n = prior.groupby("symbol")["deliv_pct"].count()
    avg = prior.groupby("symbol")["deliv_pct"].mean()[n >= 5]
    out = today.join(avg.rename("deliv_pct_avg")).reset_index()
    out["deliv_vs_avg"] = out["deliv_pct"] - out["deliv_pct_avg"]
    return uni[["symbol", "name", "sector"]].merge(out, on="symbol")


def is_published_yet(now=None) -> bool:
    """NSE posts FII/DII around 17:00-19:00 IST and the full bhavcopy around 18:00 IST on trading days."""
    now = now or pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    return not market_calendar.is_trading_day(now.date()) or now.hour >= 19
