"""Which days NSE trades, and what to tell the reader when it does not.

NSE (the National Stock Exchange of India) is closed on Saturdays, Sundays and a list of trading
holidays it publishes each December for the following year (for example Guru Nanak Jayanti on
Tue 24 Nov 2026). The app needs this to:

- say "Market closed: <holiday>" instead of "Market open" on a holiday,
- label prices as the last session's close when the market is shut, and
- make every forecast target the next day NSE actually trades.

Sources, best first:
1. NSE's own holiday list (www.nseindia.com/api/holiday-master?type=trading, segment "CM",
   the cash market). `update_from_nse()` downloads it once a day during the automatic update and
   saves it to data/processed/nse_holidays_online.csv. NSE sometimes blocks cloud servers, so
   this is best effort.
2. data/nse_holidays.csv, a copy of NSE's list bundled with the app (2026 copied from NSE on
   24 Nov 2026). NSE publishes the 2027 list in December 2026: add it here when it is out, or
   the online download above picks it up by itself.

A year with no list at all is treated as weekends-only, and `covered()` says so, so the page can
warn that holidays for that year are not known yet.

The US calendar (NYSE holidays) is here too, because the forecast needs the last US session
before India's next open: when the US is shut (Thanksgiving), there is no new overnight move.
"""
from __future__ import annotations

import http.cookiejar
import json
import threading
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import pandas as pd

from .paths import DATA, PROCESSED

BUNDLED_FILE = DATA / "nse_holidays.csv"
ONLINE_FILE = PROCESSED / "nse_holidays_online.csv"
NSE_HOME = "https://www.nseindia.com/"
NSE_API = "https://www.nseindia.com/api/holiday-master?type=trading"
OPEN, CLOSE = time(9, 15), time(15, 30)
PRE_OPEN = time(9, 0)

# NYSE full-day closures (nyse.com/markets/hours-calendars). Weekends are added in code.
US_HOLIDAYS = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
    date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26), date(2027, 5, 31),
    date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25), date(2027, 12, 24),
}

_cache: dict = {"key": None, "days": {}}
_lock = threading.Lock()


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return pd.Timestamp(d).date()


def _read(path) -> dict[date, str]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        return {pd.Timestamp(r.date).date(): str(r.name).strip() for r in df.itertuples()}
    except Exception:  # a half-written or damaged file must never break the page
        return {}


def holidays() -> dict[date, str]:
    """NSE trading holidays {date: name}: the bundled list, with NSE's online list on top."""
    key = tuple(p.stat().st_mtime if p.exists() else 0 for p in (BUNDLED_FILE, ONLINE_FILE))
    with _lock:
        if _cache["key"] != key:
            _cache["days"] = {**_read(BUNDLED_FILE), **_read(ONLINE_FILE)}
            _cache["key"] = key
        return _cache["days"]


def covered(d) -> bool:
    """True if a holiday list for d's year is known (otherwise only weekends count as closed)."""
    year = _as_date(d).year
    return any(h.year == year for h in holidays())


def holiday_name(d) -> str | None:
    """Why NSE is shut on d ('Saturday', 'Christmas', ...), or None on a trading day."""
    d = _as_date(d)
    if d.weekday() >= 5:
        return d.strftime("%A")
    return holidays().get(d)


def is_trading_day(d) -> bool:
    return holiday_name(d) is None


def next_trading_day(d) -> date:
    """The first NSE trading day strictly after d."""
    d = _as_date(d) + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def previous_trading_day(d) -> date:
    """The last NSE trading day strictly before d."""
    d = _as_date(d) - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def us_trading_day(d) -> bool:
    d = _as_date(d)
    return d.weekday() < 5 and d not in US_HOLIDAYS


def last_us_session_before(d) -> date:
    """The last US trading day strictly before d (its close lands before India's open on d)."""
    d = _as_date(d) - timedelta(days=1)
    while not us_trading_day(d):
        d -= timedelta(days=1)
    return d


@dataclass
class Status:
    state: str            # "open", "pre_open", "before_open", "after_close", "holiday", "weekend"
    is_open: bool         # True from 09:15 to 15:30 IST on a trading day
    reason: str | None    # holiday or weekend name when closed all day
    last_session: date    # the latest day whose trading has started (today once 09:00 pre-open begins)
    next_session: date    # the next day the 09:15 open is still ahead
    headline: str         # short line, e.g. "Market closed today: Christmas"
    detail: str           # one plain sentence on what the prices and forecast refer to


def status(now: datetime | None = None) -> Status:
    """Market state at `now` (naive IST; default: the current time)."""
    now = now or pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).to_pydatetime()
    today = now.date()
    t = now.time()
    day = f"{now:%a %d %b}"
    if not is_trading_day(today):
        reason = holiday_name(today)
        last, nxt = previous_trading_day(today), next_trading_day(today)
        kind = "weekend" if today.weekday() >= 5 else "holiday"
        what = "it's the weekend" if kind == "weekend" else f"NSE holiday for {reason}"
        return Status(kind, False, reason, last, nxt,
                      f"Market closed today ({day}): {what}",
                      f"Prices shown are the last session's close ({last:%a %d %b}). "
                      f"The next trading day is {nxt:%a %d %b}, and the forecast is for that day's 09:15 open.")
    if t < PRE_OPEN:
        last = previous_trading_day(today)
        return Status("before_open", False, None, last, today, f"Market opens today ({day}) at 09:15 IST",
                      f"Prices shown are the last session's close ({last:%a %d %b}) until trading starts.")
    if t < OPEN:
        return Status("pre_open", False, None, today, today, "Pre-open session (09:00-09:15 IST)",
                      "Orders are being collected; normal trading starts at 09:15 IST.")
    if t <= CLOSE:
        return Status("open", True, None, today, next_trading_day(today), "Market open",
                      "Normal trading until 15:30 IST. Prices are a few minutes behind the exchange.")
    nxt = next_trading_day(today)
    return Status("after_close", False, None, today, nxt, f"Market closed for the day ({day})",
                  f"Prices shown are today's close. The next trading day is {nxt:%a %d %b}.")


def market_open(at: datetime | None = None) -> bool:
    return status(at).is_open


# --- NSE's online list -------------------------------------------------------------------------

def parse_nse(payload: dict) -> pd.DataFrame:
    """Rows of NSE's holiday-master answer for the cash market (segment CM) as date, name."""
    rows = [{"date": pd.to_datetime(r["tradingDate"], format="%d-%b-%Y").date(),
             "name": str(r.get("description", "")).rstrip("*").strip()}
            for r in payload.get("CM", []) if r.get("tradingDate")]
    return pd.DataFrame(rows, columns=["date", "name"])


def update_from_nse(timeout: float = 10, max_age_hours: float = 20) -> str:
    """Download NSE's holiday list if the saved copy is older than max_age_hours. Never raises."""
    if ONLINE_FILE.exists():
        age = (datetime.now().timestamp() - ONLINE_FILE.stat().st_mtime) / 3600
        if age < max_age_hours:
            return "NSE holiday list: saved copy is recent"
    try:
        jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"),
                             ("Accept", "application/json, text/plain, */*"), ("Accept-Language", "en-US,en")]
        try:  # NSE wants the cookies its home page sets before it answers the API
            opener.open(NSE_HOME, timeout=timeout).read(1000)
        except Exception:
            pass
        with opener.open(NSE_API, timeout=timeout) as resp:
            df = parse_nse(json.loads(resp.read().decode("utf-8")))
    except Exception as exc:  # blocked, offline, or NSE changed the format
        return f"NSE holiday list: not reachable ({type(exc).__name__}); using the bundled list"
    if df.empty:
        return "NSE holiday list: empty answer; using the bundled list"
    ONLINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ONLINE_FILE, index=False)
    return f"NSE holiday list: {len(df)} holidays saved from nseindia.com"


def forecast_target_wrong(story: dict | None) -> bool:
    """True when a saved forecast aims at a day NSE does not trade (made before the app knew the
    holiday list, or before NSE added a holiday), so it must be made again."""
    if not story or not story.get("next_open") or not story.get("as_of_close"):
        return False
    target = _as_date(story["next_open"])
    return not is_trading_day(target) or target != next_trading_day(story["as_of_close"])
