"""NSE trading calendar: holidays, weekends, the next trading day and the plain-language status."""
from datetime import date, datetime

import pandas as pd

from capstone import guidance, live, market_calendar as mc


def test_guru_nanak_jayanti_2026_is_closed():
    assert mc.holiday_name(date(2026, 11, 24)) == "Prakash Gurpurb Sri Guru Nanak Dev"
    assert not mc.is_trading_day("2026-11-24")
    assert mc.is_trading_day("2026-11-23") and mc.is_trading_day("2026-11-25")


def test_next_and_previous_trading_day_skip_holidays_and_weekends():
    assert mc.next_trading_day("2026-11-23") == date(2026, 11, 25)   # Mon -> Wed (Tue holiday)
    assert mc.next_trading_day("2026-12-24") == date(2026, 12, 28)   # Thu -> Mon (Fri Christmas)
    assert mc.next_trading_day("2026-11-20") == date(2026, 11, 23)   # Fri -> Mon
    assert mc.previous_trading_day("2026-11-25") == date(2026, 11, 23)


def test_status_on_the_holiday_says_closed_and_points_to_next_day():
    s = mc.status(datetime(2026, 11, 24, 10, 54))
    assert s.state == "holiday" and not s.is_open
    assert "Guru Nanak" in s.headline
    assert s.last_session == date(2026, 11, 23) and s.next_session == date(2026, 11, 25)
    assert "Mon 23 Nov" in s.detail and "Wed 25 Nov" in s.detail
    assert not live.market_open(datetime(2026, 11, 24, 10, 54))


def test_status_through_a_normal_day():
    assert mc.status(datetime(2026, 11, 25, 8, 0)).state == "before_open"
    assert mc.status(datetime(2026, 11, 25, 9, 5)).state == "pre_open"
    assert mc.status(datetime(2026, 11, 25, 11, 0)).is_open
    after = mc.status(datetime(2026, 11, 25, 16, 0))
    assert after.state == "after_close" and after.next_session == date(2026, 11, 26)
    assert mc.status(datetime(2026, 11, 28, 11, 0)).state == "weekend"


def test_us_calendar_for_the_overnight_signal():
    assert mc.last_us_session_before("2026-11-25") == date(2026, 11, 24)
    assert mc.last_us_session_before("2026-11-27") == date(2026, 11, 25)  # Thanksgiving 26 Nov
    assert mc.last_us_session_before("2026-11-23") == date(2026, 11, 20)


def test_forecast_target_wrong_catches_a_holiday_target():
    assert mc.forecast_target_wrong({"as_of_close": "2026-11-23", "next_open": "2026-11-24"})
    assert not mc.forecast_target_wrong({"as_of_close": "2026-11-23", "next_open": "2026-11-25"})
    assert not mc.forecast_target_wrong(None)


def test_freshness_on_a_holiday():
    now = datetime(2026, 11, 24, 10, 54)
    status, msg = guidance.freshness("2026-11-24", True, now)
    assert status == "old" and "closed" in msg and "Wed 25 Nov" in msg
    status, msg = guidance.freshness("2026-11-25", True, now)
    assert status == "ok" and "closed today" in msg and "Wed 25 Nov 2026" in msg


def test_parse_nse_holiday_master():
    payload = {"CM": [{"tradingDate": "24-Nov-2026", "weekDay": "Tuesday",
                       "description": "Prakash Gurpurb Sri Guru Nanak Dev"},
                      {"tradingDate": "08-Nov-2026", "description": "Diwali Laxmi Pujan*"}],
               "FO": [{"tradingDate": "01-Jan-2026", "description": "x"}]}
    df = mc.parse_nse(payload)
    assert list(df["date"]) == [date(2026, 11, 24), date(2026, 11, 8)]
    assert df["name"].iloc[1] == "Diwali Laxmi Pujan"


def test_live_rows_do_not_check_a_forecast_against_an_older_session():
    from capstone.ui.tab_scalping import Settings  # noqa: F401  (import check only)
    assert live._same_day(pd.Timestamp("2026-11-23 15:29"), "2026-11-25") is False
    assert live._same_day(pd.Timestamp("2026-11-25 09:20"), "2026-11-25") is True
    assert live._same_day(pd.Timestamp("2026-11-25 09:20"), float("nan")) is True
