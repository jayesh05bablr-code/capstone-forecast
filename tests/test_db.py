"""Supabase persistence: local mode, validation, upsert requests, duplicate prevention.

The optional integration test replays the same upserts as SQL against a real PostgreSQL that has
migrations/001_init.sql applied. It runs only when CAPSTONE_TEST_PSQL is set to a psql command
line, e.g. CAPSTONE_TEST_PSQL="psql -h /tmp -p 55432 -U postgres -d postgres".
"""
import json
import os
import shlex
import subprocess
import urllib.parse

import numpy as np
import pandas as pd
import pytest

from capstone import backtest, costs, db, montecarlo


class FakeRest:
    """Records requests and keeps upserted rows keyed by their conflict columns."""

    def __init__(self):
        self.calls, self.tables = [], {}

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, headers))
        table, _, query = url.split("/rest/v1/")[1].partition("?")
        if method == "POST":
            keys = urllib.parse.parse_qs(query)["on_conflict"][0].split(",")
            for row in json.loads(body):
                self.tables.setdefault(table, {})[tuple(row[k] for k in keys)] = row
            return 201, None
        return 200, list(self.tables.get(table, {}).values())


def test_local_mode_without_credentials(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    assert db.get_store() is None


def test_validation_cleans_values_and_rejects_missing():
    rows = db.validate("stocks", [{"symbol": "TCS", "fx_exposure": np.float64("nan")}])
    assert rows == [{"symbol": "TCS", "fx_exposure": None}]
    with pytest.raises(ValueError, match="missing symbol"):
        db.validate("stocks", [{"symbol": None}])
    r = db.validate("weekly_performance", [{"symbol": "TCS", "week_start": pd.Timestamp("2026-11-09"),
                                             "entry_date": pd.Timestamp("2026-11-10"), "exit_date": pd.Timestamp("2026-11-13"),
                                             "sessions": np.int64(4), "entry_open": 1.0, "exit_close": 2.0,
                                             "valid": np.bool_(True)}])[0]
    assert r["week_start"] == "2026-11-09" and r["sessions"] == 4 and r["valid"] is True


def test_upsert_sends_conflict_keys_and_auth_headers_and_dedupes():
    fake = FakeRest()
    store = db.SupabaseStore("https://example.supabase.co", "test-key-123", transport=fake)
    preds = pd.DataFrame({"symbol": ["TCS"], "as_of_close": ["2026-11-13"], "close": [2407.5], "pred_gap_pct": [0.05],
                          "pred_open": [2408.8]})
    for _ in range(2):  # saving twice must not create a second row
        store.upsert("daily_predictions", db.prediction_rows(preds, "hgb"))
    method, url, headers = fake.calls[0]
    assert method == "POST" and "on_conflict=symbol%2Cas_of_close%2Cmodel_name" in url
    assert headers["apikey"] == "test-key-123" and "merge-duplicates" in headers["Prefer"]
    assert len(fake.tables["daily_predictions"]) == 1
    assert next(iter(fake.tables["daily_predictions"].values()))["direction"] == "up"


def _analysis():
    sig = pd.DataFrame({"symbol": ["AAA", "AAA"], "week_start": pd.to_datetime(["2026-01-05", "2026-01-12"]),
                        "entry_date": pd.to_datetime(["2026-01-05", "2026-01-12"]),
                        "exit_date": pd.to_datetime(["2026-01-09", "2026-01-16"]),
                        "entry_open": [100.0, 100.0], "exit_close": [103.0, 99.0], "pred": [0.05, 0.05]})
    cfg = backtest.BacktestConfig(initial_capital=100_000)
    t, w = backtest.run(sig, "B", cfg)
    s = backtest.summary(t, w, cfg)
    sims = montecarlo.simulate(w["net_return"], cfg.initial_capital, n_sims=50, block=1)
    weeks = pd.DataFrame({"symbol": ["AAA"], "week_start": pd.to_datetime(["2026-01-05"]),
                          "entry_date": pd.to_datetime(["2026-01-05"]), "exit_date": pd.to_datetime(["2026-01-09"]),
                          "sessions": [5], "entry_open": [100.0], "exit_close": [103.0], "weekly_return": [0.03],
                          "valid": [True], "exclude_reason": [""]})
    overall = pd.DataFrame({"model": ["naive"], "n": [10], "mae_pct": [0.5], "rmse_pct": [0.7], "bias_pct": [0.0],
                            "direction_acc_pct": [2.0]})
    by_stock = overall.assign(symbol="AAA")
    return dict(stock_map=pd.DataFrame({"symbol": ["AAA"], "india_sector": ["IT"], "adr": [""]}), weeks=weeks,
                evaluation=("daily_gap", overall, by_stock, {"test_start": "2020-01-01", "test_end": "2026-01-01"}),
                config={"capital": 100_000, "costs": costs.CostAssumptions().to_dict()},
                backtests=[("B", "hgb", s, t, w)],
                montecarlo={"strategy": "B", "summary": montecarlo.summarise(sims, 100_000), "seed": 42, "block": 1,
                            "horizon": 2, "initial_capital": 100_000})


def test_save_analysis_is_idempotent():
    fake = FakeRest()
    store = db.SupabaseStore("https://example.supabase.co", "k" * 20, transport=fake)
    first = db.save_analysis(store, **_analysis())
    sizes = {t: len(v) for t, v in fake.tables.items()}
    db.save_analysis(store, **_analysis())
    assert {t: len(v) for t, v in fake.tables.items()} == sizes  # same deterministic ids, no duplicates
    assert first["backtest_trades"] == 2 and sizes["monte_carlo_runs"] == 1 and sizes["model_metrics"] == 2


def test_errors_are_redacted(monkeypatch):
    monkeypatch.setenv("SUPABASE_KEY", "secret-service-key-999")
    assert "secret-service-key-999" not in db.env.redact("HTTP 401 key=secret-service-key-999")


@pytest.mark.skipif(not os.environ.get("CAPSTONE_TEST_PSQL"), reason="needs a PostgreSQL with the migration applied")
def test_rows_fit_the_real_schema():
    psql = shlex.split(os.environ["CAPSTONE_TEST_PSQL"]) + ["-v", "ON_ERROR_STOP=1", "-q", "-At"]

    def sql_transport(method, url, headers, body):
        table, _, query = url.split("/rest/v1/")[1].partition("?")
        keys = urllib.parse.parse_qs(query)["on_conflict"][0]
        rows = json.loads(body)
        cols = sorted({k for r in rows for k in r})
        payload = json.dumps(rows).replace("'", "''")
        upd = ", ".join(f"{c} = excluded.{c}" for c in cols if c not in keys.split(","))
        stmt = (f"insert into {table} ({', '.join(cols)}) select {', '.join(cols)} from "
                f"jsonb_populate_recordset(null::{table}, '{payload}'::jsonb) "
                f"on conflict ({keys}) do {'update set ' + upd if upd else 'nothing'};")
        subprocess.run(psql + ["-c", stmt], check=True, capture_output=True, text=True)
        return 201, None

    store = db.SupabaseStore("https://example.supabase.co", "k" * 20, transport=sql_transport)
    for _ in range(2):
        db.save_analysis(store, **_analysis())
    out = subprocess.run(psql + ["-c", "select count(*) from backtest_trades; select count(*) from monte_carlo_runs;"],
                         check=True, capture_output=True, text=True).stdout.split()
    assert out == ["2", "1"]
