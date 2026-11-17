"""Optional Supabase persistence for the analytical outputs.

Uses Supabase's REST API (PostgREST) directly with the standard library, so no extra package is
needed. Configure with two environment variables (or Streamlit Secrets):

    SUPABASE_URL   https://<project>.supabase.co
    SUPABASE_KEY   the service_role key (server side only; it can write)

Without them `get_store()` returns None and everything runs in local mode (CSV files). Every
write is an upsert on the table's natural key, so saving the same result twice changes nothing
(duplicate prevention). Run ids are deterministic hashes of the inputs for the same reason.
Errors raise SupabaseError with the key redacted; callers treat persistence as best-effort.
Schema: migrations/001_init.sql.
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import date, datetime
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from . import env

BATCH = 500
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "capstone-forecast")

# Natural keys used for upserts (must match the unique constraints in the migration).
CONFLICT_KEYS = {
    "stocks": "symbol",
    "daily_predictions": "symbol,as_of_close,model_name",
    "evaluation_runs": "run_id",
    "model_metrics": "run_id,model,symbol",
    "strategy_configs": "config_hash",
    "weekly_performance": "symbol,week_start",
    "backtest_runs": "run_id",
    "backtest_trades": "run_id,symbol,week_start",
    "monte_carlo_runs": "run_id",
}
REQUIRED = {
    "stocks": ["symbol"],
    "daily_predictions": ["symbol", "as_of_close", "model_name", "close", "pred_gap_pct", "pred_open"],
    "evaluation_runs": ["run_id", "kind"],
    "model_metrics": ["run_id", "model", "symbol", "n"],
    "strategy_configs": ["config_hash", "config"],
    "weekly_performance": ["symbol", "week_start", "entry_date", "exit_date", "sessions", "entry_open",
                           "exit_close", "valid"],
    "backtest_runs": ["run_id", "config_hash", "strategy", "signal_model"],
    "backtest_trades": ["run_id", "symbol", "week_start", "entry_date", "exit_date", "qty", "entry_price",
                        "exit_price", "gross_pnl", "total_cost", "net_pnl"],
    "monte_carlo_runs": ["run_id", "backtest_run_id", "simulations", "seed", "block_weeks", "horizon_weeks",
                         "initial_capital", "percentiles"],
}


class SupabaseError(RuntimeError):
    pass


def _value(v):
    """JSON-safe value: NaN/inf -> None, dates -> ISO strings, numpy -> Python."""
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d") if (v.hour, v.minute, v.second) == (0, 0, 0) else v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, dict):
        return {k: _value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_value(x) for x in v]
    return v


def validate(table: str, rows: Iterable[dict]) -> list[dict]:
    """Clean every value and check required fields; raises ValueError naming the first bad row."""
    out = []
    for i, row in enumerate(rows):
        clean = {k: _value(v) for k, v in row.items()}
        missing = [k for k in REQUIRED.get(table, []) if clean.get(k) is None]
        if missing:
            raise ValueError(f"{table} row {i}: missing {', '.join(missing)}")
        out.append(clean)
    return out


def stable_id(*parts) -> str:
    """Deterministic UUID from any JSON-serialisable inputs."""
    return str(uuid.uuid5(NAMESPACE, json.dumps(_value(list(parts)), sort_keys=True, default=str)))


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(_value(config), sort_keys=True).encode()).hexdigest()


class SupabaseStore:
    """Minimal PostgREST client: upsert and select."""

    def __init__(self, url: str, key: str, transport: Callable | None = None):
        self.base = url.rstrip("/") + "/rest/v1"
        self._key = key
        self._transport = transport or self._http

    def _headers(self, extra: dict | None = None) -> dict:
        return {"apikey": self._key, "Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                **(extra or {})}

    def _http(self, method: str, url: str, headers: dict, body: bytes | None):
        req = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8")
                return resp.status, (json.loads(text) if text else None)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise SupabaseError(env.redact(f"Supabase {method} failed with HTTP {exc.code}: {detail}")) from None
        except urllib.error.URLError as exc:
            raise SupabaseError(f"Supabase unreachable: {exc.reason}") from None

    def upsert(self, table: str, rows: Iterable[dict]) -> int:
        rows = validate(table, rows)
        if not rows:
            return 0
        query = urllib.parse.urlencode({"on_conflict": CONFLICT_KEYS[table]})
        headers = self._headers({"Prefer": "resolution=merge-duplicates,return=minimal"})
        for i in range(0, len(rows), BATCH):
            self._transport("POST", f"{self.base}/{table}?{query}", headers, json.dumps(rows[i:i + BATCH]).encode())
        return len(rows)

    def select(self, table: str, params: dict | None = None) -> list[dict]:
        query = urllib.parse.urlencode({"select": "*", **(params or {})})
        _, body = self._transport("GET", f"{self.base}/{table}?{query}", self._headers(), None)
        return body or []


def get_store() -> SupabaseStore | None:
    """The configured store, or None in local mode."""
    url, key = env.get("SUPABASE_URL"), env.get("SUPABASE_KEY")
    if not url or not key:
        return None
    if not url.startswith("https://"):
        raise SupabaseError("SUPABASE_URL must start with https://")
    return SupabaseStore(url, key)


# --- row builders ----------------------------------------------------------------------------
def stock_rows(stock_map: pd.DataFrame) -> list[dict]:
    cols = ["symbol", "india_sector", "adr", "us_etf", "us_sector", "fx_exposure"]
    sm = stock_map.reindex(columns=cols).replace("", None)
    sm["symbol"] = sm["symbol"].str.upper().str.strip()
    return sm.drop_duplicates("symbol").to_dict("records")


def prediction_rows(preds: pd.DataFrame, model_name: str) -> list[dict]:
    rows = []
    for r in preds.to_dict("records"):
        gap = r["pred_gap_pct"]
        rows.append({"symbol": r["symbol"], "as_of_close": r["as_of_close"], "target_date": r.get("target_date"),
                     "model_name": model_name, "close": r["close"], "pred_gap_pct": gap,
                     "q10_gap_pct": r.get("q10_gap_pct"), "q90_gap_pct": r.get("q90_gap_pct"),
                     "pred_open": r["pred_open"], "pred_open_low": r.get("pred_open_low"),
                     "pred_open_high": r.get("pred_open_high"),
                     "direction": "up" if gap > 0 else "down" if gap < 0 else "flat"})
    return rows


def evaluation_rows(kind: str, overall: pd.DataFrame, by_stock: pd.DataFrame, details: dict) -> tuple[dict, list[dict]]:
    run_id = stable_id(kind, details)
    run = {"run_id": run_id, "kind": kind, "test_start": details.get("test_start"),
           "test_end": details.get("test_end"), "n_test_rows": details.get("test_rows_per_model"), "details": details}
    metric_cols = ["n", "mae_pct", "rmse_pct", "bias_pct", "direction_acc_pct"]
    rows = [{"run_id": run_id, "model": r["model"], "symbol": "ALL", **{c: r[c] for c in metric_cols}}
            for r in overall.to_dict("records")]
    rows += [{"run_id": run_id, "model": r["model"], "symbol": r["symbol"], **{c: r[c] for c in metric_cols}}
             for r in by_stock.to_dict("records")]
    return run, rows


def weekly_rows(weeks: pd.DataFrame) -> list[dict]:
    cols = ["symbol", "week_start", "entry_date", "exit_date", "sessions", "entry_open", "exit_close",
            "weekly_return", "valid", "exclude_reason"]
    return weeks[cols].to_dict("records")


def backtest_rows(strategy: str, signal_model: str, cfg_hash: str, summary: dict, trades: pd.DataFrame,
                  weeks: pd.DataFrame) -> tuple[dict, list[dict]]:
    first = weeks["week_start"].min() if len(weeks) else None
    last = weeks["week_start"].max() if len(weeks) else None
    run_id = stable_id("backtest", strategy, signal_model, cfg_hash, first, last)
    keys = ["weeks", "trades", "winning_trades", "losing_trades", "win_rate_pct", "gross_pnl", "total_costs",
            "net_pnl", "ending_capital", "cumulative_return_pct", "max_drawdown_pct", "sharpe_annualised"]
    run = {"run_id": run_id, "config_hash": cfg_hash, "strategy": strategy, "signal_model": signal_model,
           "first_week": first, "last_week": last, **{k: summary.get(k) for k in keys}}
    tcols = ["symbol", "week_start", "entry_date", "exit_date", "pred_pct", "qty", "entry_price", "exit_price",
             "gross_pnl", "explicit_cost", "spread_cost", "slippage_cost", "total_cost", "net_pnl"]
    trade_rows = [{"run_id": run_id, **r} for r in trades[tcols].to_dict("records")] if len(trades) else []
    return run, trade_rows


def montecarlo_row(backtest_run_id: str, mc_summary: dict, seed: int, block: int, horizon: int,
                   initial_capital: float) -> dict:
    pct = {k: v for k, v in mc_summary.items() if "_p" in k}
    return {"run_id": stable_id("mc", backtest_run_id, seed, block, horizon, mc_summary["simulations"]),
            "backtest_run_id": backtest_run_id, "simulations": mc_summary["simulations"], "seed": seed,
            "block_weeks": block, "horizon_weeks": horizon, "initial_capital": initial_capital,
            "prob_loss_pct": mc_summary["prob_loss_pct"], "percentiles": pct}


def save_analysis(store: SupabaseStore, *, stock_map: pd.DataFrame | None = None, weeks: pd.DataFrame | None = None,
                  evaluation: tuple | None = None, config: dict | None = None, backtests: list | None = None,
                  montecarlo: dict | None = None) -> dict[str, int]:
    """Write whatever is given, parents before children. Returns rows written per table.

    backtests: list of (strategy, signal_model, summary, trades, weeks).
    montecarlo: {"strategy": name, "summary": mc_summary, "seed", "block", "horizon", "initial_capital"}.
    """
    counts: dict[str, int] = {}
    if stock_map is not None:
        counts["stocks"] = store.upsert("stocks", stock_rows(stock_map))
    if weeks is not None:
        counts["weekly_performance"] = store.upsert("weekly_performance", weekly_rows(weeks))
    if evaluation is not None:
        run, metric_rows = evaluation_rows(*evaluation)
        counts["evaluation_runs"] = store.upsert("evaluation_runs", [run])
        counts["model_metrics"] = store.upsert("model_metrics", metric_rows)
    run_ids = {}
    if config is not None and backtests:
        h = config_hash(config)
        counts["strategy_configs"] = store.upsert("strategy_configs", [{"config_hash": h, "config": config}])
        for strategy, signal_model, summ, trades, wk in backtests:
            run, trade_rows = backtest_rows(strategy, signal_model, h, summ, trades, wk)
            store.upsert("backtest_runs", [run])
            counts["backtest_trades"] = counts.get("backtest_trades", 0) + store.upsert("backtest_trades", trade_rows)
            run_ids[strategy] = run["run_id"]
        counts["backtest_runs"] = len(backtests)
    if montecarlo is not None and montecarlo["strategy"] in run_ids:
        m = montecarlo
        counts["monte_carlo_runs"] = store.upsert("monte_carlo_runs", [montecarlo_row(
            run_ids[m["strategy"]], m["summary"], m["seed"], m["block"], m["horizon"], m["initial_capital"])])
    return counts
