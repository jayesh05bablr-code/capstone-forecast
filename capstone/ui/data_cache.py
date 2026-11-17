"""Cached loaders and computations for the analysis tabs (recomputed only when inputs change)."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from capstone import backtest, model_eval, montecarlo, weekly, weekly_model
from capstone.paths import MODELS, RAW


def _mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _oos(mtime: float) -> pd.DataFrame | None:
    return model_eval.load_oos()


def oos() -> pd.DataFrame | None:
    return _oos(_mtime(model_eval.OOS_FILE))


@st.cache_data(show_spinner=False)
def _csv(name: str, mtime: float) -> pd.DataFrame | None:
    path = MODELS / name
    return pd.read_csv(path) if path.exists() else None


def models_csv(name: str) -> pd.DataFrame | None:
    return _csv(name, _mtime(MODELS / name))


def meta() -> dict:
    return json.loads(model_eval.META_FILE.read_text()) if model_eval.META_FILE.exists() else {}


@st.cache_data(show_spinner=False)
def _weeks(stamp: tuple) -> pd.DataFrame:
    return weekly.all_weeks(weekly.load_prices())


def weeks() -> pd.DataFrame:
    return _weeks(tuple(sorted((p.name, p.stat().st_mtime) for p in RAW.glob("*.csv"))))


@st.cache_data(show_spinner=False)
def _weekly_oos(mtime: float) -> pd.DataFrame | None:
    return weekly_model.load_oos()


def weekly_oos() -> pd.DataFrame | None:
    return _weekly_oos(_mtime(weekly_model.WEEKLY_OOS_FILE))


def _cfg_key(cfg: backtest.BacktestConfig) -> str:
    return json.dumps({"c": cfg.initial_capital, "a": cfg.allocation, "m": cfg.cost_multiple,
                       "s": sorted(cfg.symbols or []), "costs": cfg.assumptions.to_dict()}, sort_keys=True)


@st.cache_data(show_spinner="Running the backtests...")
def _backtests(model: str, key: str, mtime: float, _cfg: backtest.BacktestConfig) -> dict:
    sig = weekly_model.load_oos()
    return backtest.compare(sig[sig["model"] == model], _cfg)


def backtests(model: str, cfg: backtest.BacktestConfig) -> dict | None:
    if weekly_oos() is None:
        return None
    return _backtests(model, _cfg_key(cfg), _mtime(weekly_model.WEEKLY_OOS_FILE), cfg)


@st.cache_data(show_spinner="Running 5,000 simulations...")
def simulate(returns: tuple, capital: float, n_sims: int, horizon: int, block: int, seed: int) -> pd.DataFrame:
    return montecarlo.simulate(list(returns), capital, n_sims=n_sims, horizon=horizon, block=block, seed=seed)
