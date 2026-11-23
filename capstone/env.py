"""Load API keys and other settings from the environment, never from source code.

Order of precedence (highest first):
1. A local `.env` file in the project folder (read with python-dotenv). Values in `.env` win over
   older environment variables, so a stale key set earlier with `setx` cannot hide a new one.
2. Ordinary environment variables (e.g. set by a deployment platform).
3. Streamlit Community Cloud's Secrets box, copied in by `load_streamlit_secrets()` only for
   names that are not already set.

The `.env` file stays on your computer: it is git-ignored and never printed. Use `redact()` on any
text that is shown to a user (logs, error messages) so a key can never leak through it.
"""
from __future__ import annotations

import os

from .paths import ROOT

ENV_FILE = ROOT / ".env"

# Every setting that holds a credential. redact() hides the values of these.
SECRET_NAMES = ("TWELVEDATA_API_KEY", "EXCHANGERATE_API_KEY", "TEJ_API_KEY", "ALPHAVANTAGE_API_KEY",
                "SUPABASE_KEY", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_ANON_KEY", "GROQ_API_KEY",
                "GEMINI_API_KEY")


def _load_without_dotenv(path) -> None:
    """Fallback parser used only if python-dotenv is not installed (same NAME=value format)."""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if name and value:
            os.environ[name] = value


def load(path=ENV_FILE) -> None:
    """Read `.env` into os.environ. Empty values (e.g. `TEJ_API_KEY=`) are ignored."""
    if not path.exists():
        return
    try:
        from dotenv import dotenv_values
    except ImportError:
        _load_without_dotenv(path)
        return
    for name, value in dotenv_values(path, encoding="utf-8-sig").items():
        if name and value is not None and value.strip():
            os.environ[name] = value.strip()


def get(name: str, default: str | None = None) -> str | None:
    """os.getenv with blank values treated as missing."""
    value = os.getenv(name, "")
    return value.strip() or default


def redact(text: str) -> str:
    """Replace every configured secret value in `text` with ***."""
    for name in SECRET_NAMES:
        value = os.environ.get(name, "")
        if len(value) >= 6:
            text = text.replace(value, "***")
    return text


def load_streamlit_secrets() -> int:
    """On Streamlit Community Cloud, keys live in the app's Secrets box, not in a .env file.

    Copies each top-level text secret (e.g. TWELVEDATA_API_KEY = "...") into the environment so
    the API clients, and the train/predict processes started from the dashboard, can read it.
    Returns how many were copied. Does nothing when there are no secrets (e.g. on your laptop).
    """
    try:
        import streamlit as st

        items = dict(st.secrets)
    except Exception:  # no secrets.toml, or not running under Streamlit
        return 0
    n = 0
    for name, value in items.items():
        if isinstance(value, (str, int, float)) and str(value).strip() and not os.environ.get(name):
            os.environ[name] = str(value).strip()
            n += 1
    return n
