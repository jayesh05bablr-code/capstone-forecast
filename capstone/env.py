"""Load API keys from a local .env file in the project folder.

Each line looks like NAME=value. Values in .env win over old environment
variables, so a stale or placeholder key set earlier with setx can't hide them.
The .env file stays on your computer: it is git-ignored and never printed.
"""
from __future__ import annotations

import os

from .paths import ROOT

ENV_FILE = ROOT / ".env"


def load(path=ENV_FILE) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if name and value:
            os.environ[name] = value


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
