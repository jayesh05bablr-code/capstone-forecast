#!/usr/bin/env bash
# One-command run (Mac/Linux). The dashboard fetches data, trains and predicts by itself.
set -e
cd "$(dirname "$0")"
[ -f .env ] || { cp .env.example .env; echo "Created .env: paste your keys into it, then run again."; exit 1; }
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
python -m capstone.check_keys
streamlit run app.py
