"""Retrain and re-predict from the dashboard, so uploads need no terminal commands.

Runs `python -m capstone.train` then `python -m capstone.predict` in a fresh
process (same Python as the dashboard). Their output is echoed to the terminal
running Streamlit and also returned, so the dashboard can show it.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading

from .env import redact
from .paths import DATA, FUNDAMENTALS_FILE, MODELS, PROCESSED, ROOT

MODEL_FILE = MODELS / "gap_model.joblib"
PREDICTIONS_FILE = PROCESSED / "predictions.csv"
# Inputs the model reads besides prices: when one changes, the model is out of date.
MODEL_INPUTS = [FUNDAMENTALS_FILE, DATA / "stock_map.csv", DATA / "sector_links.csv"]


INPUTS_FILE = MODELS / "model_inputs.json"
_lock = threading.Lock()


def input_hashes() -> dict[str, str]:
    """Content fingerprint of each model input file (missing files are left out)."""
    from .features import VERSION  # local import keeps this module light for the dashboard

    out = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in MODEL_INPUTS if p.exists()}
    out["features_version"] = VERSION
    return out


def save_input_hashes() -> None:
    """Called after training: remembers which inputs this model was built from."""
    INPUTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    INPUTS_FILE.write_text(json.dumps(input_hashes(), indent=2))


def stale() -> bool:
    """True when an input the model uses has changed since the model was trained.

    Compares file contents, not save times: redeploying the app (e.g. on Streamlit Cloud after a
    GitHub merge) rewrites every file with a new time but the same content, and must not retrain.
    """
    if not MODEL_FILE.exists():
        return False  # never trained here: the first run is done with run_all
    built = MODEL_FILE.stat().st_mtime
    if not PREDICTIONS_FILE.exists() or PREDICTIONS_FILE.stat().st_mtime < built:
        return True  # model retrained but predict did not finish
    try:
        return json.loads(INPUTS_FILE.read_text()) != input_hashes()
    except (OSError, ValueError):
        # A model trained before fingerprints existed: take today's inputs as its inputs, once.
        save_input_hashes()
        return False


def _echo(text: str) -> None:
    try:
        print(text, end="", flush=True)
    except UnicodeEncodeError:  # a Windows console without UTF-8 cannot print ₹
        print(text.encode("ascii", "replace").decode(), end="", flush=True)


def _run(module: str) -> tuple[bool, str]:
    _echo(f"\n[auto-update] python -m {module}\n")
    # On Windows a piped child writes in the ANSI code page (cp1252), which has no ₹ and crashes
    # predict's printout; force UTF-8 for the child.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.Popen([sys.executable, "-m", module], cwd=ROOT, stdout=subprocess.PIPE, env=env,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    lines = []
    for line in proc.stdout:
        line = redact(line)  # a key must never reach the terminal or the dashboard log
        _echo(line)
        lines.append(line)
    return proc.wait() == 0, "".join(lines)


def retrain_and_predict() -> tuple[bool, str]:
    """Train then predict. Returns (ok, combined output).

    Only one retrain runs at a time: when several visitors open the page together, the others
    wait for the one in progress and use its result instead of starting their own.
    """
    if not _lock.acquire(blocking=False):
        with _lock:
            pass
        return not stale(), "Another visitor's update was already running; using its results.\n"
    try:
        return _retrain_and_predict()
    finally:
        _lock.release()


def _retrain_and_predict() -> tuple[bool, str]:
    ok, log = _run("capstone.train")
    if ok:
        ok, log2 = _run("capstone.predict")
        log += log2
    _echo(f"[auto-update] {'done' if ok else 'FAILED'}\n")
    return ok, log
