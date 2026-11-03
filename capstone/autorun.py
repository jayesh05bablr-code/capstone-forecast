"""Keep the forecast up to date by itself, so nobody has to run commands in a terminal.

When the dashboard opens (on your laptop or on Streamlit Community Cloud) it calls
`ensure_fresh()`. If the predictions are older than the latest update time, it runs the same
steps as run_all.bat in the background:

    fetch_data -> fetch_overnight -> train (only if the model is over a week old) -> predict

Update times (IST, every day): 08:45, after the US close and before NSE opens, so the overnight
US and rupee moves are in; and 16:15, after NSE closes, so today's close is in.

The first start on a new server has no data at all, so that run happens while the page waits
(a few minutes: about 3 for Twelve Data's free-tier rate limit, plus training).
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta

import pandas as pd

from . import refresh
from .paths import PROCESSED, RAW

SLOTS = [(8, 45), (16, 15)]
RETRAIN_AFTER = timedelta(days=7)
RETRY_AFTER_FAIL = 30 * 60  # seconds
STATUS_FILE = PROCESSED / "auto_status.json"

_lock = threading.Lock()
_state = {"running": False, "step": "", "last_fail": 0.0}


def last_slot(now: datetime) -> datetime:
    """The latest update time at or before `now` (naive IST)."""
    times = [now.replace(hour=h, minute=m, second=0, microsecond=0) for h, m in SLOTS]
    past = [t for t in times if t <= now]
    if past:
        return max(past)
    h, m = SLOTS[-1]
    return (now - timedelta(days=1)).replace(hour=h, minute=m, second=0, microsecond=0)


def _now_ist() -> datetime:
    return pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).to_pydatetime()


def _mtime(path) -> datetime | None:
    if not path.exists():
        return None
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").tz_convert("Asia/Kolkata").tz_localize(None).to_pydatetime()


def is_stale(now: datetime | None = None) -> bool:
    built = _mtime(refresh.PREDICTIONS_FILE)
    return built is None or built < last_slot(now or _now_ist())


def needs_training(now: datetime | None = None) -> bool:
    built = _mtime(refresh.MODEL_FILE)
    return built is None or (now or _now_ist()) - built > RETRAIN_AFTER


def status() -> dict:
    """What the background updater is doing, and how the last run went."""
    saved = json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else {}
    return {**saved, "running": _state["running"], "step": _state["step"]}


def run_pipeline() -> tuple[bool, str]:
    """fetch_data, fetch_overnight, (train), predict. Returns (ok, log)."""
    log = ""
    steps = [("capstone.fetch_data", False), ("capstone.fetch_overnight", False)]
    if needs_training() or not any(RAW.glob("*.csv")):
        steps.append(("capstone.train", True))
    steps.append(("capstone.predict", True))
    ok = True
    for module, required in steps:
        _state["step"] = module.split(".")[-1]
        step_ok, out = refresh._run(module)
        log += f"\n$ python -m {module}\n{out}"
        if not step_ok and module == "capstone.predict" and refresh.MODEL_FILE.exists():
            # Predict can fail when the saved model expects features the new data lacks: retrain once.
            step_ok, out = refresh._run("capstone.train")
            log += f"\n$ python -m capstone.train (retry)\n{out}"
            if step_ok:
                step_ok, out = refresh._run(module)
                log += f"\n$ python -m {module}\n{out}"
        if required and not step_ok:
            ok = False
            break
    PROCESSED.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps({"ok": ok, "finished": _now_ist().strftime("%Y-%m-%d %H:%M"),
                                       "log": log[-8000:]}, indent=2))
    return ok, log


def _guarded() -> tuple[bool, str] | None:
    if not _lock.acquire(blocking=False):
        return None  # another visitor already started it
    _state["running"] = True
    try:
        ok, log = run_pipeline()
        if not ok:
            _state["last_fail"] = time.time()
        return ok, log
    finally:
        _state["running"], _state["step"] = False, ""
        _lock.release()


def ensure_fresh(block: bool = False) -> str:
    """Start an update if one is due. Returns 'fresh', 'started', 'running', 'waiting' or 'done'.

    block=True runs it in this call (used on the very first start, when there is nothing to show).
    """
    if _state["running"]:
        return "running"
    if not is_stale():
        return "fresh"
    if time.time() - _state["last_fail"] < RETRY_AFTER_FAIL:
        return "waiting"
    if block:
        _guarded()
        return "done"
    threading.Thread(target=_guarded, daemon=True, name="forecast-updater").start()
    return "started"


def update_now() -> tuple[bool, str] | None:
    """Run an update right away (the sidebar button). None if one is already running."""
    return _guarded()
