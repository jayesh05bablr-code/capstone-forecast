"""Make a GitHub merge show up on Streamlit Community Cloud without a manual reboot.

When a change is merged, Streamlit Cloud pulls the new files and re-runs app.py, but Python keeps the
already imported capstone modules in memory. A change that only touches files under capstone/ (like a
new chart) therefore stays invisible until someone reboots the app, and a reboot wipes the downloaded
data. This module notices that the code on disk changed and makes Python import it again.

autorun and refresh are left alone: they hold the lock and status of a background update that may be
running, and re-importing them could start a second update at the same time.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent / "capstone"
KEEP = {"capstone", "capstone.autorun", "capstone.refresh"}
_loaded: str | None = None  # fingerprint of the code that is imported now


def fingerprint(root: Path = PACKAGE) -> str:
    """Changes whenever any .py file under capstone/ is added, removed or edited."""
    h = hashlib.sha1()
    for f in sorted(root.rglob("*.py")):
        st = f.stat()
        h.update(f"{f.relative_to(root)}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def reload_if_changed() -> bool:
    """Forget the imported capstone modules if the code changed; True when it did."""
    global _loaded
    now = fingerprint()
    if now == _loaded:
        return False
    stale = [m for m in sys.modules if m.startswith("capstone.") and m not in KEEP]
    first = _loaded is None
    _loaded = now
    if first and not stale:
        return False  # fresh server: nothing imported yet
    for name in stale:
        del sys.modules[name]
        parent, _, child = name.rpartition(".")
        if parent in KEEP and hasattr(sys.modules.get(parent), child):
            delattr(sys.modules[parent], child)  # else "from capstone import x" would return the old x
    return True
