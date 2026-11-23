"""codeversion: a merged change under capstone/ is imported again without rebooting the app."""
import os
import sys

import codeversion


def test_reload_after_code_change(tmp_path, monkeypatch):
    import capstone.paths
    import capstone.autorun
    old_paths, old_autorun = sys.modules["capstone.paths"], sys.modules["capstone.autorun"]
    monkeypatch.setattr(codeversion, "_loaded", None)
    codeversion.reload_if_changed()            # first call after a deploy: drops what was imported before
    assert "capstone.paths" not in sys.modules
    from capstone import paths
    assert paths is not old_paths               # fresh copy of the code
    assert sys.modules["capstone.autorun"] is old_autorun  # update lock and status kept
    assert codeversion.reload_if_changed() is False        # unchanged code: nothing happens
    f = codeversion.PACKAGE / "paths.py"
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    try:
        assert codeversion.reload_if_changed() is True
        from capstone import paths as again
        assert again is not paths
    finally:
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
        codeversion.reload_if_changed()
