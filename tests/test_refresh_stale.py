"""A redeploy rewrites files with new times but the same content: that must not retrain."""
import os
import threading
import time

from capstone import refresh


def _setup(tmp_path, monkeypatch):
    inp = tmp_path / "stock_map.csv"
    inp.write_text("a,b\n")
    model, preds = tmp_path / "gap_model.joblib", tmp_path / "predictions.csv"
    model.write_text("m")
    preds.write_text("p")
    monkeypatch.setattr(refresh, "MODEL_INPUTS", [inp])
    monkeypatch.setattr(refresh, "MODEL_FILE", model)
    monkeypatch.setattr(refresh, "PREDICTIONS_FILE", preds)
    monkeypatch.setattr(refresh, "INPUTS_FILE", tmp_path / "model_inputs.json")
    return inp, model


def test_same_content_newer_time_is_not_stale(tmp_path, monkeypatch):
    inp, model = _setup(tmp_path, monkeypatch)
    refresh.save_input_hashes()
    later = model.stat().st_mtime + 100
    os.utime(inp, (later, later))  # what a fresh checkout does
    assert not refresh.stale()
    inp.write_text("a,b\nc,d\n")  # a real change
    assert refresh.stale()


def test_only_one_retrain_at_a_time(monkeypatch):
    runs = []

    def slow():
        runs.append(1)
        time.sleep(0.3)
        return True, "trained"
    monkeypatch.setattr(refresh, "_retrain_and_predict", slow)
    monkeypatch.setattr(refresh, "stale", lambda: False)
    out = []
    threads = [threading.Thread(target=lambda: out.append(refresh.retrain_and_predict())) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(runs) == 1 and all(ok for ok, _ in out)


def test_older_model_without_fingerprints_is_not_retrained(tmp_path, monkeypatch):
    inp, model = _setup(tmp_path, monkeypatch)
    later = model.stat().st_mtime + 100
    os.utime(inp, (later, later))
    assert not refresh.stale() and refresh.INPUTS_FILE.exists()
