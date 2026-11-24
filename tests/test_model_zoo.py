"""Model tournament (no network)."""
import numpy as np
import pandas as pd

from capstone import model_zoo


def _data(n_days=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    X = pd.DataFrame({"a": rng.normal(size=n_days), "b": rng.normal(size=n_days)})
    y = 0.002 * X["a"] + rng.normal(scale=0.01, size=n_days)  # weak signal in heavy noise
    return X, y, pd.Series(dates)


def test_every_candidate_fits_and_predicts():
    X, y, d = _data()
    for name in model_zoo.CANDIDATES:
        p = model_zoo.fit_candidate(name, X, y, d).predict(X.head(5))
        assert p.shape == (5,) and np.isfinite(p).all(), name


def test_shrink_factor_is_chosen_out_of_sample_and_floored():
    X, y, d = _data()
    m = model_zoo.fit_candidate("huber_shrunk", X, y, d)
    assert model_zoo.SHRINK_GRID.min() <= m.scale_ <= 1.0
    assert m.scale_ >= 0.1  # floor keeps the up/down call
    np.testing.assert_allclose(m.predict(X.head(3)), m.scale_ * m.model_.predict(X.head(3)))


def test_shrunk_without_dates_is_unscaled():
    X, y, _ = _data()
    assert model_zoo.Shrunk(model_zoo.huber).fit(X, y).scale_ == 1.0


def test_pick_lowest_error_candidate():
    s = pd.DataFrame({"mape_pct": [0.50, 0.52, 0.49, 0.40]},
                     index=["candidate:hgb", "candidate:ridge", "candidate:huber_shrunk", "baseline_open_eq_close"])
    assert model_zoo.pick(s) == "huber_shrunk"  # baselines are not candidates
    assert model_zoo.pick(s.drop(index=[i for i in s.index if i.startswith("candidate:")])) == model_zoo.DEFAULT
