"""Candidate forecasters for the opening gap, and the walk-forward tournament that picks one.

Why these candidates (research behind them, in short):
- Overnight and opening returns are close to unpredictable from past prices alone (efficient-market
  evidence; Lou, Polk & Skouras 2019 on overnight returns). Most of the forecastable part of an
  Indian opening gap comes from what happened abroad while NSE was shut: US index and sector moves,
  Indian ADRs and the rupee (overnight.py). A model should therefore be judged against "open = last
  close", and only kept when it beats it out of sample.
- With a low signal-to-noise target, simple, regularised and robust models often beat flexible ones
  (Gu, Kelly & Xiu 2020 find gains from trees, but small and fragile): ridge and Huber regression.
- Forecast combination: averaging different models is reliably at least as good as the average
  model (Rapach, Strauss & Zhou 2010; Timmermann 2006).
- Shrinkage toward zero: return forecasts are too large in absolute size; scaling them down,
  with the scale chosen on data the model did not train on, cuts the error (Campbell & Thompson
  2008). The scale is floored at 0.1 so the forecast keeps its up/down call.

Every candidate is fitted only on data before each test period (train.folds) and scored on the same
rows. The one with the lowest average error is saved; the others are kept in the report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SHRINK_GRID = np.round(np.linspace(0.1, 1.0, 19), 2)
INNER_SHARE = 0.8  # shrink factor chosen on the last 20% of the training window's dates


def hgb(**kw) -> HistGradientBoostingRegressor:
    p = dict(loss="absolute_error", max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=50,
             l2_regularization=1.0, random_state=0)
    p.update(kw)
    return HistGradientBoostingRegressor(**p)


def _linear(model):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), model)


def ridge():
    return _linear(RidgeCV(alphas=np.logspace(-2, 4, 13)))


def huber():
    return _linear(HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=500))


class Average:
    """Equal-weight combination of several fitted models."""

    def __init__(self, makers):
        self.makers = makers

    def fit(self, X, y):
        self.models_ = [m().fit(X, y) for m in self.makers]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models_], axis=0)


class Shrunk:
    """A model whose forecasts are scaled toward zero by a factor chosen out of sample.

    fit(X, y, dates): fit on the first 80% of the window's dates, pick the factor that minimises the
    absolute error on the last 20%, then refit on the whole window. Needs dates, so call fit_candidate.
    """

    def __init__(self, maker):
        self.maker = maker

    def fit(self, X, y, dates=None):
        self.scale_ = 1.0
        if dates is not None:
            d = pd.Series(pd.to_datetime(np.asarray(dates)))
            u = np.sort(d.unique())
            if len(u) >= 50:
                cut = u[int(len(u) * INNER_SHARE)]
                a, b = (d < cut).to_numpy(), (d >= cut).to_numpy()
                ok = [c for c in X.columns if X.loc[a, c].nunique(dropna=True) >= 2]
                p = self.maker().fit(X.loc[a, ok], y[a]).predict(X.loc[b, ok])
                yb = np.asarray(y)[b]
                self.scale_ = float(SHRINK_GRID[np.argmin([np.abs(yb - s * p).mean() for s in SHRINK_GRID])])
        self.model_ = self.maker().fit(X, y)
        return self

    def predict(self, X):
        return self.scale_ * self.model_.predict(X)


def _ens():
    return Average([hgb, huber])


# name -> (plain label, maker)
CANDIDATES = {
    "hgb": ("Gradient boosting (previous model)", hgb),
    "hgb_regularised": ("Gradient boosting, shallower and more regularised",
                        lambda: hgb(max_depth=2, learning_rate=0.03, max_iter=200, min_samples_leaf=200,
                                    l2_regularization=5.0)),
    "ridge": ("Ridge regression", ridge),
    "huber": ("Huber robust regression", huber),
    "ensemble": ("Average of gradient boosting and Huber", _ens),
    "hgb_shrunk": ("Gradient boosting, shrunk toward zero", lambda: Shrunk(hgb)),
    "huber_shrunk": ("Huber regression, shrunk toward zero", lambda: Shrunk(huber)),
    "ensemble_shrunk": ("Average of boosting and Huber, shrunk toward zero", lambda: Shrunk(_ens)),
}
DEFAULT = "hgb"


def label(name: str) -> str:
    return CANDIDATES.get(name, (name,))[0]


def fit_candidate(name: str, X: pd.DataFrame, y, dates=None):
    model = CANDIDATES[name][1]()
    return model.fit(X, y, dates=dates) if isinstance(model, Shrunk) else model.fit(X, y)


def pick(summary: pd.DataFrame, metric: str = "mape_pct") -> str:
    """Lowest average error among the candidates in a walk-forward summary (rows 'candidate:<name>')."""
    rows = summary[[i.startswith("candidate:") for i in summary.index]]
    if rows.empty:
        return DEFAULT
    return rows[metric].idxmin().split(":", 1)[1]
