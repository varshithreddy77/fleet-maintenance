"""Shared preprocessing and cost logic, imported by train.py and app.py."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

COST_FP, COST_FN = 10, 500  # inspect a healthy truck vs. miss a truck that fails


class OutlierClipLog(BaseEstimator, TransformerMixin):
    """Winsorise each sensor at its training 0.1/99.9 percentiles, then log1p.

    Sensor readings are heavy-tailed counters; clipping + log stops a handful of
    extreme trucks from dominating linear models and split points.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, 0.1, axis=0)
        self.hi_ = np.nanpercentile(X, 99.9, axis=0)
        return self

    def transform(self, X):
        X = np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)
        return np.log1p(np.clip(X, 0, None))


def load_aps(path):
    """Read a Scania APS csv (20-line licence header, 'na' = missing)."""
    df = pd.read_csv(path, skiprows=20, na_values="na")
    y = (df.pop("class") == "pos").astype(int).to_numpy() if "class" in df else None
    return df.astype(float), y


def add_missing_count(X, sensor_cols):
    """Number of missing sensors per truck; missingness itself is a failure signal."""
    X = X.copy()
    X["n_missing"] = X[sensor_cols].isna().sum(axis=1)
    return X
