import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SensorCleaner(BaseEstimator, TransformerMixin):
    """Drops mostly-empty, constant and near-duplicate columns.
    Which columns to keep is learned in fit() (training data only) and applied in transform()."""

    def __init__(self, max_missing=0.5, corr_threshold=0.95):
        self.max_missing = max_missing
        self.corr_threshold = corr_threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        # 1. drop columns that are mostly empty
        keep = X.columns[X.isna().mean() <= self.max_missing]
        # 2. drop columns that never change
        keep = [c for c in keep if X[c].nunique(dropna=True) > 1]
        # 3. drop one column from every highly correlated pair
        corr = X[keep].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        redundant = {c for c in upper.columns if (upper[c] > self.corr_threshold).any()}

        self.columns_ = [c for c in keep if c not in redundant]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.columns_]