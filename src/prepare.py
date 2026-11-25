from sklearn.model_selection import train_test_split

from src.ingest import read_from_sqlite
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
import pandas as pd

def load_dataset(config):
    """Read the table from SQLite and drop exact duplicate rows."""
    df = read_from_sqlite(config["db_path"], config["table"])
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {before - len(df):,} duplicate rows ({len(df):,} remain)")
    return df


def split_features_target(df, target, positive_label, drop_columns=None):
    """Separate inputs (X) from the label (y), with y as 1 = failure, 0 = normal.
    drop_columns: non-feature columns (e.g. timestamps) to keep out of X."""
    X = df.drop(columns=[target] + list(drop_columns or []))
    y = (df[target] == positive_label).astype(int)
    return X, y


def train_test(X, y, test_size=0.2, random_state=42, order=None):
    """Stratified random split, or, if `order` (timestamps) is given,
    a time-based split: train on the earliest rows, test on the latest."""
    if order is None:
        return train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
    sorted_idx = pd.to_datetime(order).sort_values(kind="stable").index
    n_test = int(round(len(sorted_idx) * test_size))
    train_idx, test_idx = sorted_idx[:-n_test], sorted_idx[-n_test:]
    return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx]


def prepare_splits(config):
    """Load, separate X/y and split, all according to the config."""
    df = load_dataset(config)
    X, y = split_features_target(
        df, config["target"], config["positive_label"], config.get("drop_columns")
    )
    order = df[config["time_column"]] if config.get("time_column") else None
    X_train, X_test, y_train, y_test = train_test(
        X, y, config.get("test_size", 0.2), config.get("random_state", 42), order
    )
    kind = "time-based" if order is not None else "stratified random"
    print(f"Split ({kind}): train {len(X_train):,} rows ({y_train.sum()} failures), "
          f"test {len(X_test):,} rows ({y_test.sum()} failures)")
    return X_train, X_test, y_train, y_test

def make_cv(config):
    """Time-ordered data -> each fold trains on the past and validates on the future.
    Otherwise -> shuffled stratified folds."""
    n_splits = config.get("cv_folds", 5)
    if config.get("time_column"):
        return TimeSeriesSplit(n_splits=n_splits)
    return StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=config.get("random_state", 42)
    )