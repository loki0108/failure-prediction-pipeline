from sklearn.model_selection import train_test_split

from src.ingest import read_from_sqlite


def load_dataset(config):
    """Read the table from SQLite and drop exact duplicate rows."""
    df = read_from_sqlite(config["db_path"], config["table"])
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {before - len(df):,} duplicate rows ({len(df):,} remain)")
    return df


def split_features_target(df, target, positive_label):
    """Separate inputs (X) from the label (y), with y as 1 = failure, 0 = normal."""
    X = df.drop(columns=[target])
    y = (df[target] == positive_label).astype(int)
    return X, y


def train_test(X, y, test_size=0.2, random_state=42):
    """Stratified split so train and test keep the same failure rate."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )