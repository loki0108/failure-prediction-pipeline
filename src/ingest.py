import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import load_config


def load_csv(raw_path, target):
    """Read the raw CSV and check that the target column exists."""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. See README for download steps.")

    df = pd.read_csv(raw_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {raw_path}")
    return df


def write_to_sqlite(df, db_path, table):
    """Write a DataFrame into a SQLite table, replacing it if it already exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table, conn, if_exists="replace", index=False, chunksize=10_000)
    finally:
        conn.close()


def read_from_sqlite(db_path, table):
    """Read a whole table back as a DataFrame."""
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    finally:
        conn.close()


def summarize(df, target, positive_label):
    """Print the two things that matter first: size/missing data and class balance."""
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(f"Missing values (total): {df.isna().sum().sum()}")
    print("Class counts:")
    print(df[target].value_counts().to_string())
    failure_rate = (df[target] == positive_label).mean()
    print(f"Failure rate: {failure_rate:.4%}")


def run(config):
    df = load_csv(config["raw_path"], config["target"])
    write_to_sqlite(df, config["db_path"], config["table"])

    df_db = read_from_sqlite(config["db_path"], config["table"])
    assert df_db.shape == df.shape, "Row/column count changed during the round trip"

    print(f"Loaded '{config['name']}' into {config['db_path']} (table: {config['table']})")
    summarize(df_db, config["target"], config["positive_label"])


def main():
    parser = argparse.ArgumentParser(description="Load a raw CSV into SQLite.")
    parser.add_argument("config", help="Path to a YAML config file")
    args = parser.parse_args()
    run(load_config(args.config))

if __name__ == "__main__":
    main()