"""One-off converter: SECOM ships as two headerless, space-separated files.
Combines them into data/raw/secom.csv with named columns."""
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")


def main():
    sensors = pd.read_csv(RAW / "secom.data", sep=r"\s+", header=None)
    sensors.columns = [f"sensor_{i}" for i in range(sensors.shape[1])]

    labels = pd.read_csv(
        RAW / "secom_labels.data", sep=r"\s+", header=None, names=["label", "timestamp"]
    )
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S")

    if len(sensors) != len(labels):
        raise ValueError(f"Row mismatch: {len(sensors)} sensor rows vs {len(labels)} labels")

    df = pd.concat([sensors, labels], axis=1)
    df.to_csv(RAW / "secom.csv", index=False)
    print(f"Wrote {RAW / 'secom.csv'}: {df.shape[0]} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main()