import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config
from src.ingest import read_from_sqlite


def overview(df, target, positive_label):
    """Size, duplicates and how rare failures are."""
    is_fail = df[target] == positive_label
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")
    print(f"Failures: {is_fail.sum():,} ({is_fail.mean():.4%})")


def missing_report(df):
    """Fraction of missing values per column, only for columns that have any."""
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    print(f"Columns with missing values: {len(missing)} of {df.shape[1]}")
    if len(missing) > 0:
        print(missing.head(10).to_string())
    return missing


def feature_separation(df, target, positive_label):
    """Rank numeric features by how differently they behave in failures vs normal rows."""
    is_fail = df[target] == positive_label
    features = df.drop(columns=[target]).select_dtypes("number")

    result = pd.DataFrame({
        "mean_fail": features[is_fail].mean(),
        "mean_normal": features[~is_fail].mean(),
        "std": features.std(),
    })
    result["effect"] = (result["mean_fail"] - result["mean_normal"]) / result["std"]
    return result.sort_values("effect", key=abs, ascending=False)


def plot_distributions(df, target, positive_label, columns, out_path):
    """Overlay the distribution of each column for failures vs normal rows."""
    is_fail = df[target] == positive_label
    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 3.5), squeeze=False)

    for ax, col in zip(axes[0], columns):
        ax.hist(df.loc[~is_fail, col], bins=50, density=True, alpha=0.5, label="normal")
        ax.hist(df.loc[is_fail, col], bins=50, density=True, alpha=0.5, label="failure")
        ax.set_title(col)

    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Explore a dataset stored in SQLite.")
    parser.add_argument("config", help="Path to a YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    target, positive_label = config["target"], config["positive_label"]

    df = read_from_sqlite(config["db_path"], config["table"])
    out_dir = Path("reports") / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Overview ===")
    overview(df, target, positive_label)

    print("\n=== Missing values ===")
    missing = missing_report(df)
    missing.to_csv(out_dir / "missing.csv", header=["missing_fraction"])

    print("\n=== Features that best separate failures from normal ===")
    separation = feature_separation(df, target, positive_label)
    print(separation.head(10).round(3).to_string())
    separation.to_csv(out_dir / "feature_separation.csv")

    top_features = separation.index[:4].tolist()
    plot_distributions(df, target, positive_label, top_features, out_dir / "top_features.png")
    print(f"\nSaved reports to {out_dir}/")


def run(config):
    target, positive_label = config["target"], config["positive_label"]

    df = read_from_sqlite(config["db_path"], config["table"])
    out_dir = Path("reports") / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Overview ===")
    overview(df, target, positive_label)

    print("\n=== Missing values ===")
    missing = missing_report(df)
    missing.to_csv(out_dir / "missing.csv", header=["missing_fraction"])

    print("\n=== Features that best separate failures from normal ===")
    separation = feature_separation(df, target, positive_label)
    print(separation.head(10).round(3).to_string())
    separation.to_csv(out_dir / "feature_separation.csv")

    top_features = separation.index[:4].tolist()
    plot_distributions(df, target, positive_label, top_features, out_dir / "top_features.png")
    print(f"\nSaved reports to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Explore a dataset stored in SQLite.")
    parser.add_argument("config", help="Path to a YAML config file")
    args = parser.parse_args()
    run(load_config(args.config))