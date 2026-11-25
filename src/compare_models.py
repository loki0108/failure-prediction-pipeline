import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import load_config
from src.models import build_candidates
from src.prepare import load_dataset, split_features_target, train_test

SCORING = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "pr_auc": "average_precision",
}


def main():
    parser = argparse.ArgumentParser(description="Compare models with cross-validation.")
    parser.add_argument("config", help="Path to a YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    random_state = config.get("random_state", 42)

    df = load_dataset(config)
    X, y = split_features_target(df, config["target"], config["positive_label"])
    # The test set is created but NOT used here; it stays locked for the final evaluation.
    X_train, X_test, y_train, y_test = train_test(
        X, y, config.get("test_size", 0.2), random_state
    )

    cv = StratifiedKFold(
        n_splits=config.get("cv_folds", 5), shuffle=True, random_state=random_state
    )

    rows = []
    for name, model in build_candidates(random_state).items():
        start = time.time()
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING)

        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        row["seconds"] = time.time() - start
        rows.append(row)

        print(f"{name:18s} recall={row['recall_mean']:.3f}±{row['recall_std']:.3f}  "
              f"precision={row['precision_mean']:.3f}±{row['precision_std']:.3f}  "
              f"f1={row['f1_mean']:.3f}  pr_auc={row['pr_auc_mean']:.3f}  "
              f"({row['seconds']:.0f}s)")

    out_dir = Path("reports") / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows).set_index("model").round(4)
    results.to_csv(out_dir / "cv_comparison.csv")
    print(f"\nSaved to {out_dir / 'cv_comparison.csv'}")


def run(config):
    random_state = config.get("random_state", 42)

    df = load_dataset(config)
    X, y = split_features_target(df, config["target"], config["positive_label"])
    X_train, X_test, y_train, y_test = train_test(
        X, y, config.get("test_size", 0.2), random_state
    )

    cv = StratifiedKFold(
        n_splits=config.get("cv_folds", 5), shuffle=True, random_state=random_state
    )

    rows = []
    for name, model in build_candidates(random_state).items():
        start = time.time()
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING)

        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        row["seconds"] = time.time() - start
        rows.append(row)

        print(f"{name:18s} recall={row['recall_mean']:.3f}±{row['recall_std']:.3f}  "
              f"precision={row['precision_mean']:.3f}±{row['precision_std']:.3f}  "
              f"f1={row['f1_mean']:.3f}  pr_auc={row['pr_auc_mean']:.3f}  "
              f"({row['seconds']:.0f}s)")

    out_dir = Path("reports") / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows).set_index("model").round(4)
    results.to_csv(out_dir / "cv_comparison.csv")
    print(f"\nSaved to {out_dir / 'cv_comparison.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Compare models with cross-validation.")
    parser.add_argument("config", help="Path to a YAML config file")
    args = parser.parse_args()
    run(load_config(args.config))