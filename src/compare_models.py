import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, f1_score

from src.config import load_config
from src.models import build_candidates
from src.prepare import make_cv, prepare_splits

SCORING = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": make_scorer(f1_score, zero_division=0),
    "pr_auc": "average_precision",
}


def run(config):
    random_state = config.get("random_state", 42)

    X_train, X_test, y_train, y_test = prepare_splits(config)

    cv = make_cv(config)

    rows = []
    for name, model in build_candidates(random_state, config.get("cleaning"), config.get("select_k")).items():
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

if __name__ == "__main__":
    main()