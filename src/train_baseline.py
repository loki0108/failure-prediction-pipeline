import argparse
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.config import load_config
from src.evaluate import evaluate, print_report
from src.prepare import prepare_splits


def build_models(random_state):
    """The two models we compare: a useless baseline and a real one."""
    return {
        "always_normal": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
    ]),
    }


def run(config):
    random_state = config.get("random_state", 42)

    X_train, X_test, y_train, y_test = prepare_splits(config)

    results = {}
    for name, model in build_models(random_state).items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]
        results[name] = evaluate(y_test, y_pred, y_score)
        print_report(name, results[name])

    out_dir = Path("reports") / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).T.round(4).to_csv(out_dir / "baseline_metrics.csv")
    print(f"\nSaved metrics to {out_dir / 'baseline_metrics.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate baseline models.")
    parser.add_argument("config", help="Path to a YAML config file")
    args = parser.parse_args()
    run(load_config(args.config))

if __name__ == "__main__":
    main()