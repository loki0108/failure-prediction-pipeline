import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve

from src.config import load_config
from src.evaluate import evaluate, print_report
from src.models import build_candidates
from src.prepare import make_cv, prepare_splits


def oof_predictions(model, X, y, cv):
    """Out-of-fold probabilities for any CV, including TimeSeriesSplit
    (where the earliest block is never validated, so only covered rows are returned)."""
    y_true, y_score = [], []
    for train_idx, val_idx in cv.split(X, y):
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_true.append(y.iloc[val_idx].values)
        y_score.append(fold_model.predict_proba(X.iloc[val_idx])[:, 1])
    return np.concatenate(y_true), np.concatenate(y_score)


def choose_threshold(y_true, y_score, min_recall=None):
    """min_recall=None -> maximise F1; otherwise best precision with recall >= min_recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision, recall = precision[:-1], recall[:-1]

    if min_recall is None:
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        idx = np.argmax(f1)
    else:
        meets_target = recall >= min_recall
        if not meets_target.any():
            raise ValueError(f"No threshold reaches recall {min_recall}")
        idx = np.argmax(np.where(meets_target, precision, -1))

    return thresholds[idx], precision[idx], recall[idx]


def tradeoff_table(y_true, y_score, cutoffs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
    rows = []
    for t in cutoffs:
        pred = (y_score >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        rows.append({
            "cutoff": t,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "caught": tp,
            "false_alarms": fp,
        })
    return pd.DataFrame(rows)


def plot_pr_curve(y_true, y_score, chosen, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    threshold, p, r = chosen
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.scatter([r], [p], color="red", zorder=3, label=f"chosen cut-off = {threshold:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (out-of-fold, training set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def selected_features(model):
    """Which features survived cleaning and selection, ranked by selection score."""
    steps = model.named_steps
    features = steps["clean"].columns_
    if "select" not in steps:
        return None
    selector = steps["select"]
    table = pd.DataFrame({"feature": features, "f_score": selector.scores_})
    return table[selector.get_support()].sort_values("f_score", ascending=False)


def run(config):
    random_state = config.get("random_state", 42)
    model_name = config.get("model", "random_forest")
    min_recall = config.get("min_recall")

    X_train, X_test, y_train, y_test = prepare_splits(config)
    model = build_candidates(
        random_state, config.get("cleaning"), config.get("select_k")
    )[model_name]
    out_dir = Path("reports") / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Part 1: choose the cut-off using the TRAINING set only ----
    y_oof, oof_score = oof_predictions(model, X_train, y_train, make_cv(config))

    print(f"\n=== Cut-off trade-off for {model_name} (out-of-fold, training set) ===")
    print(tradeoff_table(y_oof, oof_score).round(3).to_string(index=False))

    chosen = choose_threshold(y_oof, oof_score, min_recall)
    threshold, cv_precision, cv_recall = chosen
    rule = "max F1" if min_recall is None else f"recall >= {min_recall}"
    print(f"\nChosen cut-off ({rule}): {threshold:.3f}  "
          f"-> CV precision {cv_precision:.3f}, CV recall {cv_recall:.3f}")
    plot_pr_curve(y_oof, oof_score, chosen, out_dir / "pr_curve.png")

    # ---- Part 2: unlock the test set, ONCE ----
    model.fit(X_train, y_train)
    print(f"Features kept after cleaning: {len(model.named_steps['clean'].columns_)} "
          f"of {X_train.shape[1]}")

    features = selected_features(model)
    if features is not None:
        print("\nTop selected features:")
        print(features.head(10).round(2).to_string(index=False))
        features.to_csv(out_dir / "selected_features.csv", index=False)

    test_score = model.predict_proba(X_test)[:, 1]
    results = {
        "default_0.5": evaluate(y_test, (test_score >= 0.5).astype(int), test_score),
        f"tuned_{threshold:.3f}": evaluate(y_test, (test_score >= threshold).astype(int), test_score),
    }
    print("\n=== FINAL TEST SET RESULTS ===")
    for name, metrics in results.items():
        print_report(f"{model_name} @ {name}", metrics)

    pd.DataFrame(results).T.round(4).to_csv(out_dir / "final_metrics.csv")
    print(f"\nSaved results to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Tune the threshold, then evaluate once on the test set.")
    parser.add_argument("config", help="Path to a YAML config file")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()