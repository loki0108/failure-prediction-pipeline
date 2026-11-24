from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(y_true, y_pred, y_score):
    """Return the metrics we care about as a dict."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def print_report(name, metrics):
    """Print metrics in a readable block."""
    print(f"\n--- {name} ---")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}")
    print(f"PR-AUC   : {metrics['pr_auc']:.4f}")
    print(f"Caught {metrics['tp']} of {metrics['tp'] + metrics['fn']} failures, "
          f"{metrics['fp']} false alarms")