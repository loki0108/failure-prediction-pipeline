from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def build_candidates(random_state=42):
    """All the models we compare. Each one is a full pipeline, so preprocessing
    is re-fit inside every cross-validation fold (no leakage)."""
    return {
        # Module 3 baseline, for comparison
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        # Fix 1: make each fraud count more during training
        "logreg_balanced": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=random_state)),
        ]),
        # Fix 2: create synthetic frauds (training folds only)
        "logreg_smote": Pipeline([
            ("scale", StandardScaler()),
            ("smote", SMOTE(random_state=random_state)),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        # Fix 3: stronger, tree-based models
        "random_forest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=random_state),
        "gradient_boosting": HistGradientBoostingClassifier(random_state=random_state),
    }