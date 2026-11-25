from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.cleaning import SensorCleaner


def build_candidates(random_state=42, cleaning=None):
    """All models we compare. Each starts with SensorCleaner, so column
    dropping is re-learned inside every CV fold (no leakage)."""
    cleaning = cleaning or {}

    def clean():
        return ("clean", SensorCleaner(**cleaning))

    return {
        "logreg": Pipeline([
            clean(),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "logreg_balanced": Pipeline([
            clean(),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=random_state)),
        ]),
        "logreg_smote": Pipeline([
            clean(),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("smote", SMOTE(random_state=random_state)),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            clean(),
            ("model", RandomForestClassifier(
                n_estimators=100, n_jobs=-1, random_state=random_state)),
        ]),
        "gradient_boosting": Pipeline([
            clean(),
            ("model", HistGradientBoostingClassifier(random_state=random_state)),
        ]),
    }