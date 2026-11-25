from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.cleaning import SensorCleaner


def build_candidates(random_state=42, cleaning=None, select_k=None):
    """All models we compare. Preprocessing lives inside each pipeline,
    so it is re-learned in every CV fold (no leakage)."""
    cleaning = cleaning or {}

    def prep(scale=True, impute=True):
        steps = [("clean", SensorCleaner(**cleaning))]
        if impute:
            steps.append(("impute", SimpleImputer(strategy="median")))
        if select_k:
            steps.append(("select", SelectKBest(f_classif, k=select_k)))
        if scale:
            steps.append(("scale", StandardScaler()))
        return steps

    # Trees handle NaN natively, but SelectKBest doesn't, so impute only if selecting
    tree_prep = lambda: prep(scale=False, impute=bool(select_k))

    return {
        "logreg": Pipeline(prep() + [
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "logreg_balanced": Pipeline(prep() + [
            ("model", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=random_state)),
        ]),
        "logreg_regularized": Pipeline(prep() + [
            ("model", LogisticRegression(
                max_iter=1000, class_weight="balanced", C=0.1, random_state=random_state)),
        ]),
        "logreg_smote": Pipeline(prep() + [
            ("smote", SMOTE(random_state=random_state)),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "random_forest": Pipeline(tree_prep() + [
            ("model", RandomForestClassifier(
                n_estimators=100, n_jobs=-1, random_state=random_state)),
        ]),
        "gradient_boosting": Pipeline(tree_prep() + [
            ("model", HistGradientBoostingClassifier(random_state=random_state)),
        ]),
    }