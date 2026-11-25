# Failure Prediction Pipeline

A reusable, config-driven pipeline for predicting rare failures in imbalanced tabular data.

- **Stage 1:** Credit Card Fraud dataset (baseline, learning the pipeline)
- **Stage 2:** SECOM semiconductor manufacturing dataset (UCI): 590 sensors, pass/fail per production run

**Status:** in progress

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Data

### Stage 1: Credit Card Fraud
1. Download from Kaggle: "Credit Card Fraud Detection" (mlg-ulb)
2. Place the file at `data/raw/creditcard.csv`
3. Load it into SQLite:
   ```bash
   python -m src.ingest configs/creditcard.yaml
   ```

## How to run
Every step takes a config file, so the same commands work for any dataset.
```bash
python -m src.ingest configs/creditcard.yaml          # CSV -> SQLite
python -m src.explore configs/creditcard.yaml         # EDA reports in reports/creditcard/
python -m src.train_baseline configs/creditcard.yaml  # baseline vs logistic regression
python -m src.compare_models configs/creditcard.yaml  # 5-fold CV model comparison
python -m src.final_evaluation configs/creditcard.yaml # tune cut-off on CV, then one-time test evaluation
```

## Results

### Stage 1: Credit Card Fraud (baseline)
Test set: 56,746 transactions, 95 frauds (stratified 80/20 split, after removing duplicates)

| Model | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Always predict "normal" | 0.9983 | 0.00 | 0.00 | 0.00 | 0.0017 |
| Logistic regression (scaled) | 0.9991 | 0.85 | 0.58 | 0.69 | 0.69 |

Accuracy is misleading here: predicting "normal" for everything scores 99.83% while catching no fraud.
Logistic regression caught 55 of 95 frauds with 10 false alarms; recall is the weak point.

### Stage 1: Model comparison (5-fold stratified cross-validation on the training set)

| Model | Recall | Precision | F1 | PR-AUC |
|---|---|---|---|---|
| Logistic regression | 0.61 ± 0.06 | 0.86 ± 0.03 | 0.71 | 0.75 |
| Logistic regression, class_weight="balanced" | 0.92 ± 0.03 | 0.06 ± 0.01 | 0.11 | 0.75 |
| Logistic regression + SMOTE | 0.91 ± 0.03 | 0.06 ± 0.01 | 0.10 | 0.75 |
| Random forest | 0.78 ± 0.07 | 0.94 ± 0.02 | 0.85 | 0.83 |
| HistGradientBoosting (defaults) | 0.55 ± 0.14 | 0.47 ± 0.07 | 0.50 | 0.49 |

### Stage 1: Final model (random forest, evaluated once on the held-out test set)
Cut-off chosen by maximising F1 on out-of-fold training predictions (CV: precision 0.92, recall 0.81 at 0.40).

| Cut-off | Precision | Recall | F1 | PR-AUC | Caught | False alarms |
|---|---|---|---|---|---|---|
| 0.50 (default) | 0.97 | 0.73 | 0.83 | 0.79 | 69 / 95 | 2 |
| 0.40 (tuned) | 0.96 | 0.76 | 0.85 | 0.79 | 72 / 95 | 3 |

![Precision-recall curve](reports/creditcard/pr_curve.png)

## Log: what broke / what changed

### Credit Card: exploration (Module 2)
- 284,807 rows, 31 columns, 0 missing values; fraud rate 0.17% (492 rows).
- 1,081 exact duplicate rows → dropped before splitting, so the same row can't appear in both train and test (leakage).
- Strongest separating features: V17, V14, V12, V10 (frauds take strongly negative values).
  Every feature still overlaps with the normal range, so a multi-feature model is needed rather than threshold rules.

### Credit Card: baseline (Module 3)
- Added a "dumb" baseline (always predict normal) to show why accuracy can't be the metric.
- Scaling is done inside a scikit-learn Pipeline so it's fit on training data only (no test leakage).
- Only 95 frauds in the test set → each fraud ≈ 1% recall; single-split results are noisy. Cross-validation is next.

### Credit Card: model comparison (Module 4A)
- Switched from a single train/test split to 5-fold stratified cross-validation on the training set;
  the test set is held back for one final evaluation.
- Class weighting and SMOTE raised recall from 0.61 to ~0.91 but precision fell to ~0.06 (≈15 false alarms per fraud).
  PR-AUC stayed at 0.75 for all three: they moved the decision threshold rather than improving the model.
- SMOTE runs inside an imblearn Pipeline so synthetic samples are created from training folds only (no leakage).
- Random forest was best (PR-AUC 0.83, F1 0.85) → selected for threshold tuning.
- HistGradientBoosting underperformed with high variance (PR-AUC 0.49 ± large spread). Disabling early stopping
  made it worse (0.35), so early stopping wasn't the cause. Not investigated further; possible next step is a lower learning rate.

  ### Credit Card: threshold tuning + final test (Module 4B)
- Cut-off chosen from out-of-fold probabilities on the training set, never from the test set.
- Selection rule (max F1) fixed before looking at test results.
- Tuned cut-off 0.40 caught 3 more frauds than the default 0.5 for 1 extra false alarm.
- Test recall (0.76) slightly below CV recall (0.81), within the CV spread (±0.07) → CV estimate was reliable.