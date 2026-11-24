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

## Results

### Stage 1: Credit Card Fraud (baseline)
Test set: 56,746 transactions, 95 frauds (stratified 80/20 split, after removing duplicates)

| Model | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Always predict "normal" | 0.9983 | 0.00 | 0.00 | 0.00 | 0.0017 |
| Logistic regression (scaled) | 0.9991 | 0.85 | 0.58 | 0.69 | 0.69 |

Accuracy is misleading here: predicting "normal" for everything scores 99.83% while catching no fraud.
Logistic regression caught 55 of 95 frauds with 10 false alarms; recall is the weak point.

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