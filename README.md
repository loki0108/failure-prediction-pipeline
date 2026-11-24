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

## Log: what broke / what changed
(Filled in as the project progresses.)