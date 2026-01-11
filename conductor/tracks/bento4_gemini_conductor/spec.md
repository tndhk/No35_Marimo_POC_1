# Specification: Bento Sales Prediction

## Goal
Predict the daily sales count of bento boxes sold at the company cafe floor.

## Data
- **Target Variable:** Number of bentos sold (`y`).
- **Input Files:**
  - `bento_train.csv`: Training data.
  - `bento_test.csv`: Test data (for prediction).
  - `bento_sample.csv`: Sample submission format.

## Submission Format
- **Format:** CSV (comma-separated), no header.
- **Columns:**
  1. Date (Index). Format: `yyyy-m-d` (e.g., `2014-10-1`, NOT `2014-10-01`).
  2. Predicted Sales Count.
- **Encoding:** UTF-8 (Note: Excel may require conversion to SJIS).

## Evaluation Metric
- **RMSE (Root Mean Squared Error):** Lower is better.
