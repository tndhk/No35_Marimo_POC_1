# Plan: Bento4 Analysis (Gemini Conductor)

## Phase 1: Environment & Data Setup
- [x] Task: Update `projects/bento4_gemini_conductor/requirements.txt` with necessary libraries (marimo, polars, altair, scikit-learn, numpy).
- [x] Task: Create `projects/bento4_gemini_conductor/notebooks/bento4_analysis.py` (marimo notebook) and import libraries.
- [x] Task: Load `train` and `test` data using Polars and display the first few rows to verify loading.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Environment & Data Setup' (Protocol in workflow.md)

## Phase 2: Exploratory Data Analysis (EDA)
- [x] Task: Inspect data types and missing values in train and test sets.
- [x] Task: Visualize the target variable (`y`) distribution.
- [x] Task: Visualize sales trends over time (Time Series plot).
- [x] Task: Analyze correlations between numerical features and the target.
- [x] Task: Perform diagnostic check (stats, correlations, monthly trends) to confirm data quality.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Exploratory Data Analysis (EDA)' (Protocol in workflow.md)

## Phase 3: Preprocessing & Feature Engineering
- [x] Task: Handle missing values (imputation or removal).
- [x] Task: Process 'Date' column (ensure correct format and extract features like Month, Day, DayOfWeek if useful).
- [x] Task: Encode categorical variables if present.
- [x] Task: Split training data into Train and Validation sets.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Preprocessing & Feature Engineering' (Protocol in workflow.md)

## Phase 4: Modeling & Evaluation
- [x] Task: Train a baseline model (e.g., Linear Regression or Random Forest).
- [x] Task: Evaluate the model on the Validation set using RMSE.
- [x] Task: Visualize Actual vs Predicted values on the validation set.
- [x] Task: Refine model (hyperparameter tuning or trying different algorithms) to improve RMSE.
- [x] Task: Feature Engineering: Add 'payday' flag and extract key menu items (e.g., Curry) from 'name'.
- [x] Task: Retrain model with new features and evaluate RMSE (Target: ~10).
- [x] Task: Switch model to GradientBoostingRegressor to improve handling of sparse features.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Modeling & Evaluation' (Protocol in workflow.md)

## Phase 5: Submission Generation
- [x] Task: Retrain model on full training data (optional, or use best model).
- [x] Task: Generate predictions for the `test` dataset.
- [x] Task: Format the submission dataframe according to `kadai.md` (No header, yyyy-m-d date format).
- [x] Task: Save the submission file as `projects/bento4_gemini_conductor/data/submission.csv`.
- [x] Task: Conductor - User Manual Verification 'Phase 5: Submission Generation' (Protocol in workflow.md)
