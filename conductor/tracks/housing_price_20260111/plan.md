# Plan: Housing Price Prediction Learning Project

## Phase 1: Environment Setup & Data Acquisition
- [ ] Task: Create project directory structure (`projects/housing_price/{data,notebooks}`)
- [ ] Task: Create `projects/housing_price/requirements.txt` with dependencies (marimo, polars, altair, scikit-learn)
- [ ] Task: Create `projects/housing_price/notebooks/housing_analysis.py` (marimo notebook) with initial imports and config
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Environment Setup & Data Acquisition' (Protocol in workflow.md)

## Phase 2: Data Loading & Exploratory Data Analysis (EDA)
- [ ] Task: Implement data loading function (California Housing dataset from scikit-learn)
- [ ] Task: Write tests for data loading
- [ ] Task: Add markdown cell explaining the dataset (Context & Features)
- [ ] Task: Implement interactive summary statistics view (Polars describe)
- [ ] Task: Implement distribution plots (Altair histograms)
- [ ] Task: Implement correlation heatmap (Altair)
- [ ] Task: Write tests for visualization components (ensure functions return charts)
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Data Loading & Exploratory Data Analysis (EDA)' (Protocol in workflow.md)

## Phase 3: Feature Engineering & Preprocessing
- [ ] Task: Add markdown explaining preprocessing steps
- [ ] Task: Implement data splitting (Train/Test)
- [ ] Task: Implement feature scaling (StandardScaler)
- [ ] Task: Write tests for preprocessing functions
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Feature Engineering & Preprocessing' (Protocol in workflow.md)

## Phase 4: Model Building & Evaluation
- [ ] Task: Add markdown explaining Linear Regression
- [ ] Task: Implement Linear Regression model training
- [ ] Task: Implement evaluation metrics calculation (RMSE, R2)
- [ ] Task: Write tests for model training and evaluation functions
- [ ] Task: Implement "Actual vs Predicted" scatter plot
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Model Building & Evaluation' (Protocol in workflow.md)
