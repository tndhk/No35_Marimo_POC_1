# Specification: Housing Price Prediction Learning Project

## 1. Overview
This track implements a new learning project within the monorepo, focusing on housing price prediction. The goal is to provide a hands-on tutorial for beginners to learn data analysis workflow using `marimo`, `Polars`, and `scikit-learn`.

## 2. Goals
- **Educational**: Teach the end-to-end process of a regression problem.
- **Interactive**: Use marimo features to allow users to interact with data and model parameters.
- **Standardized**: Follow the project's monorepo structure and guidelines.

## 3. Features
- **Data Loading**: Load the California Housing dataset (or similar open dataset).
- **EDA (Exploratory Data Analysis)**: Interactive visualization of distributions and correlations using Altair.
- **Preprocessing**: Handling missing values (if any) and scaling.
- **Modeling**: Linear Regression and potentially a more complex model (e.g., Random Forest or Gradient Boosting) for comparison.
- **Evaluation**: Visualize actual vs. predicted values and metrics (RMSE, R2).

## 4. Technical Requirements
- **Directory**: `projects/housing_price/`
- **Dependencies**: `marimo`, `polars`, `altair`, `scikit-learn`, `pandas` (if needed for dataset loading).
- **Style**: Adhere to `product-guidelines.md` (Conversational tone, high-density comments).

## 5. User Experience
The user will run `marimo edit notebooks/housing_analysis.py` and step through the notebook. They should see explanation markdown cells followed by code cells that they can execute and interact with (e.g., sliders for filtering data).
