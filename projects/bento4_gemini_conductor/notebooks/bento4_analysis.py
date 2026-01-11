import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    return (
        GradientBoostingRegressor,
        RandomForestRegressor,
        alt,
        mean_squared_error,
        mo,
        np,
        pd,
        pl,
        train_test_split,
    )


@app.cell
def _(GradientBoostingRegressor, X_train, mo, y_train):
    # Initialize and train GradientBoostingRegressor
    # Using somewhat aggressive parameters for boosting
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    mo.md("## Advanced Model (GradientBoosting) Trained")
    return (model,)


@app.cell
def _(alt, mo, pd, y_pred, y_val):
    # Prepare data for scatter plot
    _plot_df = pd.DataFrame({
        "actual": y_val,
        "predicted": y_pred
    })

    _chart = alt.Chart(_plot_df).mark_circle(size=60).encode(
        x=alt.X("actual:Q", title="Actual Sales"),
        y=alt.Y("predicted:Q", title="Predicted Sales"),
        tooltip=["actual", "predicted"]
    ).properties(
        title="Actual vs Predicted Sales (Validation Set)",
        width=400,
        height=400
    )

    # Diagonal line
    _line = alt.Chart(pd.DataFrame({'x': [20, 180], 'y': [20, 180]})).mark_line(color='red', strokeDash=[5, 5]).encode(
        x='x',
        y='y'
    )

    mo.vstack([
        mo.md("## Prediction Visualization"),
        _chart + _line
    ])
    return


@app.cell
def _(X, alt, mo, model, pd):
    # Feature Importance
    _importances = model.feature_importances_
    _feat_imp_df = pd.DataFrame({
        "feature": X.columns,
        "importance": _importances
    }).sort_values("importance", ascending=False)

    _chart = alt.Chart(_feat_imp_df.head(15)).mark_bar().encode(
        x="importance:Q",
        y=alt.Y("feature:N", sort="-x")
    ).properties(title="Top 15 Feature Importances", width=400)

    mo.vstack([
        mo.md("## Feature Importance Analysis"),
        _chart
    ])
    return


@app.cell
def _(X_val, mo, y_pred, y_val):
    # Error Analysis: Top 5 errors
    _val_df = X_val.copy()
    _val_df["actual"] = y_val
    _val_df["predicted"] = y_pred
    _val_df["abs_error"] = (y_val - y_pred).abs()

    _top_errors = _val_df.sort_values("abs_error", ascending=False).head(5)

    mo.vstack([
        mo.md("## Residual Analysis: Top 5 Worst Predictions"),
        _top_errors
    ])
    return


@app.cell
def _(GradientBoostingRegressor, X, mo, y):
    # Retrain on ALL training data for final submission
    final_model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    final_model.fit(X, y)
    
    mo.md("## Final Model Retrained on Full Dataset")
    return (final_model,)


@app.cell
def _(final_model, mo, pd, test_enhanced):
    # Prepare test features
    _exclude_cols = ["datetime", "name", "remarks", "event", "y"]
    _features = [c for c in test_enhanced.columns if c not in _exclude_cols]
    
    X_test = test_enhanced.select(_features).to_pandas()
    
    # Predict
    test_preds = final_model.predict(X_test)
    
    # Format submission: [Date, Prediction]
    # Date must be yyyy-m-d
    submission_df = pd.DataFrame({
        "datetime": test_enhanced["datetime"].dt.strftime("%Y-%-m-%-d"),
        "y": test_preds
    })
    
    mo.vstack([
        mo.md("## Submission Preview (First 5 rows)"),
        submission_df.head(5)
    ])
    return X_test, submission_df, test_preds


@app.cell
def _(mo, os, submission_df):
    # Save submission
    _output_path = "../data/submission.csv"
    submission_df.to_csv(_output_path, index=False, header=False)
    
    mo.md(f"## Submission saved to: `{os.path.abspath(_output_path)}`")
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Bento Sales Prediction Analysis (Bento4)
    """)
    return


@app.cell
def _(pl):
    import os
    # Try to find the data file in common locations
    _possible_paths = [
        "../data/bento_train.csv",  # Relative to notebooks dir
        "projects/bento4_gemini_conductor/data/bento_train.csv", # Relative to project root
        "data/bento_train.csv" # Relative if inside project dir
    ]
    
    _train_path = None
    for p in _possible_paths:
        if os.path.exists(p):
            _train_path = p
            break
            
    if _train_path is None:
        raise FileNotFoundError(f"Could not find bento_train.csv. Checked: {_possible_paths}")
        
    _test_path = _train_path.replace("bento_train.csv", "bento_test.csv")

    train = pl.read_csv(_train_path)
    test = pl.read_csv(_test_path)
    return test, train


@app.cell
def _(mo, train):
    mo.vstack([
        mo.md("## Train Data (First 5 rows)"),
        train.head(5)
    ])
    return


@app.cell
def _(mo, test):
    mo.vstack([
        mo.md("## Test Data (First 5 rows)"),
        test.head(5)
    ])
    return


@app.cell
def _(mo, train):
    mo.vstack([
        mo.md("## Data Inspection: Train"),
        mo.hstack([
            mo.vstack([mo.md("### Schema"), train.schema]),
            mo.vstack([mo.md("### Null Counts"), train.null_count()])
        ])
    ])
    return


@app.cell
def _(mo, test):
    mo.vstack([
        mo.md("## Data Inspection: Test"),
        mo.hstack([
            mo.vstack([mo.md("### Schema"), test.schema]),
            mo.vstack([mo.md("### Null Counts"), test.null_count()])
        ])
    ])
    return


@app.cell
def _(alt, mo, train):
    _chart = alt.Chart(train.to_pandas()).mark_bar().encode(
        alt.X("y:Q", bin=alt.Bin(maxbins=20), title="Bento Sales Count (y)"),
        y='count()',
    ).properties(
        title="Distribution of Target Variable (y)",
        width=500,
        height=300
    )
    mo.vstack([
        mo.md("## Target Distribution"),
        _chart
    ])
    return


@app.cell
def _(alt, mo, pl, train):
    # Ensure datetime is parsed
    _df_trend = train.with_columns(
        pl.col("datetime").str.to_date("%Y-%m-%d")
    ).to_pandas()

    _chart = alt.Chart(_df_trend).mark_line().encode(
        x=alt.X("datetime:T", title="Date"),
        y=alt.Y("y:Q", title="Sales Count (y)")
    ).properties(
        title="Sales Trend Over Time",
        width=800,
        height=400
    )
    mo.vstack([
        mo.md("## Time Series Trend"),
        _chart
    ])
    return


@app.cell
def _(alt, mo, train):
    # Select numeric columns for correlation
    _numeric_cols = train.select([
        c for c, t in zip(train.columns, train.dtypes) 
        if t in [float, int] or t.is_numeric()
    ])

    _corr_data = _numeric_cols.to_pandas().corr()["y"].reset_index()
    _corr_data.columns = ["feature", "correlation"]
    # Remove 'y' itself from correlation plot
    _corr_data = _corr_data[_corr_data["feature"] != "y"]

    _chart = alt.Chart(_corr_data).mark_bar().encode(
        x=alt.X("correlation:Q", title="Correlation with y"),
        y=alt.Y("feature:N", sort="-x", title="Feature")
    ).properties(
        title="Correlation with Target Variable (y)",
        width=500,
        height=300
    )

    mo.vstack([
        mo.md("## Feature Correlation"),
        _chart
    ])
    return


@app.cell
def _(mo, pl, test, train):
    # Impute kcal with mean
    _kcal_mean = train["kcal"].mean()

    train_filled = train.with_columns([
        pl.col("kcal").fill_null(_kcal_mean),
        pl.col("remarks").fill_null("None"),
        pl.col("event").fill_null("None"),
        pl.col("payday").fill_null(0.0),
        pl.col("precipitation").replace("--", "0").cast(pl.Float64).fill_null(0.0)
    ])

    test_filled = test.with_columns([
        pl.col("kcal").fill_null(_kcal_mean),
        pl.col("remarks").fill_null("None"),
        pl.col("event").fill_null("None"),
        pl.col("payday").fill_null(0.0),
        pl.col("precipitation").replace("--", "0").cast(pl.Float64).fill_null(0.0)
    ])

    mo.vstack([
        mo.md("## Missing Values Handled"),
        mo.md("### Train Null Counts after processing:"),
        train_filled.null_count(),
        mo.md("### Test Null Counts after processing:"),
        test_filled.null_count()
    ])
    return test_filled, train_filled


@app.cell
def _(mo, pl, test_filled, train_filled):
    # Process datetime and extract month/day
    train_preprocessed = train_filled.with_columns([
        pl.col("datetime").str.to_date("%Y-%m-%d")
    ]).with_columns([
        pl.col("datetime").dt.month().alias("month"),
        pl.col("datetime").dt.day().alias("day")
    ])

    test_preprocessed = test_filled.with_columns([
        pl.col("datetime").str.to_date("%Y-%m-%d")
    ]).with_columns([
        pl.col("datetime").dt.month().alias("month"),
        pl.col("datetime").dt.day().alias("day")
    ])

    mo.vstack([
        mo.md("## Date Features Extracted"),
        train_preprocessed.head(5)
    ])
    return test_preprocessed, train_preprocessed


@app.cell
def _(mo, pl, test_preprocessed, train_preprocessed):
    # One-hot encoding for week and weather
    train_encoded = train_preprocessed.to_dummies(["week", "weather"])
    test_encoded = test_preprocessed.to_dummies(["week", "weather"])

    # Ensure both have the same columns
    _train_cols = set(train_encoded.columns)
    _test_cols = set(test_encoded.columns)

    # Add missing columns to test with 0
    for col in _train_cols - _test_cols:
        if col != "y":
            test_encoded = test_encoded.with_columns(pl.lit(0).alias(col))

    # Add missing columns to train with 0 (if any from test)
    for col in _test_cols - _train_cols:
        train_encoded = train_encoded.with_columns(pl.lit(0).alias(col))

    mo.vstack([
        mo.md("## Categorical Variables Encoded"),
        train_encoded.head(5)
    ])
    return test_encoded, train_encoded


@app.cell
def _(mo, pl, test_encoded, train_encoded):
    # Feature Engineering: Payday and Menu Items
    def add_features(df):
        return df.with_columns([
            pl.col("payday").fill_null(0.0).cast(pl.Int32).alias("payday_flag"),
            pl.col("name").str.contains("カレー").cast(pl.Int32).alias("is_curry"),
            pl.col("remarks").str.contains("お楽しみメニュー").cast(pl.Int32).fill_null(0).alias("is_fun_menu")
        ])

    train_enhanced = add_features(train_encoded)
    test_enhanced = add_features(test_encoded)

    mo.vstack([
        mo.md("## Enhanced Features Added"),
        train_enhanced.select(["datetime", "y", "payday_flag", "is_curry", "is_fun_menu"]).head(5)
    ])
    return test_enhanced, train_enhanced


@app.cell
def _(mo, train_enhanced, train_test_split):
    # Select features for modeling
    _exclude_cols = ["datetime", "name", "remarks", "event", "y"]
    _features = [c for c in train_enhanced.columns if c not in _exclude_cols]

    X = train_enhanced.select(_features).to_pandas()
    y = train_enhanced["y"].to_pandas()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    mo.vstack([
        mo.md("## Data Split into Train and Validation (Enhanced)"),
        mo.md(f"Features: {', '.join(_features)}"),
        mo.md(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
    ])
    return X, X_train, X_val, y_train, y_val


@app.cell
def _(mo, train_preprocessed):
    mo.vstack([
        mo.md("## Diagnostic: Statistics of y"),
        train_preprocessed.select("y").describe()
    ])
    return


@app.cell
def _(alt, mo, pl, train_preprocessed):
    # Correlation Check
    _numeric = train_preprocessed.select([
        c for c, t in zip(train_preprocessed.columns, train_preprocessed.dtypes)
        if t in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]
    ])
    _corr = _numeric.to_pandas().corr()["y"].reset_index()
    _corr.columns = ["feature", "correlation"]
    _corr = _corr[_corr["feature"] != "y"].sort_values("correlation", ascending=False)

    _chart = alt.Chart(_corr).mark_bar().encode(
        x="correlation:Q",
        y=alt.Y("feature:N", sort="-x")
    ).properties(title="Feature Correlation with y")

    mo.vstack([
        mo.md("## Diagnostic: Feature Correlation"),
        _chart
    ])
    return


@app.cell
def _(alt, mo, pl, train_preprocessed):
    # Monthly average sales
    _monthly_avg = train_preprocessed.group_by("month").agg(
        pl.col("y").mean().alias("avg_sales")
    ).sort("month")

    _chart = alt.Chart(_monthly_avg.to_pandas()).mark_line(point=True).encode(
        x="month:O",
        y="avg_sales:Q"
    ).properties(title="Average Sales by Month", width=400)

    mo.vstack([
        mo.md("## Diagnostic: Monthly Trends"),
        _chart
    ])
    return


if __name__ == "__main__":
    app.run()
