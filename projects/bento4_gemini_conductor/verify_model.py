import polars as pl
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

def load_data():
    base_dir = "projects/bento4_gemini_conductor/data"
    train = pl.read_csv(os.path.join(base_dir, "bento_train.csv"))
    test = pl.read_csv(os.path.join(base_dir, "bento_test.csv") )
    return train, test

def preprocess(train, test):
    kcal_mean = train["kcal"].mean()
    
    def clean(df):
        return df.with_columns([
            pl.col("kcal").fill_null(kcal_mean),
            pl.col("remarks").fill_null("None"),
            pl.col("event").fill_null("None"),
            pl.col("payday").fill_null(0.0),
            pl.col("precipitation").replace("--", "0").cast(pl.Float64).fill_null(0.0),
            pl.col("datetime").str.to_date("%Y-%m-%d")
        ]).with_columns([
            pl.col("datetime").dt.month().alias("month"),
            pl.col("datetime").dt.day().alias("day")
        ])

    train_p = clean(train)
    test_p = clean(test)
    
    # One-hot encoding
    train_enc = train_p.to_dummies(["week", "weather"])
    test_enc = test_p.to_dummies(["week", "weather"])
    
    # Align columns
    train_cols = set(train_enc.columns)
    test_cols = set(test_enc.columns)
    
    for col in train_cols - test_cols:
        if col != "y":
            test_enc = test_enc.with_columns(pl.lit(0).alias(col))
            
    for col in test_cols - train_cols:
        train_enc = train_enc.with_columns(pl.lit(0).alias(col))
        
    return train_enc, test_enc

def feature_engineering(df):
    return df.with_columns([
        pl.col("payday").fill_null(0.0).cast(pl.Int32).alias("payday_flag"),
        pl.col("name").str.contains("カレー").cast(pl.Int32).alias("is_curry"),
        pl.col("remarks").str.contains("お楽しみメニュー").cast(pl.Int32).fill_null(0).alias("is_fun_menu")
    ])

def main():
    print("Loading Data...")
    train, test = load_data()
    
    print("Preprocessing...")
    train_enc, test_enc = preprocess(train, test)
    
    print("Feature Engineering...")
    train_enhanced = feature_engineering(train_enc)
    
    # Split
    exclude_cols = ["datetime", "name", "remarks", "event", "y"]
    features = [c for c in train_enhanced.columns if c not in exclude_cols]
    
    print(f"Features: {features}")
    
    X = train_enhanced.select(features).to_pandas()
    y = train_enhanced["y"].to_pandas()
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Model (GradientBoosting)...")
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"Validation RMSE: {rmse:.4f}")
    
    print("\nFeature Importances:")
    importances = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(10)
    print(importances)

if __name__ == "__main__":
    main()
