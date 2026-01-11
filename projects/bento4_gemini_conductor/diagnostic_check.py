import polars as pl
import numpy as np

# Load data
train_path = "projects/bento4_gemini_conductor/data/bento_train.csv"
train = pl.read_csv(train_path)

print("--- 1. Target Variable (y) Statistics ---")
print(train["y"].describe())

print("\n--- 2. Missing Values (Raw Data) ---")
print(train.null_count())

# Preprocessing (Replicating logic)
kcal_mean = train["kcal"].mean()
train_filled = train.with_columns([
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

# Correlations
# Select numeric columns
numeric_cols = [c for c, t in zip(train_filled.columns, train_filled.dtypes) if t in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]]
# Filter out 'y' from features to check correlation against
features = [c for c in numeric_cols if c != "y"]

print("\n--- 3. Top Correlations with y ---")
correlations = []
for feat in features:
    corr = train_filled.select(pl.corr("y", feat)).item()
    if corr is not None:
        correlations.append((feat, corr))

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

for feat, corr in correlations[:10]:
    print(f"{feat}: {corr:.4f}")

print("\n--- 4. Feature Check (First 3 rows processed) ---")
print(train_filled.select(["datetime", "y", "month", "day", "kcal", "precipitation", "temperature"]).head(3))
