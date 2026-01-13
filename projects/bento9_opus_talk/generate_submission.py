"""
submission.csv生成スクリプト

marimoノートブック(bento9_analysis.py)のロジックを使用して、
submission.csvを生成します。
"""

import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import jpholiday
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import optuna
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# パス設定
project_root = Path(__file__).parent
data_path_train = project_root / "data" / "bento_train.csv"
data_path_test = project_root / "data" / "bento_test.csv"

# データ読み込み
print("データ読み込み中...")
df_train = pl.read_csv(data_path_train)
df_test = pl.read_csv(data_path_test)

print(f"訓練データ: {df_train.shape[0]} 行 × {df_train.shape[1]} 列")
print(f"テストデータ: {df_test.shape[0]} 行 × {df_test.shape[1]} 列")

# 日付特徴量の追加
def add_date_features(df):
    return df.with_columns([
        pl.col("datetime").str.strptime(pl.Date, "%Y-%m-%d").alias("date"),
    ]).with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.day().alias("day"),
        pl.col("date").dt.weekday().alias("weekday"),  # 0=月, 6=日
    ])

print("日付特徴量追加中...")
df_train_fe = add_date_features(df_train)
df_test_fe = add_date_features(df_test)

# 祝日フラグの追加
def is_holiday(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return 1 if jpholiday.is_holiday(dt) else 0
    except:
        return 0

print("祝日フラグ追加中...")
df_train_fe = df_train_fe.with_columns([
    pl.col("datetime").map_elements(is_holiday, return_dtype=pl.Int64).alias("is_holiday")
])
df_test_fe = df_test_fe.with_columns([
    pl.col("datetime").map_elements(is_holiday, return_dtype=pl.Int64).alias("is_holiday")
])

# カテゴリカルエンコーディング
print("カテゴリカルエンコーディング中...")
week_map = {"月": 0, "火": 1, "水": 2, "木": 3, "金": 4, "土": 5, "日": 6}
weather_map = {"快晴": 0, "晴れ": 1, "薄曇": 2, "曇": 3, "雨": 4}

df_train_fe = df_train_fe.with_columns([
    pl.col("week").replace(week_map).alias("week_encoded"),
    pl.col("weather").replace(weather_map).alias("weather_encoded")
])
df_test_fe = df_test_fe.with_columns([
    pl.col("week").replace(week_map).alias("week_encoded"),
    pl.col("weather").replace(weather_map).alias("weather_encoded")
])

# 数値特徴量の変換
print("数値特徴量変換中...")
df_train_fe = df_train_fe.with_columns([
    pl.col("soldout").cast(pl.Int64),
    pl.col("payday").fill_null(0).cast(pl.Int64)
])
df_test_fe = df_test_fe.with_columns([
    pl.col("soldout").cast(pl.Int64),
    pl.col("payday").fill_null(0).cast(pl.Int64)
])

# 欠損値補完（訓練データの統計量を使用）
print("欠損値補完中...")
train_kcal_median = df_train_fe["kcal"].median()
train_precipitation_median = df_train_fe["precipitation"].median()
train_temp_median = df_train_fe["temperature"].median()

df_train_filled = df_train_fe.with_columns([
    pl.col("kcal").fill_null(train_kcal_median),
    pl.col("precipitation").fill_null(train_precipitation_median),
    pl.col("temperature").fill_null(train_temp_median)
])
df_test_filled = df_test_fe.with_columns([
    pl.col("kcal").fill_null(train_kcal_median),
    pl.col("precipitation").fill_null(train_precipitation_median),
    pl.col("temperature").fill_null(train_temp_median)
])

# 特徴量選択
feature_cols = [
    "year", "month", "day", "weekday", "is_holiday",
    "week_encoded", "weather_encoded",
    "soldout", "payday",
    "kcal", "precipitation", "temperature"
]

X_train = df_train_filled.select(feature_cols).to_pandas()
y_train = df_train_filled.select("y").to_pandas()["y"]
X_test = df_test_filled.select(feature_cols).to_pandas()

print(f"特徴量数: {len(feature_cols)}")
print(f"訓練データ: {X_train.shape}")
print(f"テストデータ: {X_test.shape}")

# Optunaによるハイパーパラメータ最適化
print("\nOptunaでハイパーパラメータ最適化中...")

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        cv_scores.append(rmse)

    return np.mean(cv_scores)

# 最適化実行
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt'
})

print(f"\n最良CV RMSE: {study.best_value:.2f}")
print("最良パラメータ:")
for k, v in best_params.items():
    if k not in ['objective', 'metric', 'verbosity', 'boosting_type']:
        print(f"  {k}: {v}")

# 最良パラメータで再学習
print("\n最良パラメータでモデル訓練中...")
tscv = TimeSeriesSplit(n_splits=5)
lgb_cv_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        best_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    lgb_cv_scores.append(rmse)
    print(f"  Fold {fold_idx}: RMSE = {rmse:.2f}")

print(f"\nCV RMSE (平均): {np.mean(lgb_cv_scores):.2f} ± {np.std(lgb_cv_scores):.2f}")

# 全訓練データで最終モデルを訓練
print("\n全訓練データで最終モデル訓練中...")
train_data_full = lgb.Dataset(X_train, label=y_train)
lgb_model = lgb.train(
    best_params,
    train_data_full,
    num_boost_round=1000
)

# テストデータの予測
print("\nテストデータの予測実行中...")
y_pred = lgb_model.predict(X_test)
y_pred = np.maximum(y_pred, 0)  # 負の値を0にクリップ

# submission.csv生成
print("\nsubmission.csv生成中...")
dates = df_test_filled["datetime"].to_list()
predictions_int = [int(round(p)) for p in y_pred]

submission_df = pd.DataFrame({
    "datetime": dates,
    "y": predictions_int
})

submission_path = project_root / "submission.csv"
submission_df.to_csv(submission_path, index=False, header=False)

print(f"\n完了！")
print(f"ファイル保存先: {submission_path}")
print(f"行数: {len(submission_df)}")
print(f"\n最初の5行:")
print(submission_df.head())
