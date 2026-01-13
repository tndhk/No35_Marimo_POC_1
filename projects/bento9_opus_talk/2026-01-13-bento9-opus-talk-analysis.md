# お弁当販売数予測モデル実装計画（bento9_opus_talk）

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** カフェフロアで販売されるお弁当の販売数を予測する機械学習モデルを構築し、RMSE評価でsubmission.csvを生成する

**Architecture:** Polarsでデータ処理、LightGBMを主要モデルとし、時系列分割によるCV検証を行う。marimoノートブックで対話的に分析・実装し、特徴量エンジニアリングと欠損値処理でデータリークを防止する。

**Tech Stack:** marimo, polars-lts-cpu, scikit-learn, lightgbm, altair, pandas, optuna, jpholiday

---

## Task 1: プロジェクト基盤構築

**Files:**
- Create: `projects/bento9_opus_talk/requirements.txt`
- Create: `projects/bento9_opus_talk/notebooks/bento9_analysis.py`

**Step 1: Write requirements.txt**

```txt
marimo>=0.19.0
polars-lts-cpu>=0.20.0
altair>=5.2.0
pyarrow>=12.0.0
pandas>=1.5.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
optuna>=3.0.0
jpholiday>=0.1.0
```

**Step 2: Create notebooks directory**

Run: `mkdir -p projects/bento9_opus_talk/notebooks`

**Step 3: Initialize marimo notebook structure**

Create `projects/bento9_opus_talk/notebooks/bento9_analysis.py` with basic app structure:

```python
import marimo

__generated_with = "0.19.0"
app = marimo.App()


# ===== グループA: 初期化 =====

@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import altair as alt
    from pathlib import Path
    import numpy as np
    import lightgbm as lgb
    import re
    import jpholiday
    from datetime import datetime
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    import warnings
    warnings.filterwarnings('ignore')
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return (mo, pl, pd, alt, Path, np, lgb, re, jpholiday, datetime,
            TimeSeriesSplit, Ridge, RandomForestRegressor,
            GradientBoostingRegressor, mean_squared_error, optuna)


if __name__ == "__main__":
    app.run()
```

**Step 4: Install dependencies**

Run: `cd projects/bento9_opus_talk && pip install -r requirements.txt`
Expected: All packages installed successfully

**Step 5: Verify marimo notebook runs**

Run: `cd projects/bento9_opus_talk/notebooks && marimo edit bento9_analysis.py`
Expected: marimo server starts without errors

**Step 6: Commit project setup**

```bash
git add projects/bento9_opus_talk/requirements.txt projects/bento9_opus_talk/notebooks/bento9_analysis.py
git commit -m "feat: initialize bento9_opus_talk project structure with dependencies

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: データロードと基本構造理解

**Files:**
- Modify: `projects/bento9_opus_talk/notebooks/bento9_analysis.py`

**Step 1: Add data loading cells**

Insert after the import cell:

```python
@app.cell
def _(Path, pl):
    # 訓練データロード
    data_path_train = Path(__file__).parent.parent / "data" / "bento_train.csv"
    df_train = pl.read_csv(data_path_train)
    return df_train, data_path_train


@app.cell
def _(Path, pl):
    # テストデータロード
    data_path_test = Path(__file__).parent.parent / "data" / "bento_test.csv"
    df_test = pl.read_csv(data_path_test)
    return df_test, data_path_test


@app.cell
def _(mo, df_train, df_test):
    # タイトルと概要
    mo.md(f"""
    # お弁当販売数予測 - bento9_opus_talk

    訓練データ: {df_train.shape[0]} 行 × {df_train.shape[1]} 列

    テストデータ: {df_test.shape[0]} 行 × {df_test.shape[1]} 列

    評価指標: RMSE（Root Mean Squared Error）
    """)
    return
```

**Step 2: Run marimo to verify data loads**

Run: `cd projects/bento9_opus_talk/notebooks && marimo edit bento9_analysis.py`
Expected: Data loads successfully, shapes displayed correctly

**Step 3: Commit data loading**

```bash
git add projects/bento9_opus_talk/notebooks/bento9_analysis.py
git commit -m "feat: add data loading for train and test datasets

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: 基本統計量と欠損値分析

**Files:**
- Modify: `projects/bento9_opus_talk/notebooks/bento9_analysis.py`

**Step 1: Add EDA section header cell**

```python
# ===== グループB: 基本EDA =====

@app.cell
def _(mo):
    mo.md("""
    ## 1. 基本統計とデータ理解
    """)
    return
```

**Step 2: Add basic statistics cell**

```python
@app.cell
def _(df_train):
    # 基本統計量
    train_stats = df_train.describe()
    train_stats
    return train_stats,
```

**Step 3: Add null value analysis cell**

```python
@app.cell
def _(df_train, pl, mo):
    # 欠損値確認
    null_counts = df_train.null_count()
    total_rows = df_train.shape[0]

    null_info = pl.DataFrame({
        "カラム": list(null_counts.columns),
        "欠損数": [null_counts[col][0] for col in null_counts.columns],
        "欠損率(%)": [
            round(null_counts[col][0] * 100 / total_rows, 2)
            for col in null_counts.columns
        ]
    }).filter(pl.col("欠損数") > 0)

    mo.md("### 欠損値確認")
    null_info
    return null_info,
```

**Step 4: Verify EDA outputs in marimo**

Run: `cd projects/bento9_opus_talk/notebooks && marimo edit bento9_analysis.py`
Expected: Statistics table and null count table display correctly

**Step 5: Commit EDA basics**

```bash
git add projects/bento9_opus_talk/notebooks/bento9_analysis.py
git commit -m "feat: add basic statistics and null value analysis

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: 目的変数と曜日の可視化

**Files:**
- Modify: `projects/bento9_opus_talk/notebooks/bento9_analysis.py`

**Step 1: Add target variable distribution cell**

```python
@app.cell
def _(df_train, alt):
    # 目的変数yの分布
    chart_y = alt.Chart(df_train.to_pandas()).mark_bar().encode(
        alt.X("y:Q", bin=alt.Bin(maxbins=30), title="販売数"),
        alt.Y("count()", title="頻度"),
        tooltip=["count()"]
    ).properties(
        width=600,
        height=300,
        title="販売数（y）の分布"
    )
    chart_y
    return chart_y,
```

**Step 2: Add day-of-week analysis cell**

```python
@app.cell
def _(df_train, alt):
    # 曜日別販売数
    week_order = ["月", "火", "水", "木", "金", "土", "日"]

    chart_week = alt.Chart(df_train.to_pandas()).mark_boxplot().encode(
        alt.X("week:N", title="曜日", sort=week_order),
        alt.Y("y:Q", title="販売数"),
        tooltip=["week", "y"]
    ).properties(
        width=600,
        height=300,
        title="曜日別販売数の分布"
    )
    chart_week
    return chart_week, week_order
```

**Step 3: Verify visualizations in marimo**

Run: `cd projects/bento9_opus_talk/notebooks && marimo edit bento9_analysis.py`
Expected: Distribution histogram and boxplot render correctly

**Step 4: Commit visualizations**

```bash
git add projects/bento9_opus_talk/notebooks/bento9_analysis.py
git commit -m "feat: add target variable and day-of-week visualizations

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: 特徴量エンジニアリング（日付・曜日・休日）

**Files:**
- Modify: `projects/bento9_opus_talk/notebooks/bento9_analysis.py`

**Step 1: Add feature engineering section header**

```python
# ===== グループC: 特徴量エンジニアリング =====

@app.cell
def _(mo):
    mo.md("""
    ## 2. 特徴量エンジニアリング
    """)
    return
```

**Step 2: Add date feature extraction cell**

```python
@app.cell
def _(df_train, df_test, pl, jpholiday, datetime):
    # 日付特徴量
    def add_date_features(df):
        return df.with_columns([
            pl.col("datetime").str.strptime(pl.Date, "%Y-%m-%d").alias("date"),
        ]).with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.day().alias("day"),
            pl.col("date").dt.weekday().alias("weekday"),  # 0=月, 6=日
        ])

    df_train_fe = add_date_features(df_train)
    df_test_fe = add_date_features(df_test)

    return df_train_fe, df_test_fe, add_date_features
```

**Step 3: Add holiday feature cell**

```python
@app.cell
def _(df_train_fe, df_test_fe, pl, jpholiday, datetime):
    # 祝日フラグ
    def is_holiday(date_str):
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return 1 if jpholiday.is_holiday(dt) else 0
        except:
            return 0

    df_train_fe2 = df_train_fe.with_columns([
        pl.col("datetime").map_elements(is_holiday, return_dtype=pl.Int64).alias("is_holiday")
    ])

    df_test_fe2 = df_test_fe.with_columns([
        pl.col("datetime").map_elements(is_holiday, return_dtype=pl.Int64).alias("is_holiday")
    ])

    return df_train_fe2, df_test_fe2, is_holiday
```

**Step 4: Verify feature engineering in marimo**

Run: `cd projects/bento9_opus_talk/notebooks && marimo edit bento9_analysis.py`
Expected: New columns (year, month, day, weekday, is_holiday) created successfully

**Step 5: Commit feature engineering**

```bash
git add projects/bento9_opus_talk/notebooks/bento9_analysis.py
git commit -m "feat: add date and holiday feature engineering

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: カテゴリカル特徴量のエンコーディング

**Files:**
- Modify: `projects/bento9_opus_talk/notebooks/bento9_analysis.py`

**Step 1: Add categorical encoding cell**

```python
@app.cell
def _(df_train_fe2, df_test_fe2, pl):
    # カテゴリカル特徴量のエンコーディング
    week_map = {"月": 0, "火": 1, "水": 2, "木": 3, "金": 4, "土": 5, "日": 6}
    weather_map = {"快晴": 0, "晴れ": 1, "薄曇": 2, "曇": 3, "雨": 4}

    df_train_fe3 = df_train_fe2.with_columns([
        pl.col("week").replace(week_map).alias("week_encoded"),
        pl.col("weather").replace(weather_map).alias("weather_encoded")
    ])

    df_test_fe3 = df_test_fe2.with_columns([
        pl.col("week").replace(week_map).alias("week_encoded"),
        pl.col("weather").replace(weather_map).alias("weather_encoded")
    ])

    return df_train_fe3, df_test_fe3, week_map, weather_map
```

**Step 2: Add soldout and payday feature processing**

```python
@app.cell
def _(df_train_fe3, df_test_fe3, pl):
    # soldout, paydayは既に数値なので確認
    df_train_fe4 = df_train_fe3.with_columns([
        pl.col("soldout").cast(pl.Int64),
        pl.col("payday").fill_null(0).cast(pl.Int64)
    ])

    df_test_fe4 = df_test_fe3.with_columns([
        pl.col("soldout").cast(pl.Int64),
        pl.col("payday").fill_null(0).cast(pl.Int64)
    ])

    return df_train_fe4, df_test_fe4
```

**Step 3: Verify encoding in marimo**

Run: `cd projects/bento9_opus_talk/notebooks && marimo edit bento9_analysis.py`
Expected: week_encoded, weather_encoded columns created with numeric values

**Step 4: Commit categorical encoding**

```bash
git add projects/bento9_opus_talk/notebooks/bento9_analysis.py
git commit -m "feat: add categorical feature encoding for week and weather

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: 欠損値処理（データリーク防止）

**Files:**
- Modify: `projects/bento9_opus_talk/notebooks/bento9_analysis.py`

**Step 1: Add missing value imputation cell**

CRITICAL: Use ONLY training data statistics for test data imputation

```python
@app.cell
def _(df_train_fe4, df_test_fe4, pl):
    # 欠損値補完（訓練データの統計量を使用）
    # データリーク防止: テストデータの補完には必ず訓練データの中央値を使用

    # 訓練データで中央値を計算
    train_kcal_median = df_train_fe4["kcal"].median()
    train_precipitation_median = df_train_fe4["precipitation"].median()
    train_temp_median = df_train_fe4["temperature"].median()

    # 訓練データの欠損値補完
    df_train_filled = df_train_fe4.with_columns([
        pl.col("kcal").fill_null(train_kcal_median),
        pl.col("precipitation").fill_null(train_precipitation_median),
        pl.col("temperature").fill_null(train_temp_median)
    ])

    # テストデータの欠損値補完（訓練データの統計量を使用）
    df_test_filled = df_test_fe4.with_columns([
        pl.col("kcal").fill_null(train_kcal_median),
        pl.col("precipitation").fill_null(train_precipitation_median),
        pl.col("temperature").fill_null(train_temp_median)
    ])

    return (df_train_filled, df_test_filled,
            train_kcal_median, train_precipitation_median, train_temp_median)
```

**Step 2: Add verification cell for no remaining nulls**

```python
@app.cell
def _(df_train_filled, df_test_filled, mo):
    # 欠損値処理確認
    train_nulls_after = df_train_filled.null_count().sum_horizontal()[0]
    test_nulls_after = df_test_filled.null_count().sum_horizontal()[0]

    mo.md(f"""
    ### 欠損値処理後の確認

    - 訓練データ欠損数: {train_nulls_after}
    - テストデータ欠損数: {test_nulls_after}
    """)
    return train_nulls_after, test_nulls_after
```

**Step 3: Verify null handling in marimo**

Run: `cd projects/bento9_opus_talk/notebooks && marimo edit bento9_analysis.py`
Expected: Both train_nulls_after and test_nulls_after are 0

**Step 4: Commit missing value handling**

```bash
git add projects/bento9_opus_talk/notebooks/bento9_analysis.py
git commit -m "feat: add missing value imputation with leak prevention

Use training data statistics for test data imputation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: 特徴量選択とデータセット準備

**Files:**
- Modify: `projects/bento9_opus_talk/notebooks/bento9_analysis.py`

**Step 1: Add feature selection cell**

```python
@app.cell
def _(df_train_filled, df_test_filled, pl):
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

    return X_train, y_train, X_test, feature_cols
```

**Step 2: Add dataset info display cell**

```python
@app.cell
def _(mo, X_train, y_train, X_test, feature_cols):
    mo.md(f"""
    ## 3. モデリング準備完了

    - 訓練データ: {X_train.shape[0]} samples × {X_train.shape[1]} features
    - テストデータ: {X_test.shape[0]} samples × {X_test.shape[1]} features
    - 特徴量リスト: {', '.join(feature_cols)}
    """)
    return
```

**Step 3: Verify datasets in marimo**

Run: `cd projects/bento9_opus_talk/notebooks && marimo edit bento9_analysis.py`
Expected: Feature count matches selected columns, shapes displayed correctly

**Step 4: Commit dataset preparation**

```bash
git add projects/bento9_opus_talk/notebooks/bento9_analysis.py
git commit -m "feat: add feature selection and dataset preparation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: ベースラインモデル（Ridge回帰）

**Files:**
- Modify: `projects/bento9_opus_talk/notebooks/bento9_analysis.py`

**Step 1: Add modeling section header**

```python
# ===== グループD: モデリング =====

@app.cell
def _(mo):
    mo.md("""
    ## 4. ベースラインモデル: Ridge回帰
    """)
    return
```

**Step 2: Add Ridge regression with TimeSeriesSplit**

```python
@app.cell
def _(X_train, y_train, Ridge, TimeSeriesSplit, mean_squared_error, np):
    # Ridge回帰 with TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    ridge_scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_tr, y_tr)
        y_pred = ridge_model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        ridge_scores.append(rmse)

    ridge_mean_rmse = np.mean(ridge_scores)
    ridge_std_rmse = np.std(ridge_scores)

    return ridge_model, ridge_scores, ridge_mean_rmse, ridge_std_rmse, tscv
```

**Step 3: Add Ridge results display**

```python
@app.cell
def _(mo, ridge_mean_rmse, ridge_std_rmse, ridge_scores):
    mo.md(f"""
    ### Ridge回帰結果

    - CV RMSE (mean ± std): {ridge_mean_rmse:.3f} ± {ridge_std_rmse:.3f}
    - Fold scores: {[f"{s:.3f}" for s in ridge_scores]}
    """)
    return
```

**Step 4: Verify Ridge model in marimo**

Run: `cd projects/bento9_opus_talk/notebooks && marimo edit bento9_analysis.py`
Expected: RMSE scores displayed, model trains without errors

**Step 5: Commit baseline model**

```bash
git add projects/bento9_opus_talk/notebooks/bento9_analysis.py
git commit -m "feat: add Ridge regression baseline with TimeSeriesSplit CV

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: LightGBMモデル構築

**Files:**
- Modify: `projects/bento9_opus_talk/notebooks/bento9_analysis.py`

**Step 1: Add LightGBM section**

```python
@app.cell
def _(mo):
    mo.md("""
    ## 5. LightGBMモデル
    """)
    return
```

**Step 2: Add LightGBM training with TimeSeriesSplit**

```python
@app.cell
def _(X_train, y_train, lgb, TimeSeriesSplit, mean_squared_error, np, tscv):
    # LightGBM with TimeSeriesSplit
    lgb_scores = []
    lgb_models = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "random_state": 42
        }

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        lgb_model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        y_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        lgb_scores.append(rmse)
        lgb_models.append(lgb_model)

    lgb_mean_rmse = np.mean(lgb_scores)
    lgb_std_rmse = np.std(lgb_scores)

    return lgb_models, lgb_scores, lgb_mean_rmse, lgb_std_rmse, lgb_params
```

**Step 3: Add LightGBM results display**

```python
@app.cell
def _(mo, lgb_mean_rmse, lgb_std_rmse, lgb_scores):
    mo.md(f"""
    ### LightGBM結果

    - CV RMSE (mean ± std): {lgb_mean_rmse:.3f} ± {lgb_std_rmse:.3f}
    - Fold scores: {[f"{s:.3f}" for s in lgb_scores]}
    """)
    return
```

**Step 4: Verify LightGBM in marimo**

Run: `cd projects/bento9_opus_talk/notebooks && marimo edit bento9_analysis.py`
Expected: LightGBM trains successfully, RMSE likely lower than Ridge

**Step 5: Commit LightGBM model**

```bash
git add projects/bento9_opus_talk/notebooks/bento9_analysis.py
git commit -m "feat: add LightGBM model with TimeSeriesSplit CV

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 11: モデル比較と特徴量重要度

**Files:**
- Modify: `projects/bento9_opus_talk/notebooks/bento9_analysis.py`

**Step 1: Add model comparison cell**

```python
@app.cell
def _(mo, ridge_mean_rmse, lgb_mean_rmse, pd, alt):
    # モデル比較
    comparison = pd.DataFrame({
        "Model": ["Ridge", "LightGBM"],
        "CV RMSE": [ridge_mean_rmse, lgb_mean_rmse]
    })

    chart_comparison = alt.Chart(comparison).mark_bar().encode(
        alt.X("Model:N", title="モデル"),
        alt.Y("CV RMSE:Q", title="RMSE"),
        color=alt.Color("Model:N", legend=None),
        tooltip=["Model", "CV RMSE"]
    ).properties(
        width=400,
        height=300,
        title="モデル比較（CV RMSE）"
    )

    mo.md("## 6. モデル比較")
    chart_comparison
    return comparison, chart_comparison
```

**Step 2: Add feature importance cell**

```python
@app.cell
def _(lgb_models, feature_cols, pd, alt, np):
    # 特徴量重要度（全foldの平均）
    importance_list = []
    for model in lgb_models:
        importance_list.append(model.feature_importance(importance_type="gain"))

    avg_importance = np.mean(importance_list, axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": avg_importance
    }).sort_values("Importance", ascending=False)

    chart_importance = alt.Chart(importance_df).mark_bar().encode(
        alt.X("Importance:Q", title="重要度"),
        alt.Y("Feature:N", title="特徴量", sort="-x"),
        tooltip=["Feature", "Importance"]
    ).properties(
        width=600,
        height=400,
        title="特徴量重要度（LightGBM）"
    )

    chart_importance
    return importance_df, chart_importance, avg_importance, importance_list
```

**Step 3: Verify comparison and importance in marimo**

Run: `cd projects/bento9_opus_talk/notebooks && marimo edit bento9_analysis.py`
Expected: Bar charts display, feature importance shows meaningful rankings

**Step 4: Commit model comparison**

```bash
git add projects/bento9_opus_talk/notebooks/bento9_analysis.py
git commit -m "feat: add model comparison and feature importance visualization

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 12: テストデータへの予測とsubmission生成

**Files:**
- Modify: `projects/bento9_opus_talk/notebooks/bento9_analysis.py`
- Create: `projects/bento9_opus_talk/submission.csv`

**Step 1: Add prediction on test data cell**

```python
@app.cell
def _(lgb_models, X_test, np):
    # テストデータへの予測（全foldの平均）
    test_predictions = []

    for model in lgb_models:
        pred = model.predict(X_test, num_iteration=model.best_iteration)
        test_predictions.append(pred)

    final_predictions = np.mean(test_predictions, axis=0)

    return test_predictions, final_predictions
```

**Step 2: Add submission file generation cell**

```python
@app.cell
def _(df_test, final_predictions, pd, Path):
    # submission.csv生成
    # 日付フォーマット: yyyy-m-d（1桁の日は0埋めしない）
    dates = df_test["datetime"].to_list()

    submission_df = pd.DataFrame({
        "datetime": dates,
        "y": final_predictions
    })

    # 保存
    submission_path = Path(__file__).parent.parent / "submission.csv"
    submission_df.to_csv(submission_path, index=False, header=False)

    return submission_df, submission_path
```

**Step 3: Add submission preview cell**

```python
@app.cell
def _(mo, submission_df):
    mo.md(f"""
    ## 7. 予測結果とSubmission

    - 予測完了: {len(submission_df)} 件
    - ファイル: submission.csv（ヘッダーなし、カンマ区切り）
    """)
    submission_df.head(10)
    return
```

**Step 4: Verify submission file in marimo**

Run: `cd projects/bento9_opus_talk/notebooks && marimo edit bento9_analysis.py`
Expected: submission.csv created with correct format (no header, date,prediction)

**Step 5: Verify submission file format manually**

Run: `head -5 projects/bento9_opus_talk/submission.csv`
Expected: Lines like "2014-10-1,123.45" (no header, single-digit days)

**Step 6: Commit prediction and submission**

```bash
git add projects/bento9_opus_talk/notebooks/bento9_analysis.py projects/bento9_opus_talk/submission.csv
git commit -m "feat: add test prediction and submission.csv generation

Date format: yyyy-m-d (single-digit days without zero padding)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 13: ドキュメント作成（オプション）

**Files:**
- Create: `projects/bento9_opus_talk/README.md`

**Step 1: Create project README**

```markdown
# bento9_opus_talk - お弁当販売数予測

## 概要

カフェフロアで販売されるお弁当の日次販売数を予測する機械学習プロジェクト。

## データ

- 訓練データ: `data/bento_train.csv`（2013-11-18 〜 2014-09-30）
- テストデータ: `data/bento_test.csv`（2014-10-01 〜 2014-11-28）
- 評価指標: RMSE（Root Mean Squared Error）

## 特徴量

- 日付特徴: year, month, day, weekday, is_holiday
- カテゴリカル: week_encoded, weather_encoded
- 数値: soldout, payday, kcal, precipitation, temperature

## モデル

1. Ridge回帰（ベースライン）
2. LightGBM（メインモデル、5-fold TimeSeriesSplit CV）

## 実行方法

```bash
# 依存パッケージインストール
cd projects/bento9_opus_talk
pip install -r requirements.txt

# marimoノートブック起動
cd notebooks
marimo edit bento9_analysis.py
```

## 結果

- 予測ファイル: `submission.csv`（ヘッダーなし、フォーマット: yyyy-m-d,prediction）

## 注意事項

- データリーク防止: テストデータの欠損値補完には訓練データの統計量を使用
- 日付フォーマット: 1桁の日は0埋めしない（例: 2014-10-1）
```

**Step 2: Save README**

Write to: `projects/bento9_opus_talk/README.md`

**Step 3: Commit documentation**

```bash
git add projects/bento9_opus_talk/README.md
git commit -m "docs: add project README with setup and usage instructions

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 14: 最終検証とクリーンアップ

**Files:**
- Verify all files

**Step 1: Run final marimo check**

Run: `cd projects/bento9_opus_talk/notebooks && marimo edit bento9_analysis.py`
Expected: All cells execute without errors, submission.csv generated

**Step 2: Verify submission format**

Run: `head -3 projects/bento9_opus_talk/submission.csv && wc -l projects/bento9_opus_talk/submission.csv`
Expected: 40 lines (matching test data rows), format "yyyy-m-d,prediction"

**Step 3: Check git status**

Run: `git status`
Expected: Clean working tree or only expected uncommitted files

**Step 4: Create final summary commit if needed**

```bash
git add .
git commit -m "chore: final cleanup for bento9_opus_talk project

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## 追加改善案（オプショナルタスク）

以下は時間があれば実施可能な改善項目です:

1. Optunaによるハイパーパラメータチューニング
2. Random ForestやGradient Boostingとのアンサンブル
3. ラグ特徴量（過去N日の売上平均など）
4. メニュー名のテキスト特徴量抽出
5. 交差検証での各foldの予測分布可視化

---

## 実装完了チェックリスト

- [ ] Task 1: プロジェクト基盤構築完了
- [ ] Task 2: データロード実装完了
- [ ] Task 3: 基本統計量・欠損値分析完了
- [ ] Task 4: 可視化実装完了
- [ ] Task 5: 日付・休日特徴量完了
- [ ] Task 6: カテゴリカルエンコーディング完了
- [ ] Task 7: 欠損値処理（リーク防止）完了
- [ ] Task 8: データセット準備完了
- [ ] Task 9: Ridgeベースライン完了
- [ ] Task 10: LightGBMモデル完了
- [ ] Task 11: モデル比較・重要度完了
- [ ] Task 12: Submission生成完了
- [ ] Task 13: ドキュメント作成完了
- [ ] Task 14: 最終検証完了

---

## 技術的注意事項

1. データリーク防止が最重要
   - テストデータの欠損値補完には必ず訓練データの統計量を使用

2. marimoのリアクティブ制約
   - 変数は1つのセルでのみ定義
   - UI要素の値は別セルで参照

3. TimeSeriesSplitの使用理由
   - 時系列データのため、未来のデータで過去を予測してはいけない

4. Submission形式の厳守
   - ヘッダーなし
   - フォーマット: yyyy-m-d,prediction
   - 1桁の日は0埋めしない
