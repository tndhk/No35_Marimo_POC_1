import marimo

__generated_with = "0.19.0"
app = marimo.App()


# ===== グループA: 初期化 =====

@app.cell
def imports():
    """全モジュールのインポート"""
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


@app.cell
def load_train_data(Path, pl):
    """訓練データの読み込み"""
    # 訓練データロード
    data_path_train = Path(__file__).parent.parent / "data" / "bento_train.csv"
    df_train = pl.read_csv(data_path_train)
    return df_train, data_path_train


@app.cell
def load_test_data(Path, pl):
    """テストデータの読み込み"""
    # テストデータロード
    data_path_test = Path(__file__).parent.parent / "data" / "bento_test.csv"
    df_test = pl.read_csv(data_path_test)
    return df_test, data_path_test


@app.cell
def title(mo, df_train, df_test):
    """プロジェクト概要の表示"""
    # タイトルと概要
    mo.md(f"""
    # お弁当販売数予測 - bento9_opus_talk

    訓練データ: {df_train.shape[0]} 行 × {df_train.shape[1]} 列

    テストデータ: {df_test.shape[0]} 行 × {df_test.shape[1]} 列

    評価指標: RMSE（Root Mean Squared Error）
    """)
    return


# ===== グループB: 基本EDA =====

@app.cell
def eda_header(mo):
    """基本統計とデータ理解のセクションヘッダー"""
    mo.md("""
    ## 1. 基本統計とデータ理解
    """)
    return


@app.cell
def basic_stats(df_train):
    """訓練データの基本統計量を表示"""
    # 基本統計量
    train_stats = df_train.describe()
    train_stats
    return train_stats,


@app.cell
def null_analysis(df_train, pl, mo):
    """訓練データの欠損値を分析"""
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


@app.cell
def target_distribution(df_train, alt):
    """目的変数yの分布をヒストグラムで可視化"""
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


@app.cell
def week_analysis(df_train, alt):
    """曜日別販売数の分布をボックスプロットで可視化"""
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


# ===== グループC: 特徴量エンジニアリング =====

@app.cell
def feature_engineering_header(mo):
    """特徴量エンジニアリングのセクションヘッダー"""
    mo.md("""
    ## 2. 特徴量エンジニアリング
    """)
    return


@app.cell
def date_features(df_train, df_test, pl, jpholiday, datetime):
    """日付から年月日・曜日を抽出"""
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


@app.cell
def holiday_features(df_train_fe, df_test_fe, pl, jpholiday, datetime):
    """祝日フラグを追加"""
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


@app.cell
def categorical_encoding(df_train_fe2, df_test_fe2, pl):
    """曜日と天気をカテゴリカルエンコーディング"""
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


@app.cell
def numeric_features(df_train_fe3, df_test_fe3, pl):
    """soldoutとpaydayを数値型に変換"""
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


if __name__ == "__main__":
    app.run()
