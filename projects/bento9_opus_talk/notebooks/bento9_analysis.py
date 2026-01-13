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


if __name__ == "__main__":
    app.run()
