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
        pl.col("payday").fill_null(value=0).cast(pl.Int64)
    ])

    df_test_fe4 = df_test_fe3.with_columns([
        pl.col("soldout").cast(pl.Int64),
        pl.col("payday").fill_null(value=0).cast(pl.Int64)
    ])

    return df_train_fe4, df_test_fe4


@app.cell
def missing_value_imputation(df_train_fe4, df_test_fe4, pl):
    """訓練データの統計量を使用して欠損値を補完（データリーク防止）"""
    # 欠損値補完（訓練データの統計量を使用）
    # データリーク防止: テストデータの補完には必ず訓練データの中央値を使用

    # 訓練データで中央値を計算
    train_kcal_median = df_train_fe4["kcal"].median()
    train_precipitation_median = df_train_fe4["precipitation"].median()
    train_temp_median = df_train_fe4["temperature"].median()

    # 訓練データの欠損値補完
    df_train_filled = df_train_fe4.with_columns([
        pl.col("kcal").fill_null(value=train_kcal_median),
        pl.col("precipitation").fill_null(value=train_precipitation_median),
        pl.col("temperature").fill_null(value=train_temp_median)
    ])

    # テストデータの欠損値補完（訓練データの統計量を使用）
    df_test_filled = df_test_fe4.with_columns([
        pl.col("kcal").fill_null(value=train_kcal_median),
        pl.col("precipitation").fill_null(value=train_precipitation_median),
        pl.col("temperature").fill_null(value=train_temp_median)
    ])

    return (df_train_filled, df_test_filled,
            train_kcal_median, train_precipitation_median, train_temp_median)


@app.cell
def verify_no_nulls(df_train_filled, df_test_filled, mo):
    """欠損値処理の確認"""
    # 欠損値処理確認
    train_nulls_after = df_train_filled.null_count().sum_horizontal()[0]
    test_nulls_after = df_test_filled.null_count().sum_horizontal()[0]

    mo.md(f"""
    ### 欠損値処理後の確認

    - 訓練データ欠損数: {train_nulls_after}
    - テストデータ欠損数: {test_nulls_after}
    """)
    return train_nulls_after, test_nulls_after


@app.cell
def feature_selection(df_train_filled, df_test_filled, pl):
    """モデリング用の特徴量を選択"""
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


@app.cell
def dataset_info(mo, X_train, y_train, X_test, feature_cols):
    """データセット準備完了の確認"""
    mo.md(f"""
    ## 3. モデリング準備完了

    - 訓練データ: {X_train.shape[0]} samples × {X_train.shape[1]} features
    - テストデータ: {X_test.shape[0]} samples × {X_test.shape[1]} features
    - 特徴量リスト: {', '.join(feature_cols)}
    """)
    return


# ===== グループD: モデリング =====

@app.cell
def modeling_header(mo):
    """モデリングのセクションヘッダー"""
    mo.md("""
    ## 4. モデリング - Ridge回帰
    """)
    return


@app.cell
def ridge_model(X_train, y_train, Ridge, TimeSeriesSplit, mean_squared_error, np):
    """Ridge回帰モデルの訓練と時系列交差検証"""
    # Ridge回帰モデル
    tscv = TimeSeriesSplit(n_splits=5)
    ridge = Ridge(alpha=1.0)

    ridge_cv_scores = []
    for _train_idx, _val_idx in tscv.split(X_train):
        _X_tr, _X_val = X_train.iloc[_train_idx], X_train.iloc[_val_idx]
        _y_tr, _y_val = y_train.iloc[_train_idx], y_train.iloc[_val_idx]

        ridge.fit(_X_tr, _y_tr)
        _y_pred = ridge.predict(_X_val)
        _rmse = np.sqrt(mean_squared_error(_y_val, _y_pred))
        ridge_cv_scores.append(_rmse)

    # 全訓練データで再学習
    ridge.fit(X_train, y_train)

    return ridge, ridge_cv_scores, tscv


@app.cell
def ridge_results(mo, ridge_cv_scores, np):
    """Ridge回帰の交差検証結果を表示"""
    ridge_mean_rmse = np.mean(ridge_cv_scores)
    ridge_std_rmse = np.std(ridge_cv_scores)

    mo.md(f"""
    ### Ridge回帰の交差検証結果

    - CV RMSE (平均): {ridge_mean_rmse:.2f}
    - CV RMSE (標準偏差): {ridge_std_rmse:.2f}
    - 各Fold: {[f"{s:.2f}" for s in ridge_cv_scores]}
    """)
    return ridge_mean_rmse, ridge_std_rmse


@app.cell
def lightgbm_header(mo):
    """LightGBMのセクションヘッダー"""
    mo.md("""
    ## 5. モデリング - LightGBM
    """)
    return


@app.cell
def lightgbm_model(X_train, y_train, X_test, lgb, TimeSeriesSplit, mean_squared_error, np, optuna):
    """LightGBMモデルの訓練とOptunaによるハイパーパラメータ最適化"""

    # Optunaによるハイパーパラメータチューニング
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

        for _train_idx, _val_idx in tscv.split(X_train):
            _X_tr, _X_val = X_train.iloc[_train_idx], X_train.iloc[_val_idx]
            _y_tr, _y_val = y_train.iloc[_train_idx], y_train.iloc[_val_idx]

            train_data = lgb.Dataset(_X_tr, label=_y_tr)
            val_data = lgb.Dataset(_X_val, label=_y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )

            _y_pred = model.predict(_X_val, num_iteration=model.best_iteration)
            _rmse = np.sqrt(mean_squared_error(_y_val, _y_pred))
            cv_scores.append(_rmse)

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

    # 最良パラメータで再学習
    tscv = TimeSeriesSplit(n_splits=5)
    lgb_cv_scores = []

    for _train_idx, _val_idx in tscv.split(X_train):
        _X_tr, _X_val = X_train.iloc[_train_idx], X_train.iloc[_val_idx]
        _y_tr, _y_val = y_train.iloc[_train_idx], y_train.iloc[_val_idx]

        train_data = lgb.Dataset(_X_tr, label=_y_tr)
        val_data = lgb.Dataset(_X_val, label=_y_val, reference=train_data)

        model = lgb.train(
            best_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        _y_pred = model.predict(_X_val, num_iteration=model.best_iteration)
        _rmse = np.sqrt(mean_squared_error(_y_val, _y_pred))
        lgb_cv_scores.append(_rmse)

    # 全訓練データで最終モデルを訓練
    train_data_full = lgb.Dataset(X_train, label=y_train)
    lgb_model = lgb.train(
        best_params,
        train_data_full,
        num_boost_round=1000
    )

    return lgb_model, lgb_cv_scores, best_params, study


@app.cell
def lightgbm_results(mo, lgb_cv_scores, best_params, np):
    """LightGBMの交差検証結果を表示"""
    lgb_mean_rmse = np.mean(lgb_cv_scores)
    lgb_std_rmse = np.std(lgb_cv_scores)

    params_str = "\n".join([f"  - {k}: {v}" for k, v in best_params.items() if k not in ['objective', 'metric', 'verbosity', 'boosting_type']])

    mo.md(f"""
    ### LightGBMの交差検証結果

    - CV RMSE (平均): {lgb_mean_rmse:.2f}
    - CV RMSE (標準偏差): {lgb_std_rmse:.2f}
    - 各Fold: {[f"{s:.2f}" for s in lgb_cv_scores]}

    最適化されたハイパーパラメータ:
{params_str}
    """)
    return lgb_mean_rmse, lgb_std_rmse


@app.cell
def model_comparison(mo, ridge_mean_rmse, lgb_mean_rmse, pl, alt):
    """モデル性能比較の可視化"""
    mo.md("""
    ## 6. モデル比較
    """)

    # モデル比較用データフレーム
    comparison_df = pl.DataFrame({
        "Model": ["Ridge", "LightGBM"],
        "CV RMSE": [ridge_mean_rmse, lgb_mean_rmse]
    })

    # 比較チャート
    comparison_chart = alt.Chart(comparison_df.to_pandas()).mark_bar().encode(
        alt.X("Model:N", title="モデル"),
        alt.Y("CV RMSE:Q", title="RMSE"),
        alt.Color("Model:N", legend=None),
        tooltip=["Model", alt.Tooltip("CV RMSE:Q", format=".2f")]
    ).properties(
        width=400,
        height=300,
        title="モデル性能比較（CV RMSE）"
    )

    comparison_chart
    return comparison_df, comparison_chart


@app.cell
def feature_importance(lgb_model, feature_cols, pl, alt):
    """LightGBMの特徴量重要度を可視化"""
    # 特徴量重要度
    importance = lgb_model.feature_importance(importance_type='gain')
    importance_df = pl.DataFrame({
        "Feature": feature_cols,
        "Importance": importance
    }).sort("Importance", descending=True)

    # 重要度チャート
    importance_chart = alt.Chart(importance_df.to_pandas()).mark_bar().encode(
        alt.X("Importance:Q", title="重要度"),
        alt.Y("Feature:N", title="特徴量", sort="-x"),
        alt.Color("Importance:Q", legend=None, scale=alt.Scale(scheme="viridis")),
        tooltip=["Feature", alt.Tooltip("Importance:Q", format=".2f")]
    ).properties(
        width=500,
        height=400,
        title="LightGBM特徴量重要度"
    )

    importance_chart
    return importance_df, importance_chart


@app.cell
def test_prediction(lgb_model, X_test, np, mo):
    """LightGBMモデルでテストデータの予測を実行"""
    mo.md("""
    ## 7. 予測とSubmission生成
    """)

    # LightGBMで予測
    y_pred = lgb_model.predict(X_test)

    # 負の値を0にクリップ（販売数は負にならない）
    y_pred = np.maximum(y_pred, 0)

    return y_pred,


@app.cell
def submission_generation(df_test_filled, y_pred, Path, pl, pd):
    """submission.csvを生成（yyyy-m-d形式、ヘッダーなし）"""
    # submission.csv生成
    # 日付フォーマット: yyyy-m-d（1桁の日は0埋めしない）
    dates = df_test_filled["datetime"].to_list()

    # y値を整数に丸める
    predictions_int = [int(round(p)) for p in y_pred]

    submission_df = pd.DataFrame({
        "datetime": dates,
        "y": predictions_int
    })

    # 保存（ヘッダーなし）
    submission_path = Path(__file__).parent.parent / "submission.csv"
    submission_df.to_csv(submission_path, index=False, header=False)

    return submission_df, submission_path


@app.cell
def submission_preview(mo, submission_df, submission_path):
    """Submission内容のプレビュー"""
    mo.md(f"""
    ### Submission生成完了

    ファイルパス: `{submission_path}`

    最初の10行:
    """)
    submission_df.head(10)
    return


if __name__ == "__main__":
    app.run()
