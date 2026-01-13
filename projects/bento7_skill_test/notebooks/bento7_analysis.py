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
    # お弁当販売数予測 - bento7_skill_test

    訓練データ: {df_train.shape[0]} 行 × {df_train.shape[1]} 列

    テストデータ: {df_test.shape[0]} 行 × {df_test.shape[1]} 列

    評価指標: RMSE（Root Mean Squared Error）
    """)
    return


# ===== グループB: 基本EDA =====

@app.cell
def _(mo):
    mo.md("""
    ## 1. 基本統計とデータ理解
    """)
    return


@app.cell
def _(df_train):
    # 基本統計量
    train_stats = df_train.describe()
    train_stats
    return train_stats,


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


@app.cell
def _(df_train, mo):
    # 目的変数yの統計
    y_mean = df_train["y"].mean()
    y_median = df_train["y"].median()
    y_std = df_train["y"].std()
    y_min = df_train["y"].min()
    y_max = df_train["y"].max()

    mo.md(f"""
    ### 目的変数（販売数）の統計
    - 平均: {y_mean:.2f} 個
    - 中央値: {y_median:.2f} 個
    - 標準偏差: {y_std:.2f}
    - 範囲: {y_min} 〜 {y_max} 個
    """)
    return


@app.cell
def _(df_train, pd):
    # Polars → Pandas変換（可視化用）
    df_train_pandas = df_train.to_pandas()
    return df_train_pandas,


@app.cell
def _(df_train_pandas, alt):
    # 目的変数yの分布（ヒストグラム）
    chart_y_hist = alt.Chart(df_train_pandas).mark_bar().encode(
        alt.X('y:Q', bin=alt.Bin(maxbins=30), title='販売数'),
        alt.Y('count():Q', title='頻度'),
        tooltip=['count():Q']
    ).properties(
        width=600,
        height=300,
        title='販売数の分布'
    )
    chart_y_hist
    return chart_y_hist,


@app.cell
def _(df_train, pl):
    # 曜日別集計
    week_stats = (
        df_train.group_by("week")
        .agg([
            pl.col("y").mean().alias("平均販売数"),
            pl.col("y").median().alias("中央値"),
            pl.col("y").std().alias("標準偏差"),
            pl.count().alias("データ数")
        ])
        .sort("平均販売数", descending=True)
    )
    week_stats
    return week_stats,


@app.cell
def _(df_train, pl):
    # 天気別集計
    weather_stats = (
        df_train.group_by("weather")
        .agg([
            pl.col("y").mean().alias("平均販売数"),
            pl.col("y").median().alias("中央値"),
            pl.count().alias("データ数")
        ])
        .sort("平均販売数", descending=True)
    )
    weather_stats
    return weather_stats,


@app.cell
def _(df_train_pandas, alt, pd):
    # 時系列プロット
    df_ts = df_train_pandas.copy()
    df_ts['datetime'] = pd.to_datetime(df_ts['datetime'])

    chart_ts = alt.Chart(df_ts).mark_line(point=True).encode(
        alt.X('datetime:T', title='日付'),
        alt.Y('y:Q', title='販売数'),
        tooltip=['datetime:T', 'y:Q', 'week:N', 'weather:N']
    ).properties(
        width=800,
        height=300,
        title='販売数の時系列推移'
    )
    chart_ts
    return chart_ts, df_ts


# ===== グループC: 特徴量エンジニアリング =====

@app.cell
def _(mo):
    mo.md("""
    ## 2. 特徴量エンジニアリング
    """)
    return


@app.cell
def _(pd, np, re, jpholiday, datetime):
    def extract_menu_keywords(df_pd, column_name):
        """メニュー名からキーワードを抽出"""
        keywords = ['カレー', '唐揚', 'ハンバーグ', '魚', '肉', 'フライ']
        for keyword in keywords:
            col_name = f'menu_{keyword}'
            df_pd[col_name] = df_pd[column_name].str.contains(keyword, na=False).astype(int)
        return df_pd

    def create_features(df_pd, kcal_median=None, temp_median=None):
        """特徴量を作成する関数（データリーク防止）"""
        df_fe = df_pd.copy()

        # datetime列をパース
        df_fe['datetime'] = pd.to_datetime(df_fe['datetime'])

        # 日付関連特徴量
        df_fe['year'] = df_fe['datetime'].dt.year
        df_fe['month'] = df_fe['datetime'].dt.month
        df_fe['day'] = df_fe['datetime'].dt.day
        df_fe['dayofweek'] = df_fe['datetime'].dt.dayofweek
        df_fe['weekofyear'] = df_fe['datetime'].dt.isocalendar().week.astype(int)

        # 曜日のワンホットエンコーディング
        week_dummies = pd.get_dummies(df_fe['week'], prefix='week')
        df_fe = pd.concat([df_fe, week_dummies], axis=1)

        # 天気のワンホットエンコーディング
        weather_categories = ['快晴', '晴れ', '曇', '薄曇', '雨', '雪', '雷電']
        weather_categorical = pd.Categorical(df_fe['weather'], categories=weather_categories)
        weather_dummies = pd.get_dummies(weather_categorical, prefix='weather')
        df_fe = pd.concat([df_fe, weather_dummies], axis=1)

        # メニューキーワード抽出
        df_fe = extract_menu_keywords(df_fe, 'name')

        # 欠損値処理（データリーク防止: trainの統計量を使用）
        fill_kcal = kcal_median if kcal_median is not None else df_fe['kcal'].median()
        df_fe['kcal'] = df_fe['kcal'].fillna(fill_kcal)

        df_fe['precipitation'] = df_fe['precipitation'].replace('--', '0')
        df_fe['precipitation'] = pd.to_numeric(df_fe['precipitation'], errors='coerce').fillna(0)

        df_fe['temperature'] = pd.to_numeric(df_fe['temperature'], errors='coerce')
        fill_temp = temp_median if temp_median is not None else df_fe['temperature'].median()
        df_fe['temperature'] = df_fe['temperature'].fillna(fill_temp)

        df_fe['soldout'] = df_fe['soldout'].fillna(0).astype(int)
        df_fe['payday'] = df_fe['payday'].fillna(0).astype(float)
        df_fe['event_flag'] = df_fe['event'].notna().astype(int)
        df_fe['remarks_flag'] = df_fe['remarks'].notna().astype(int)

        # 派生特徴量
        df_fe['is_bad_weather'] = df_fe['weather'].isin(['雨', '雪', '雷電']).astype(int)
        df_fe['is_friday'] = (df_fe['dayofweek'] == 4).astype(int)

        # 祝日判定
        df_fe['is_holiday'] = df_fe['datetime'].apply(
            lambda x: 1 if jpholiday.is_holiday(x) else 0
        )

        # 翌日が祝日かどうか
        df_fe['is_next_holiday'] = df_fe['datetime'].apply(
            lambda x: 1 if jpholiday.is_holiday(x + pd.Timedelta(days=1)) else 0
        )

        # ラグ特徴量（yが存在する場合のみ）
        if 'y' in df_fe.columns:
            df_fe['lag_1'] = df_fe['y'].shift(1)
            df_fe['lag_5'] = df_fe['y'].shift(5)
            df_fe['lag_7'] = df_fe['y'].shift(7)
            df_fe['rolling_mean_5'] = df_fe['y'].shift(1).rolling(window=5, min_periods=1).mean()
            df_fe['rolling_mean_10'] = df_fe['y'].shift(1).rolling(window=10, min_periods=1).mean()
            df_fe['rolling_std_5'] = df_fe['y'].shift(1).rolling(window=5, min_periods=1).std()
            df_fe['diff_1'] = df_fe['y'].diff(1)
        else:
            # テストデータ（yなし）では、NaNで埋める
            df_fe['lag_1'] = np.nan
            df_fe['lag_5'] = np.nan
            df_fe['lag_7'] = np.nan
            df_fe['rolling_mean_5'] = np.nan
            df_fe['rolling_mean_10'] = np.nan
            df_fe['rolling_std_5'] = np.nan
            df_fe['diff_1'] = np.nan

        return df_fe

    return create_features, extract_menu_keywords


@app.cell
def _(df_train_pandas, create_features):
    # 訓練データの統計量を計算（データリーク防止）
    train_kcal_median = df_train_pandas['kcal'].median()
    train_temp_median = df_train_pandas['temperature'].median()

    # 訓練データに特徴量を作成
    df_train_fe = create_features(
        df_train_pandas,
        kcal_median=train_kcal_median,
        temp_median=train_temp_median
    )

    df_train_fe
    return df_train_fe, train_kcal_median, train_temp_median


@app.cell
def _(df_test, create_features, train_kcal_median, train_temp_median):
    # テストデータに特徴量を作成（trainの統計量を使用）
    df_test_pandas = df_test.to_pandas()
    df_test_fe = create_features(
        df_test_pandas,
        kcal_median=train_kcal_median,
        temp_median=train_temp_median
    )

    df_test_fe
    return df_test_fe, df_test_pandas


@app.cell
def _(df_train_fe):
    # 特徴量カラムの定義
    exclude_cols = ['datetime', 'y', 'week', 'weather', 'name', 'event', 'remarks']
    feature_cols = [col for col in df_train_fe.columns if col not in exclude_cols]

    X_train = df_train_fe[feature_cols]
    y_train = df_train_fe['y']

    print(f"特徴量数: {len(feature_cols)}")
    print(f"訓練データ形状: {X_train.shape}")
    return X_train, y_train, feature_cols


# ===== グループD: モデル構築 =====

@app.cell
def _(mo):
    mo.md("""
    ## 3. モデル構築と交差検証
    """)
    return


@app.cell
def _(mo):
    # Optuna最適化のON/OFFスイッチ
    optuna_switch = mo.ui.switch(label="Optuna最適化を実行", value=False)
    optuna_switch
    return optuna_switch,


@app.cell
def _(X_train, y_train, TimeSeriesSplit, lgb, RandomForestRegressor,
      Ridge, GradientBoostingRegressor, mean_squared_error, np):
    # TimeSeriesSplit による交差検証
    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
        'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100),
        'Ridge': Ridge(alpha=1.0),
        'GradientBoosting': GradientBoostingRegressor(random_state=42, n_estimators=100)
    }

    cv_results = {}

    for name, model in models.items():
        rmse_scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmse_scores.append(rmse)

        cv_results[name] = {
            'mean_rmse': np.mean(rmse_scores),
            'std_rmse': np.std(rmse_scores)
        }

    cv_results
    return cv_results, models, tscv


@app.cell
def _(cv_results, pd):
    # 交差検証結果の可視化
    cv_df = pd.DataFrame(cv_results).T
    cv_df = cv_df.sort_values('mean_rmse')
    cv_df
    return cv_df,


@app.cell
def _(X_train, y_train, optuna_switch, optuna, lgb, TimeSeriesSplit,
      mean_squared_error, np):
    # Optuna最適化（スイッチがONの場合）
    best_params_lgb = None

    if optuna_switch.value:
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'random_state': 42,
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }

            tscv_opt = TimeSeriesSplit(n_splits=5)
            rmse_scores = []

            for train_idx, val_idx in tscv_opt.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                rmse_scores.append(rmse)

            return np.mean(rmse_scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, show_progress_bar=True)

        best_params_lgb = study.best_params
        print(f"Best RMSE: {study.best_value:.4f}")
        print(f"Best params: {best_params_lgb}")
    else:
        print("Optuna最適化はスキップされました")

    best_params_lgb
    return best_params_lgb,


@app.cell
def _(X_train, y_train, best_params_lgb, lgb):
    # 最終モデルの学習
    if best_params_lgb is not None:
        final_model = lgb.LGBMRegressor(**best_params_lgb, random_state=42, verbose=-1)
    else:
        # デフォルトパラメータ
        final_model = lgb.LGBMRegressor(random_state=42, verbose=-1, n_estimators=100)

    final_model.fit(X_train, y_train)
    final_model
    return final_model,


# ===== グループE: 予測と提出 =====

@app.cell
def _(mo):
    mo.md("""
    ## 4. テストデータへの予測と提出ファイル生成
    """)
    return


@app.cell
def _(np):
    def predict_recursive(model, df_train_fe, df_test_fe, feature_cols):
        """
        テストデータを1日ずつ再帰的に予測
        訓練データの最後のy値をバッファとして使い、
        テストデータは予測値を次のラグ特徴量として使用
        """
        # 訓練データ末尾のyを取得（ラグ計算用バッファ）
        y_buffer = df_train_fe['y'].tail(10).tolist()

        predictions = []

        for idx, row_data in df_test_fe.iterrows():
            # 行をコピーして処理
            row = row_data.copy()

            # ラグ特徴量を動的に計算
            if len(y_buffer) > 0:
                row['lag_1'] = y_buffer[-1]
            if len(y_buffer) >= 5:
                row['lag_5'] = y_buffer[-5]
            else:
                row['lag_5'] = np.mean(y_buffer) if y_buffer else 0
            if len(y_buffer) >= 7:
                row['lag_7'] = y_buffer[-7]
            else:
                row['lag_7'] = np.mean(y_buffer) if y_buffer else 0

            # 移動平均と標準偏差
            row['rolling_mean_5'] = np.mean(y_buffer[-5:]) if y_buffer else 0
            row['rolling_mean_10'] = np.mean(y_buffer[-10:]) if y_buffer else 0
            row['rolling_std_5'] = np.std(y_buffer[-5:]) if len(y_buffer) >= 2 else 0

            # 差分特徴量
            if len(y_buffer) >= 2:
                row['diff_1'] = y_buffer[-1] - y_buffer[-2]
            else:
                row['diff_1'] = 0

            # 予測を実行
            X_single = row[feature_cols].values.reshape(1, -1)
            pred = model.predict(X_single)[0]
            pred = max(0, pred)  # 負の値をクリップ

            predictions.append(pred)
            y_buffer.append(pred)

        return predictions

    return predict_recursive,


@app.cell
def _(df_test_fe, feature_cols, df_train_fe, final_model, predict_recursive):
    # テストデータへの予測（再帰的予測）
    test_predictions = predict_recursive(final_model, df_train_fe, df_test_fe, feature_cols)

    # 予測結果の確認
    print(f"テスト予測数: {len(test_predictions)}")
    print(f"予測値の範囲: {min(test_predictions):.2f} - {max(test_predictions):.2f}")

    test_predictions
    return test_predictions,


@app.cell
def _(df_test_fe, test_predictions, Path, pd, mo):
    # 提出ファイル作成
    # 日付フォーマットの調整（yyyy-m-d形式、0埋めなし）
    submission_dates = [
        f"{dt.year}-{dt.month}-{dt.day}"
        for dt in df_test_fe['datetime']
    ]

    # 提出用DataFrameの作成
    submission_df = pd.DataFrame({
        'datetime': submission_dates,
        'y': [int(round(pred)) for pred in test_predictions]
    })

    # CSVファイルとして保存（ヘッダーなし）
    output_path = Path(__file__).parent.parent / "data" / "submission.csv"
    submission_df.to_csv(output_path, index=False, header=False)

    mo.md(f"""
    ### 提出ファイル生成完了

    提出ファイルを作成しました: `{output_path}`

    - 形式: yyyy-m-d（0埋めなし）
    - ヘッダーなし
    - 行数: {len(submission_df)}
    """)
    return submission_dates, submission_df, output_path


@app.cell
def _(submission_df):
    # 提出ファイルのプレビュー
    submission_df.head(10)
    return


if __name__ == "__main__":
    app.run()
