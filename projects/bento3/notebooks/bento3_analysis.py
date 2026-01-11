import marimo

__generated_with = "0.19.1"
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
    import re
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    import warnings
    warnings.filterwarnings('ignore')
    return (mo, pl, pd, alt, Path, np, re,
            TimeSeriesSplit, LinearRegression, Ridge,
            RandomForestRegressor, mean_squared_error)


@app.cell
def _(Path, pl):
    # 訓練データロード
    data_path_train = Path(__file__).parent.parent / "data" / "bento_train.csv"
    df_train = pl.read_csv(data_path_train)
    return df_train,


@app.cell
def _(Path, pl):
    # テストデータロード
    data_path_test = Path(__file__).parent.parent / "data" / "bento_test.csv"
    df_test = pl.read_csv(data_path_test)
    return df_test,


@app.cell
def _(mo, df_train, df_test):
    # タイトルと概要
    mo.md(f"""
    # お弁当販売数予測 - シンプルアプローチで高精度を目指す

    訓練データ: {df_train.shape[0]} 行 × {df_train.shape[1]} 列 (2013-11-18〜2014-09-30)

    テストデータ: {df_test.shape[0]} 行 × {df_test.shape[1]} 列 (2014-10-01〜2014-11-28)

    評価指標: RMSE（Root Mean Squared Error）

    戦略: 過学習を避け、シンプルな特徴量で高精度を目指す
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
def _(df_train, pl):
    # Pandas変換（可視化用）
    df_train_pandas = df_train.to_pandas()
    df_train_pandas['datetime'] = pd.to_datetime(df_train_pandas['datetime'])
    return df_train_pandas,


@app.cell
def _(df_train_pandas, alt):
    # 時系列トレンドの可視化
    chart_timeseries = alt.Chart(df_train_pandas).mark_line(point=True).encode(
        x=alt.X('datetime:T', title='日付'),
        y=alt.Y('y:Q', title='販売数'),
        tooltip=['datetime:T', 'y:Q', 'name:N']
    ).properties(
        width=700,
        height=300,
        title='お弁当販売数の時系列トレンド'
    )
    chart_timeseries
    return chart_timeseries,


@app.cell
def _(df_train_pandas, alt):
    # 曜日別販売数
    chart_week = alt.Chart(df_train_pandas).mark_boxplot().encode(
        x=alt.X('week:N', title='曜日'),
        y=alt.Y('y:Q', title='販売数'),
        color='week:N'
    ).properties(
        width=500,
        height=300,
        title='曜日別販売数分布'
    )
    chart_week
    return chart_week,


@app.cell
def _(df_train_pandas, pd, alt):
    # 気温と販売数の関係
    df_temp = df_train_pandas.copy()
    df_temp['temperature'] = pd.to_numeric(df_temp['temperature'], errors='coerce')

    chart_temp = alt.Chart(df_temp).mark_circle().encode(
        x=alt.X('temperature:Q', title='気温（℃）'),
        y=alt.Y('y:Q', title='販売数'),
        color=alt.Color('month:O', title='月'),
        tooltip=['datetime:T', 'y:Q', 'temperature:Q', 'name:N']
    ).properties(
        width=600,
        height=400,
        title='気温と販売数の関係'
    )
    chart_temp
    return chart_temp,


# ===== グループC: 特徴量エンジニアリング =====

@app.cell
def _(mo):
    mo.md("""
    ## 2. 特徴量エンジニアリング（シンプル戦略）

    過学習を避けるため、重要な特徴量に絞り込みます:
    - 基本: year, month, dayofweek, temperature
    - フラグ: is_friday, is_special_menu, payday, event_flag, is_rainy
    - メニュー分類: curry, fry, hamburg, fish, meat
    """)
    return


@app.cell
def _(pd, np, re):
    def extract_menu_keywords(df, col_name='name'):
        """メニュー名からキーワードフラグを作成する"""
        keywords = {
            'curry': r'カレー',
            'fry': r'フライ|カツ|唐揚げ|天ぷら',
            'hamburg': r'ハンバーグ',
            'fish': r'魚|サバ|鮭|ぶり|ブリ|カレイ|マス|まぐろ|アジ|さんま',
            'meat': r'肉|牛|豚|鶏|チキン|ポーク|ビーフ',
        }

        for key, pattern in keywords.items():
            df[f'menu_{key}'] = df[col_name].apply(
                lambda x: 1 if re.search(pattern, str(x)) else 0
            )
        return df

    def create_features(df_pd, temp_median=None):
        """特徴量を作成する関数（シンプル版）"""
        df_fe = df_pd.copy()

        # datetime列をパース
        df_fe['datetime'] = pd.to_datetime(df_fe['datetime'])

        # 日付関連特徴量
        df_fe['year'] = df_fe['datetime'].dt.year
        df_fe['month'] = df_fe['datetime'].dt.month
        df_fe['dayofweek'] = df_fe['datetime'].dt.dayofweek

        # 月末フラグ（給料日前後で需要変動）
        df_fe['month_end'] = (df_fe['datetime'].dt.day > 20).astype(int)

        # 金曜フラグ
        df_fe['is_friday'] = (df_fe['dayofweek'] == 4).astype(int)

        # お楽しみメニューフラグ
        df_fe['is_special_menu'] = df_fe['remarks'].apply(
            lambda x: 1 if 'お楽しみ' in str(x) or 'スペシャル' in str(x) else 0
        )

        # メニューキーワード抽出
        df_fe = extract_menu_keywords(df_fe, 'name')

        # 欠損値処理（データリーク防止: 訓練データの統計量を使用）
        df_fe['temperature'] = pd.to_numeric(df_fe['temperature'], errors='coerce')
        fill_temp = temp_median if temp_median is not None else df_fe['temperature'].median()
        df_fe['temperature'] = df_fe['temperature'].fillna(fill_temp)

        df_fe['precipitation'] = df_fe['precipitation'].replace('--', '0')
        df_fe['precipitation'] = pd.to_numeric(df_fe['precipitation'], errors='coerce').fillna(0)

        df_fe['payday'] = df_fe['payday'].fillna(0).astype(float)
        df_fe['event_flag'] = df_fe['event'].notna().astype(int)

        # 悪天候フラグ（雨・雪）
        df_fe['is_rainy'] = df_fe['weather'].isin(['雨', '雪', '雷電']).astype(int)

        return df_fe

    return create_features, extract_menu_keywords


@app.cell
def _(df_train, df_test, create_features, pd):
    # 訓練データの統計量（中央値）を計算
    train_pd = df_train.to_pandas()
    train_temp_median = pd.to_numeric(train_pd['temperature'], errors='coerce').median()

    # 特徴量作成の実行（訓練データの統計量を使用）
    df_train_fe = create_features(train_pd, train_temp_median)
    df_test_fe = create_features(df_test.to_pandas(), train_temp_median)

    # 作成された特徴量のカラム一覧
    feature_cols = [
        'year', 'month', 'dayofweek', 'temperature',
        'is_friday', 'is_special_menu', 'payday', 'event_flag',
        'is_rainy', 'month_end',
        'menu_curry', 'menu_fry', 'menu_hamburg', 'menu_fish', 'menu_meat'
    ]

    df_train_fe.shape, df_test_fe.shape, len(feature_cols)
    return df_train_fe, df_test_fe, feature_cols, train_temp_median


# ===== グループD: ベースラインモデル =====

@app.cell
def _(mo):
    mo.md("""
    ## 3. ベースラインモデル - 線形回帰（4特徴量のみ）

    まずはシンプルに year, month, dayofweek, temperature だけで試します。
    """)
    return


@app.cell
def _(df_train_fe, LinearRegression, TimeSeriesSplit, mean_squared_error, np):
    # ベースライン: 基本4特徴量のみ
    baseline_features = ['year', 'month', 'dayofweek', 'temperature']
    X_baseline = df_train_fe[baseline_features]
    y_baseline = df_train_fe['y']

    # 時系列交差検証
    tscv = TimeSeriesSplit(n_splits=5)
    baseline_cv_scores = []

    for train_idx, val_idx in tscv.split(X_baseline):
        X_tr, X_val = X_baseline.iloc[train_idx], X_baseline.iloc[val_idx]
        y_tr, y_val = y_baseline.iloc[train_idx], y_baseline.iloc[val_idx]

        model_baseline = LinearRegression()
        model_baseline.fit(X_tr, y_tr)
        pred = model_baseline.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        baseline_cv_scores.append(rmse)

    baseline_overall_rmse = np.mean(baseline_cv_scores)
    baseline_overall_rmse, baseline_cv_scores
    return (baseline_features, X_baseline, y_baseline,
            baseline_overall_rmse, baseline_cv_scores)


@app.cell
def _(mo, baseline_overall_rmse, baseline_cv_scores):
    mo.md(f"""
    ### ベースラインモデル結果

    - Overall RMSE: {baseline_overall_rmse:.4f}
    - 各FoldのRMSE: {', '.join([f'{s:.2f}' for s in baseline_cv_scores])}

    期待: スコア13程度（ユーザー報告値）
    """)
    return


# ===== グループE: 改良モデル =====

@app.cell
def _(mo):
    mo.md("""
    ## 4. 改良モデル - 全特徴量を使用

    15個の特徴量を使い、複数モデルで比較します:
    - Ridge回帰（正則化）
    - RandomForest（max_depth制限）
    """)
    return


@app.cell
def _(df_train_fe, feature_cols, Ridge, RandomForestRegressor,
      TimeSeriesSplit, mean_squared_error, np):
    # 全特徴量を使用
    X = df_train_fe[feature_cols]
    y = df_train_fe['y']

    def train_ridge_cv(X_data, y_data):
        """Ridge回帰の交差検証"""
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_data):
            X_tr, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
            y_tr, y_val = y_data.iloc[train_idx], y_data.iloc[val_idx]

            model = Ridge(alpha=1.0)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            cv_scores.append(rmse)
        return cv_scores

    def train_rf_cv(X_data, y_data):
        """RandomForestの交差検証"""
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_data):
            X_tr, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
            y_tr, y_val = y_data.iloc[train_idx], y_data.iloc[val_idx]

            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42
            )
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            cv_scores.append(rmse)
        return cv_scores

    ridge_cv_scores = train_ridge_cv(X, y)
    ridge_overall_rmse = np.mean(ridge_cv_scores)

    rf_cv_scores = train_rf_cv(X, y)
    rf_overall_rmse = np.mean(rf_cv_scores)

    X, y, ridge_overall_rmse, ridge_cv_scores, rf_overall_rmse, rf_cv_scores
    return (X, y, ridge_overall_rmse, ridge_cv_scores,
            rf_overall_rmse, rf_cv_scores)


@app.cell
def _(mo, ridge_overall_rmse, ridge_cv_scores, rf_overall_rmse, rf_cv_scores):
    mo.md(f"""
    ### 改良モデル比較

    Ridge回帰:
    - Overall RMSE: {ridge_overall_rmse:.4f}
    - 各FoldのRMSE: {', '.join([f'{s:.2f}' for s in ridge_cv_scores])}

    RandomForest:
    - Overall RMSE: {rf_overall_rmse:.4f}
    - 各FoldのRMSE: {', '.join([f'{s:.2f}' for s in rf_cv_scores])}
    """)
    return


# ===== グループF: 最終モデルと予測 =====

@app.cell
def _(mo):
    mo.md("""
    ## 5. 最終モデルで予測

    最も精度が高かったモデルで全データを学習し、テストデータを予測します。
    """)
    return


@app.cell
def _(X, y, Ridge, RandomForestRegressor, ridge_overall_rmse, rf_overall_rmse):
    # 最良モデルの選択
    if ridge_overall_rmse <= rf_overall_rmse:
        final_model = Ridge(alpha=1.0)
        best_model_name = "Ridge回帰"
        best_rmse = ridge_overall_rmse
    else:
        final_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
        best_model_name = "RandomForest"
        best_rmse = rf_overall_rmse

    # 全データで学習
    final_model.fit(X, y)

    best_model_name, best_rmse
    return final_model, best_model_name, best_rmse


@app.cell
def _(mo, best_model_name, best_rmse):
    mo.md(f"""
    ### 最終モデル: {best_model_name}

    - Cross Validation RMSE: {best_rmse:.4f}

    目標スコア10に対する評価: {'達成!' if best_rmse <= 10 else '未達成'}
    """)
    return


@app.cell
def _(df_test_fe, feature_cols, final_model):
    # テストデータで予測
    X_test = df_test_fe[feature_cols]
    test_predictions = final_model.predict(X_test)

    # 予測結果を確認
    test_predictions[:10]
    return X_test, test_predictions


# ===== グループG: 提出ファイル生成 =====

@app.cell
def _(mo):
    mo.md("""
    ## 6. 提出ファイル生成

    形式: ヘッダーなしCSV、日付はyyyy-m-d形式（ゼロ埋めなし）
    """)
    return


@app.cell
def _(df_test_fe, test_predictions, Path, pd):
    # 提出ファイル作成
    submission = pd.DataFrame({
        'datetime': df_test_fe['datetime'],
        'y': test_predictions
    })

    # 日付フォーマット変換（yyyy-m-d形式、ゼロ埋めなし）
    submission['datetime'] = submission['datetime'].apply(
        lambda x: f"{x.year}-{x.month}-{x.day}"
    )

    # ヘッダーなしで保存
    submission_path = Path(__file__).parent.parent / "submission.csv"
    submission.to_csv(submission_path, index=False, header=False)

    submission.head(10)
    return submission, submission_path


@app.cell
def _(mo, submission_path):
    mo.md(f"""
    ### 提出ファイル保存完了

    ファイル: {submission_path}

    このファイルをコンペティションサイトに提出してください。
    """)
    return


if __name__ == "__main__":
    app.run()
