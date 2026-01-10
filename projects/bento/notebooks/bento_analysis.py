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
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    import warnings
    warnings.filterwarnings('ignore')
    return (mo, pl, pd, alt, Path, np,
            train_test_split, LinearRegression, Ridge,
            DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor,
            mean_squared_error)


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
    # お弁当販売数予測 - EDA & モデル構築

    訓練データ: {df_train.shape[0]} 行 × {df_train.shape[1]} 列 (2013-11-18〜2014-09-30)

    テストデータ: {df_test.shape[0]} 行 × {df_test.shape[1]} 列 (2014-10-01〜2014-11-28)

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
def _(df_train, pl, mo):
    # 売り切れの影響
    soldout_stats = (
        df_train.group_by("soldout")
        .agg([
            pl.col("y").mean().alias("平均販売数"),
            pl.count().alias("データ数")
        ])
    )
    mo.md("### 売り切れの影響")
    soldout_stats
    return soldout_stats,


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


# ===== グループC: 可視化 =====

@app.cell
def _(mo):
    mo.md("""
    ## 2. データ可視化
    """)
    return


@app.cell
def _(df_train, df_test, pd):
    # Pandas変換（可視化用）
    df_train_pandas = df_train.to_pandas()
    df_test_pandas = df_test.to_pandas()
    df_train_pandas['datetime'] = pd.to_datetime(df_train_pandas['datetime'])
    df_test_pandas['datetime'] = pd.to_datetime(df_test_pandas['datetime'])
    return df_train_pandas, df_test_pandas


@app.cell
def _(alt, df_train_pandas):
    # 時系列トレンド
    chart_timeseries = alt.Chart(df_train_pandas).mark_line(
        point=True, strokeWidth=2
    ).encode(
        x=alt.X('datetime:T', title='日付'),
        y=alt.Y('y:Q', title='販売数', scale=alt.Scale(zero=False)),
        tooltip=['datetime:T', 'y:Q', 'week:N', 'name:N', 'weather:N']
    ).properties(
        width=800,
        height=300,
        title='販売数の時系列推移'
    ).interactive()

    chart_timeseries
    return chart_timeseries,


@app.cell
def _(alt, df_train_pandas):
    # 曜日別ボックスプロット
    chart_week_box = alt.Chart(df_train_pandas).mark_boxplot().encode(
        x=alt.X('week:N', title='曜日', sort=['月', '火', '水', '木', '金']),
        y=alt.Y('y:Q', title='販売数'),
        color=alt.Color('week:N', sort=['月', '火', '水', '木', '金'], legend=None)
    ).properties(
        width=600,
        height=350,
        title='曜日別販売数の分布'
    )

    chart_week_box
    return chart_week_box,


@app.cell
def _(alt, df_train_pandas):
    # 曜日別平均（棒グラフ）
    chart_week_bar = alt.Chart(df_train_pandas).mark_bar().encode(
        x=alt.X('week:N', title='曜日', sort=['月', '火', '水', '木', '金']),
        y=alt.Y('mean(y):Q', title='平均販売数'),
        color=alt.Color('week:N', sort=['月', '火', '水', '木', '金'], legend=None),
        tooltip=['week:N', 'mean(y):Q']
    ).properties(
        width=600,
        height=350,
        title='曜日別平均販売数'
    )

    chart_week_bar
    return chart_week_bar,


@app.cell
def _(alt, df_train_pandas):
    # 天気と販売数
    chart_weather = alt.Chart(
        df_train_pandas[df_train_pandas['weather'].notna()]
    ).mark_boxplot().encode(
        x=alt.X('weather:N', title='天気'),
        y=alt.Y('y:Q', title='販売数'),
        color='weather:N'
    ).properties(
        width=600,
        height=350,
        title='天気別販売数の分布'
    )

    chart_weather
    return chart_weather,


@app.cell
def _(alt, df_train_pandas):
    # 気温と販売数の散布図
    df_temp_filtered = df_train_pandas[df_train_pandas['temperature'].notna()]

    chart_temperature = alt.Chart(df_temp_filtered).mark_circle(size=60).encode(
        x=alt.X('temperature:Q', title='気温 (℃)'),
        y=alt.Y('y:Q', title='販売数'),
        color=alt.Color('week:N', title='曜日'),
        tooltip=['datetime:T', 'temperature:Q', 'y:Q', 'week:N', 'weather:N']
    ).properties(
        width=700,
        height=400,
        title='気温と販売数の関係'
    ).interactive()

    chart_temperature
    return chart_temperature,


@app.cell
def _(alt, df_train_pandas):
    # カロリーと販売数
    df_kcal_filtered = df_train_pandas[df_train_pandas['kcal'].notna()]

    chart_kcal = alt.Chart(df_kcal_filtered).mark_circle(size=60).encode(
        x=alt.X('kcal:Q', title='カロリー (kcal)'),
        y=alt.Y('y:Q', title='販売数'),
        color=alt.Color('week:N', title='曜日'),
        tooltip=['datetime:T', 'kcal:Q', 'y:Q', 'name:N']
    ).properties(
        width=700,
        height=400,
        title='カロリーと販売数の関係'
    ).interactive()

    chart_kcal
    return chart_kcal,


@app.cell
def _(alt, df_train_pandas):
    # 売り切れの影響可視化
    chart_soldout = alt.Chart(df_train_pandas).mark_bar().encode(
        x=alt.X('soldout:N', title='売り切れフラグ (0=なし, 1=あり)'),
        y=alt.Y('mean(y):Q', title='平均販売数'),
        color=alt.Color('soldout:N', legend=None),
        tooltip=['soldout:N', 'mean(y):Q', 'count():Q']
    ).properties(
        width=400,
        height=350,
        title='売り切れ有無と平均販売数'
    )

    chart_soldout
    return chart_soldout,


@app.cell
def _(alt, df_train_pandas):
    # ヒートマップ（曜日×月）
    df_heatmap = df_train_pandas.copy()
    df_heatmap['month'] = df_heatmap['datetime'].dt.month

    chart_heatmap = alt.Chart(df_heatmap).mark_rect().encode(
        x=alt.X('week:N', title='曜日', sort=['月', '火', '水', '木', '金']),
        y=alt.Y('month:O', title='月'),
        color=alt.Color('mean(y):Q', title='平均販売数', scale=alt.Scale(scheme='viridis')),
        tooltip=['week:N', 'month:O', 'mean(y):Q']
    ).properties(
        width=600,
        height=300,
        title='月×曜日 平均販売数ヒートマップ'
    )

    chart_heatmap
    return chart_heatmap,


# ===== グループD: 特徴量エンジニアリング =====

@app.cell
def _(mo):
    mo.md("""
    ## 3. 特徴量エンジニアリング

    モデル学習のための特徴量を作成します。
    """)
    return


@app.cell
def _(pd, np):
    def create_features(df_pd, kcal_median=None, temp_median=None):
        """特徴量を作成する関数"""
        df_fe = df_pd.copy()

        # datetime列をパース
        df_fe['datetime'] = pd.to_datetime(df_fe['datetime'])

        # 日付関連特徴量
        df_fe['year'] = df_fe['datetime'].dt.year
        df_fe['month'] = df_fe['datetime'].dt.month
        df_fe['day'] = df_fe['datetime'].dt.day
        df_fe['dayofweek'] = df_fe['datetime'].dt.dayofweek  # 0=月曜, 4=金曜
        df_fe['weekofyear'] = df_fe['datetime'].dt.isocalendar().week

        # 曜日のワンホットエンコーディング
        week_dummies = pd.get_dummies(df_fe['week'], prefix='week')
        df_fe = pd.concat([df_fe, week_dummies], axis=1)

        # 天気のワンホットエンコーディング（すべてのカテゴリを含める）
        weather_categories = ['快晴', '晴れ', '曇', '薄曇', '雨', '雪', '雷電']
        weather_categorical = pd.Categorical(df_fe['weather'], categories=weather_categories)
        weather_dummies = pd.get_dummies(weather_categorical, prefix='weather')
        df_fe = pd.concat([df_fe, weather_dummies], axis=1)

        # 欠損値処理
        # kcal: 指定された中央値（または自身の中央値）で補完
        fill_kcal = kcal_median if kcal_median is not None else df_fe['kcal'].median()
        df_fe['kcal'] = df_fe['kcal'].fillna(fill_kcal)

        # precipitation: "--" を 0 に変換、その後数値化
        df_fe['precipitation'] = df_fe['precipitation'].replace('--', '0')
        df_fe['precipitation'] = pd.to_numeric(df_fe['precipitation'], errors='coerce').fillna(0)

        # temperature: 数値化
        df_fe['temperature'] = pd.to_numeric(df_fe['temperature'], errors='coerce')
        fill_temp = temp_median if temp_median is not None else df_fe['temperature'].median()
        df_fe['temperature'] = df_fe['temperature'].fillna(fill_temp)

        # soldout, payday, event: 欠損値を0で補完
        df_fe['soldout'] = df_fe['soldout'].fillna(0).astype(int)
        df_fe['payday'] = df_fe['payday'].fillna(0).astype(float)
        df_fe['event'] = df_fe['event'].notna().astype(int)  # イベントの有無（1/0）
        df_fe['remarks'] = df_fe['remarks'].notna().astype(int)  # 備考の有無（1/0）

        return df_fe

    return create_features,


@app.cell
def _(df_train, df_test, create_features, pd):
    # 訓練データの統計量（中央値）を計算
    train_pd = df_train.to_pandas()
    train_kcal_median = train_pd['kcal'].median()
    train_temp_median = pd.to_numeric(train_pd['temperature'], errors='coerce').median()

    # 特徴量作成の実行（訓練データの統計量を使用）
    df_train_fe = create_features(train_pd, train_kcal_median, train_temp_median)
    df_test_fe = create_features(df_test.to_pandas(), train_kcal_median, train_temp_median)

    # 作成された特徴量のカラム一覧
    feature_cols = [col for col in df_train_fe.columns
                   if col not in ['datetime', 'y', 'week', 'name', 'weather', 'remarks']]

    return df_train_fe, df_test_fe, feature_cols, train_kcal_median, train_temp_median


# ===== グループE: モデル構築 =====

@app.cell
def _(mo):
    mo.md("""
    ## 4. モデル構築と評価

    複数のモデルを構築し、バリデーションセットでRMSEを評価します。
    """)
    return


@app.cell
def _(df_train_fe, feature_cols):
    # データ分割（時系列考慮で80/20分割）
    X_train_full = df_train_fe[feature_cols]
    y_train_full = df_train_fe['y']

    # 訓練データとバリデーションデータに分割
    split_idx = int(len(X_train_full) * 0.8)
    X_train = X_train_full.iloc[:split_idx]
    y_train = y_train_full.iloc[:split_idx]
    X_val = X_train_full.iloc[split_idx:]
    y_val = y_train_full.iloc[split_idx:]

    return X_train_full, y_train_full, X_train, y_train, X_val, y_val


@app.cell
def _(X_train, y_train, X_val, y_val, LinearRegression, Ridge, mean_squared_error, np, mo):
    # 線形回帰
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_val)
    lr_rmse = np.sqrt(mean_squared_error(y_val, lr_pred))

    # Ridge回帰
    ridge_model = Ridge(alpha=10.0)
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_val)
    ridge_rmse = np.sqrt(mean_squared_error(y_val, ridge_pred))

    mo.md(f"""
    ### 線形モデルの結果
    - 線形回帰 RMSE: {lr_rmse:.2f}
    - Ridge回帰 RMSE: {ridge_rmse:.2f}
    """)

    return lr_model, lr_rmse, ridge_model, ridge_rmse


@app.cell
def _(X_train, y_train, X_val, y_val, DecisionTreeRegressor, RandomForestRegressor, mean_squared_error, np, mo):
    # 決定木
    dt_model = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_val)
    dt_rmse = np.sqrt(mean_squared_error(y_val, dt_pred))

    # ランダムフォレスト
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_val)
    rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))

    mo.md(f"""
    ### 決定木系モデルの結果
    - 決定木 RMSE: {dt_rmse:.2f}
    - ランダムフォレスト RMSE: {rf_rmse:.2f}
    """)

    return dt_model, dt_rmse, rf_model, rf_rmse


@app.cell
def _(X_train, y_train, X_val, y_val, GradientBoostingRegressor, mean_squared_error, np, mo):
    # Gradient Boosting（LightGBM相当の性能を期待）
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        subsample=0.8,
        random_state=42,
        verbose=0
    )

    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_val)
    gb_rmse = np.sqrt(mean_squared_error(y_val, gb_pred))

    mo.md(f"""
    ### Gradient Boosting（勾配ブースティング）の結果
    - RMSE: {gb_rmse:.2f}
    """)

    return gb_model, gb_rmse


# ===== グループF: 予測と提出 =====

@app.cell
def _(lr_rmse, ridge_rmse, dt_rmse, rf_rmse, gb_rmse, pd):
    # モデル比較表
    model_comparison = pd.DataFrame({
        'モデル': ['線形回帰', 'Ridge回帰', '決定木', 'ランダムフォレスト', 'Gradient Boosting'],
        'RMSE': [lr_rmse, ridge_rmse, dt_rmse, rf_rmse, gb_rmse]
    }).sort_values('RMSE')

    model_comparison
    return model_comparison,


@app.cell
def _(model_comparison, lr_model, ridge_model, dt_model, rf_model, gb_model, mo):
    # ベストモデルの選択
    best_model_name = model_comparison.iloc[0]['モデル']
    best_rmse = model_comparison.iloc[0]['RMSE']

    # モデル名から実際のモデルオブジェクトを選択
    model_dict = {
        '線形回帰': lr_model,
        'Ridge回帰': ridge_model,
        '決定木': dt_model,
        'ランダムフォレスト': rf_model,
        'Gradient Boosting': gb_model
    }
    best_model = model_dict[best_model_name]

    mo.md(f"""
    ## 5. 最終予測

    ### ベストモデル
    - モデル: {best_model_name}
    - バリデーションRMSE: {best_rmse:.2f}
    """)

    return best_model, best_model_name


@app.cell
def _(df_test_fe, feature_cols, best_model):
    # テストデータへの予測
    X_test = df_test_fe[feature_cols]

    test_predictions = best_model.predict(X_test)

    # 負の値を0にクリップ（販売数は0以上）
    test_predictions = [max(0, pred) for pred in test_predictions]

    return X_test, test_predictions


@app.cell
def _(df_test_fe, test_predictions, Path, pd, mo):
    # 提出ファイル作成
    # 日付フォーマットの調整（yyyy-m-d形式、0埋めなし）
    submission_dates = df_test_fe['datetime'].dt.strftime('%Y-%-m-%-d').tolist()

    # 提出用DataFrameの作成
    submission_df = pd.DataFrame({
        'datetime': submission_dates,
        'y': [int(round(pred)) for pred in test_predictions]
    })

    # CSVファイルとして保存（ヘッダーなし）
    output_path = Path(__file__).parent.parent / "data" / "submission.csv"
    submission_df.to_csv(output_path, index=False, header=False)

    mo.md(f"""
    ## 6. 提出ファイル生成

    提出ファイルを作成しました: `{output_path}`

    形式: yyyy-m-d（0埋めなし）、ヘッダーなし

    ### 予測結果プレビュー:
    """)
    submission_df.head(10)

    return submission_df, output_path


if __name__ == "__main__":
    app.run()
