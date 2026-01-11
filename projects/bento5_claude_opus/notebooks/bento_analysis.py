import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def __():
    # ライブラリインポート
    import marimo as mo
    import polars as pl
    import pandas as pd
    import altair as alt
    from pathlib import Path
    import numpy as np
    import re
    # 注: LightGBMとOptunaは後のPhaseで使用
    # from sklearn.linear_model import LinearRegression
    # from sklearn.model_selection import TimeSeriesSplit
    # from sklearn.metrics import mean_squared_error
    # import lightgbm as lgb
    # import optuna
    return (
        mo,
        pl,
        pd,
        alt,
        Path,
        np,
        re,
    )


@app.cell
def __(pl, Path):
    # 訓練データロード
    data_dir = Path(__file__).parent.parent / "data"
    df_train = pl.read_csv(data_dir / "bento_train.csv")
    return df_train, data_dir


@app.cell
def __(pl, data_dir):
    # テストデータロード
    df_test = pl.read_csv(data_dir / "bento_test.csv")
    return df_test,


@app.cell
def __(mo, df_train, df_test):
    # タイトルと概要
    mo.md(f"""
    # お弁当販売数予測 - EDA

    ## データ概要

    | 項目 | 訓練データ | テストデータ |
    |------|-----------|-------------|
    | 行数 | {len(df_train)} | {len(df_test)} |
    | 列数 | {len(df_train.columns)} | {len(df_test.columns)} |

    評価指標: RMSE（Root Mean Squared Error）
    """)
    return


@app.cell
def __(mo):
    # セクションヘッダー: 基本統計
    mo.md("## 1. 基本統計とデータ理解")
    return


@app.cell
def __(df_train, mo):
    # 基本統計量
    train_stats = df_train.describe()
    mo.ui.table(train_stats)
    return train_stats,


@app.cell
def __(df_train, pl, mo):
    # 欠損値確認
    null_counts = df_train.null_count()
    null_info = pl.DataFrame({
        "カラム": df_train.columns,
        "欠損数": [null_counts[col][0] for col in df_train.columns],
        "欠損率(%)": [round(null_counts[col][0] / len(df_train) * 100, 2) for col in df_train.columns]
    })
    mo.md(f"""
    ### 欠損値の確認

    {mo.ui.table(null_info)}
    """)
    return null_counts, null_info


@app.cell
def __(df_train, mo):
    # 目的変数の統計
    y_mean = df_train["y"].mean()
    y_median = df_train["y"].median()
    y_std = df_train["y"].std()
    y_min = df_train["y"].min()
    y_max = df_train["y"].max()

    mo.md(f"""
    ### 目的変数（y: 販売数）の統計

    - 平均: {y_mean:.2f}
    - 中央値: {y_median:.2f}
    - 標準偏差: {y_std:.2f}
    - 範囲: {y_min} 〜 {y_max}
    """)
    return y_mean, y_median, y_std, y_min, y_max


@app.cell
def __(df_train, pl, mo):
    # カテゴリカル変数の集計
    week_stats = df_train.group_by("week").agg([
        pl.col("y").count().alias("件数"),
        pl.col("y").mean().alias("平均販売数"),
        pl.col("y").median().alias("中央値"),
        pl.col("y").std().alias("標準偏差")
    ]).sort("week")

    weather_stats = df_train.group_by("weather").agg([
        pl.col("y").count().alias("件数"),
        pl.col("y").mean().alias("平均販売数"),
        pl.col("y").median().alias("中央値")
    ]).sort("平均販売数", descending=True)

    soldout_stats = df_train.group_by("soldout").agg([
        pl.col("y").count().alias("件数"),
        pl.col("y").mean().alias("平均販売数")
    ]).sort("soldout")

    mo.md(f"""
    ### カテゴリカル変数の集計

    #### 曜日別
    {mo.ui.table(week_stats)}

    #### 天気別
    {mo.ui.table(weather_stats)}

    #### 売り切れ別
    {mo.ui.table(soldout_stats)}
    """)
    return week_stats, weather_stats, soldout_stats


@app.cell
def __(df_train, pl, mo):
    # remarks の詳細集計
    remarks_stats = df_train.filter(pl.col("remarks").is_not_null()).group_by("remarks").agg([
        pl.col("y").count().alias("件数"),
        pl.col("y").mean().alias("平均販売数"),
        pl.col("y").median().alias("中央値")
    ]).sort("件数", descending=True)

    # remarks有無別の集計
    remarks_summary = pl.DataFrame({
        "カテゴリ": ["remarks有り", "remarks無し"],
        "件数": [
            len(df_train.filter(pl.col("remarks").is_not_null())),
            len(df_train.filter(pl.col("remarks").is_null()))
        ],
        "平均販売数": [
            df_train.filter(pl.col("remarks").is_not_null())["y"].mean(),
            df_train.filter(pl.col("remarks").is_null())["y"].mean()
        ]
    })

    mo.md(f"""
    ### remarks（備考）の分析

    #### remarks の内容
    {mo.ui.table(remarks_stats)}

    #### remarks 有無別の販売数
    {mo.ui.table(remarks_summary)}

    remarksがあると平均販売数がやや高い傾向（90.6 vs 86.2）
    """)
    return remarks_stats, remarks_summary


@app.cell
def __(df_train, pl, mo):
    # event の詳細集計
    event_stats = df_train.filter(pl.col("event").is_not_null()).group_by("event").agg([
        pl.col("y").count().alias("件数"),
        pl.col("y").mean().alias("平均販売数"),
        pl.col("y").median().alias("中央値")
    ]).sort("件数", descending=True)

    # event有無別の集計
    event_summary = pl.DataFrame({
        "カテゴリ": ["event有り", "event無し"],
        "件数": [
            len(df_train.filter(pl.col("event").is_not_null())),
            len(df_train.filter(pl.col("event").is_null()))
        ],
        "平均販売数": [
            df_train.filter(pl.col("event").is_not_null())["y"].mean(),
            df_train.filter(pl.col("event").is_null())["y"].mean()
        ]
    })

    mo.md(f"""
    ### event（イベント）の分析

    #### event の内容
    {mo.ui.table(event_stats)}

    #### event 有無別の販売数
    {mo.ui.table(event_summary)}

    eventがあると平均販売数がやや低い傾向（80.8 vs 87.0）
    - イベント時は弁当需要が減少する可能性
    """)
    return event_stats, event_summary


@app.cell
def __(mo):
    # セクションヘッダー: 可視化
    mo.md("## 2. 可視化による分析")
    return


@app.cell
def __(df_train, df_test, pd):
    # Pandas変換（可視化用に1回のみ）
    df_train_pandas = df_train.to_pandas()
    df_test_pandas = df_test.to_pandas()

    # datetime を datetime型に変換
    df_train_pandas['datetime'] = pd.to_datetime(df_train_pandas['datetime'])
    df_test_pandas['datetime'] = pd.to_datetime(df_test_pandas['datetime'])
    return df_train_pandas, df_test_pandas


@app.cell
def __(df_train_pandas, alt, mo):
    # 販売数のヒストグラム
    chart_y_hist = alt.Chart(df_train_pandas).mark_bar().encode(
        x=alt.X('y:Q', bin=alt.Bin(maxbins=20), title='販売数'),
        y=alt.Y('count()', title='件数'),
        tooltip=['count()']
    ).properties(
        title='販売数の分布',
        width=500,
        height=300
    )

    mo.md(f"""
    ### 販売数の分布

    {mo.ui.altair_chart(chart_y_hist)}
    """)
    return chart_y_hist,


@app.cell
def __(df_train_pandas, alt, mo):
    # 時系列トレンド
    chart_timeseries = alt.Chart(df_train_pandas).mark_line(point=True).encode(
        x=alt.X('datetime:T', title='日付'),
        y=alt.Y('y:Q', title='販売数'),
        tooltip=['datetime:T', 'y:Q', 'week:N', 'weather:N']
    ).properties(
        title='時系列トレンド',
        width=700,
        height=300
    )

    mo.md(f"""
    ### 時系列トレンド

    {mo.ui.altair_chart(chart_timeseries)}
    """)
    return chart_timeseries,


@app.cell
def __(df_train_pandas, alt, mo):
    # 曜日別ボックスプロット
    chart_week_box = alt.Chart(df_train_pandas).mark_boxplot().encode(
        x=alt.X('week:N', title='曜日', sort=['月', '火', '水', '木', '金']),
        y=alt.Y('y:Q', title='販売数'),
        color='week:N',
        tooltip=['week:N', 'y:Q']
    ).properties(
        title='曜日別販売数',
        width=500,
        height=300
    )

    mo.md(f"""
    ### 曜日別販売数

    {mo.ui.altair_chart(chart_week_box)}
    """)
    return chart_week_box,


@app.cell
def __(df_train_pandas, alt, mo):
    # 天気別ボックスプロット
    chart_weather = alt.Chart(df_train_pandas).mark_boxplot().encode(
        x=alt.X('weather:N', title='天気'),
        y=alt.Y('y:Q', title='販売数'),
        color='weather:N',
        tooltip=['weather:N', 'y:Q']
    ).properties(
        title='天気別販売数',
        width=500,
        height=300
    )

    mo.md(f"""
    ### 天気別販売数

    {mo.ui.altair_chart(chart_weather)}
    """)
    return chart_weather,


@app.cell
def __(df_train_pandas, alt, pd, mo):
    # 気温と販売数の散布図
    df_temp = df_train_pandas.copy()
    df_temp['temperature'] = pd.to_numeric(df_temp['temperature'], errors='coerce')

    chart_temperature = alt.Chart(df_temp.dropna(subset=['temperature'])).mark_circle(size=60).encode(
        x=alt.X('temperature:Q', title='気温（℃）'),
        y=alt.Y('y:Q', title='販売数'),
        color=alt.Color('week:N', title='曜日'),
        tooltip=['datetime:T', 'temperature:Q', 'y:Q', 'week:N', 'weather:N']
    ).properties(
        title='気温と販売数の関係',
        width=600,
        height=400
    )

    mo.md(f"""
    ### 気温と販売数の関係

    {mo.ui.altair_chart(chart_temperature)}
    """)
    return df_temp, chart_temperature


@app.cell
def __(df_train, pl, alt, pd, mo):
    # 相関行列ヒートマップ
    # 数値変数のみ抽出
    numeric_cols = ['y', 'soldout', 'kcal', 'payday', 'precipitation', 'temperature']

    # Polarsで数値変換
    df_corr = df_train.select([
        pl.col('y'),
        pl.col('soldout').cast(pl.Float64),
        pl.col('kcal').cast(pl.Float64),
        pl.col('payday').cast(pl.Float64),
        pl.col('precipitation').str.replace('--', '0').cast(pl.Float64),
        pl.col('temperature').cast(pl.Float64)
    ])

    # 相関行列計算
    corr_matrix = df_corr.to_pandas().corr()

    # ヒートマップ用にデータ整形
    corr_data = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            corr_data.append({
                'var1': col1,
                'var2': col2,
                'correlation': corr_matrix.iloc[i, j]
            })

    corr_df = pd.DataFrame(corr_data)

    chart_corr = alt.Chart(corr_df).mark_rect().encode(
        x=alt.X('var1:N', title=''),
        y=alt.Y('var2:N', title=''),
        color=alt.Color('correlation:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1]), title='相関係数'),
        tooltip=['var1:N', 'var2:N', alt.Tooltip('correlation:Q', format='.3f')]
    ).properties(
        title='相関行列ヒートマップ',
        width=400,
        height=400
    )

    mo.md(f"""
    ### 相関行列ヒートマップ

    {mo.ui.altair_chart(chart_corr)}
    """)
    return numeric_cols, df_corr, corr_matrix, corr_data, corr_df, chart_corr


@app.cell
def __(df_train_pandas, alt, pd, mo):
    # 月×曜日ヒートマップ
    df_heatmap = df_train_pandas.copy()
    df_heatmap['month'] = df_heatmap['datetime'].dt.month

    # 月×曜日の平均販売数を計算
    heatmap_data = df_heatmap.groupby(['month', 'week'])['y'].mean().reset_index()

    chart_heatmap = alt.Chart(heatmap_data).mark_rect().encode(
        x=alt.X('month:O', title='月'),
        y=alt.Y('week:N', title='曜日', sort=['月', '火', '水', '木', '金']),
        color=alt.Color('y:Q', scale=alt.Scale(scheme='blues'), title='平均販売数'),
        tooltip=['month:O', 'week:N', alt.Tooltip('y:Q', format='.1f', title='平均販売数')]
    ).properties(
        title='月×曜日の平均販売数',
        width=500,
        height=300
    )

    mo.md(f"""
    ### 月×曜日の平均販売数

    {mo.ui.altair_chart(chart_heatmap)}
    """)
    return df_heatmap, heatmap_data, chart_heatmap


@app.cell
def __(df_train_pandas, alt, pd, mo):
    # remarks有無別の販売数ボックスプロット
    df_remarks_viz = df_train_pandas.copy()
    df_remarks_viz['remarks_flag'] = df_remarks_viz['remarks'].notna().map({True: 'remarks有り', False: 'remarks無し'})

    chart_remarks = alt.Chart(df_remarks_viz).mark_boxplot().encode(
        x=alt.X('remarks_flag:N', title=''),
        y=alt.Y('y:Q', title='販売数'),
        color='remarks_flag:N',
        tooltip=['remarks_flag:N', 'y:Q']
    ).properties(
        title='remarks有無別の販売数',
        width=400,
        height=300
    )

    mo.md(f"""
    ### remarks有無別の販売数

    {mo.ui.altair_chart(chart_remarks)}

    remarksがある場合、販売数の中央値・平均ともにやや高い
    """)
    return df_remarks_viz, chart_remarks


@app.cell
def __(df_train_pandas, alt, pd, mo):
    # event有無別の販売数ボックスプロット
    df_event_viz = df_train_pandas.copy()
    df_event_viz['event_flag'] = df_event_viz['event'].notna().map({True: 'event有り', False: 'event無し'})

    chart_event = alt.Chart(df_event_viz).mark_boxplot().encode(
        x=alt.X('event_flag:N', title=''),
        y=alt.Y('y:Q', title='販売数'),
        color='event_flag:N',
        tooltip=['event_flag:N', 'y:Q']
    ).properties(
        title='event有無別の販売数',
        width=400,
        height=300
    )

    mo.md(f"""
    ### event有無別の販売数

    {mo.ui.altair_chart(chart_event)}

    eventがある場合、販売数が低い傾向
    - イベント時は弁当を外で食べる、または用意されるため需要減少と推測
    """)
    return df_event_viz, chart_event


@app.cell
def __(df_train_pandas, alt, pd, mo):
    # 特別メニューの分析
    df_special = df_train_pandas.copy()

    # 特別メニューのフラグを作成
    df_special['is_special'] = df_special['remarks'].apply(
        lambda x: 'お楽しみメニュー' if pd.notna(x) and 'お楽しみ' in str(x)
        else '料理長のこだわり' if pd.notna(x) and '料理長' in str(x)
        else 'その他特別' if pd.notna(x)
        else '通常メニュー'
    )

    # メニュータイプ別の販売数
    chart_special = alt.Chart(df_special).mark_boxplot().encode(
        x=alt.X('is_special:N', title='メニュータイプ', sort=['お楽しみメニュー', '料理長のこだわり', 'その他特別', '通常メニュー']),
        y=alt.Y('y:Q', title='販売数'),
        color='is_special:N',
        tooltip=['is_special:N', 'y:Q']
    ).properties(
        title='メニュータイプ別の販売数',
        width=600,
        height=300
    )

    mo.md(f"""
    ### 特別メニューの効果分析

    {mo.ui.altair_chart(chart_special)}

    - お楽しみメニュー: 販売数が高い傾向
    - 料理長のこだわり: やや高い
    - 特別メニューは販売促進に効果的
    """)
    return df_special, chart_special


@app.cell
def __(mo):
    # セクションヘッダー: 特徴量エンジニアリング
    mo.md("## 3. 特徴量エンジニアリング")
    return


@app.cell
def __(pd, np):
    # 特徴量作成関数
    def create_features(df, train_temp_median=None, train_kcal_median=None):
        """
        EDAの知見を反映した特徴量を作成

        Parameters:
        - df: pd.DataFrame
        - train_temp_median: 訓練データの気温中央値（テストデータの場合に使用）
        - train_kcal_median: 訓練データのkcal中央値（テストデータの場合に使用）
        """
        df_fe = df.copy()

        # 日付特徴量
        df_fe['datetime'] = pd.to_datetime(df_fe['datetime'])
        df_fe['year'] = df_fe['datetime'].dt.year
        df_fe['month'] = df_fe['datetime'].dt.month
        df_fe['day'] = df_fe['datetime'].dt.day
        df_fe['dayofweek'] = df_fe['datetime'].dt.dayofweek  # 0=月曜日

        # 曜日ワンホット
        for i, w in enumerate(['月', '火', '水', '木', '金']):
            df_fe[f'week_{w}'] = (df_fe['week'] == w).astype(int)

        # 天気ワンホット
        weather_list = ['快晴', '晴れ', '曇', '薄曇', '雨', '雪', '雷電']
        for w in weather_list:
            df_fe[f'weather_{w}'] = (df_fe['weather'] == w).astype(int)

        # 悪天候フラグ
        df_fe['is_bad_weather'] = df_fe['weather'].isin(['雨', '雪', '雷電']).astype(int)

        # 気温（欠損は訓練データ中央値で補完）
        df_fe['temperature'] = pd.to_numeric(df_fe['temperature'], errors='coerce')
        if train_temp_median is not None:
            df_fe['temperature'] = df_fe['temperature'].fillna(train_temp_median)
        else:
            train_temp_median = df_fe['temperature'].median()
            df_fe['temperature'] = df_fe['temperature'].fillna(train_temp_median)

        # kcal（欠損は訓練データ中央値で補完）
        df_fe['kcal'] = pd.to_numeric(df_fe['kcal'], errors='coerce')
        if train_kcal_median is not None:
            df_fe['kcal'] = df_fe['kcal'].fillna(train_kcal_median)
        else:
            train_kcal_median = df_fe['kcal'].median()
            df_fe['kcal'] = df_fe['kcal'].fillna(train_kcal_median)

        # remarks関連
        df_fe['remarks_flag'] = df_fe['remarks'].notna().astype(int)
        df_fe['is_special_menu'] = df_fe['remarks'].apply(
            lambda x: 1 if pd.notna(x) and ('お楽しみ' in str(x) or '料理長' in str(x)) else 0
        )
        df_fe['is_otanoshimi'] = df_fe['remarks'].apply(
            lambda x: 1 if pd.notna(x) and 'お楽しみ' in str(x) else 0
        )

        # event関連
        df_fe['event_flag'] = df_fe['event'].notna().astype(int)

        # payday
        df_fe['payday'] = pd.to_numeric(df_fe['payday'], errors='coerce').fillna(0).astype(int)

        # soldout
        df_fe['soldout'] = pd.to_numeric(df_fe['soldout'], errors='coerce').fillna(0).astype(int)

        # precipitation
        df_fe['precipitation'] = df_fe['precipitation'].replace('--', '0')
        df_fe['precipitation'] = pd.to_numeric(df_fe['precipitation'], errors='coerce').fillna(0)

        return df_fe

    return create_features,


@app.cell
def __(df_train_pandas, df_test_pandas, create_features, mo):
    # 特徴量作成の実行

    # 訓練データの統計量を先に計算（データリーク防止）
    train_temp_median = pd.to_numeric(df_train_pandas['temperature'], errors='coerce').median()
    train_kcal_median = pd.to_numeric(df_train_pandas['kcal'], errors='coerce').median()

    # 特徴量作成
    df_train_fe = create_features(df_train_pandas)
    df_test_fe = create_features(df_test_pandas, train_temp_median, train_kcal_median)

    # 特徴量カラム
    feature_cols = [
        'year', 'month', 'day', 'dayofweek',
        'week_月', 'week_火', 'week_水', 'week_木', 'week_金',
        'weather_快晴', 'weather_晴れ', 'weather_曇', 'weather_薄曇', 'weather_雨', 'weather_雪', 'weather_雷電',
        'is_bad_weather',
        'temperature', 'kcal', 'precipitation',
        'remarks_flag', 'is_special_menu', 'event_flag',
        'payday', 'soldout'
    ]

    mo.md(f"""
    ### 特徴量作成完了

    - 訓練データ: {len(df_train_fe)} 行
    - テストデータ: {len(df_test_fe)} 行
    - 特徴量数: {len(feature_cols)} 個

    統計量（データリーク防止）:
    - 訓練データの気温中央値: {train_temp_median:.1f}℃
    - 訓練データのkcal中央値: {train_kcal_median:.1f}
    """)
    return df_train_fe, df_test_fe, feature_cols, train_temp_median, train_kcal_median


@app.cell
def __(feature_cols, mo, pd):
    # 特徴量一覧表示
    feature_df = pd.DataFrame({
        '特徴量': feature_cols,
        'カテゴリ': [
            '日付', '日付', '日付', '日付',
            '曜日', '曜日', '曜日', '曜日', '曜日',
            '天気', '天気', '天気', '天気', '天気', '天気', '天気',
            '天気',
            '数値', '数値', '数値',
            'remarks', 'remarks', 'event',
            'その他', 'その他'
        ]
    })

    mo.md(f"""
    ### 特徴量一覧

    {mo.ui.table(feature_df)}
    """)
    return feature_df,


@app.cell
def __(mo):
    # セクションヘッダー: ベースラインモデル
    mo.md("## 4. ベースラインモデル")
    return


@app.cell
def __(df_train_fe, feature_cols, np, mo):
    # ライブラリインポート（ここで必要なものだけ）
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error

    # 線形回帰モデル
    X = df_train_fe[feature_cols].values
    y = df_train_fe['y'].values

    # TimeSeriesSplit交差検証
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        cv_scores.append(rmse)

    overall_rmse = np.mean(cv_scores)

    mo.md(f"""
    ### 線形回帰モデル（ベースライン）

    交差検証: TimeSeriesSplit (n_splits=5)

    | Fold | RMSE |
    |------|------|
    | 1 | {cv_scores[0]:.2f} |
    | 2 | {cv_scores[1]:.2f} |
    | 3 | {cv_scores[2]:.2f} |
    | 4 | {cv_scores[3]:.2f} |
    | 5 | {cv_scores[4]:.2f} |
    | 平均 | {overall_rmse:.2f} |

    ベースラインRMSE: {overall_rmse:.2f}
    """)
    return LinearRegression, TimeSeriesSplit, mean_squared_error, X, y, tscv, cv_scores, overall_rmse


@app.cell
def __(mo):
    # セクションヘッダー: 高度なモデリング
    mo.md("## 5. 高度なモデリング")
    return


@app.cell
def __(X, y, TimeSeriesSplit, mean_squared_error, np, mo):
    # GradientBoostingRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    # ハイパーパラメータ
    gb_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'subsample': 0.8,
        'random_state': 42
    }

    # TimeSeriesSplit交差検証
    tscv_gb = TimeSeriesSplit(n_splits=5)
    gb_cv_scores = []

    for fold_gb, (train_idx_gb, val_idx_gb) in enumerate(tscv_gb.split(X)):
        X_train_gb, X_val_gb = X[train_idx_gb], X[val_idx_gb]
        y_train_gb, y_val_gb = y[train_idx_gb], y[val_idx_gb]

        model_gb = GradientBoostingRegressor(**gb_params)
        model_gb.fit(X_train_gb, y_train_gb)

        y_pred_gb = model_gb.predict(X_val_gb)
        rmse_gb = np.sqrt(mean_squared_error(y_val_gb, y_pred_gb))
        gb_cv_scores.append(rmse_gb)

    gb_overall_rmse = np.mean(gb_cv_scores)

    mo.md(f"""
    ### GradientBoostingRegressor

    ハイパーパラメータ:
    - n_estimators: {gb_params['n_estimators']}
    - learning_rate: {gb_params['learning_rate']}
    - max_depth: {gb_params['max_depth']}

    交差検証結果:

    | Fold | RMSE |
    |------|------|
    | 1 | {gb_cv_scores[0]:.2f} |
    | 2 | {gb_cv_scores[1]:.2f} |
    | 3 | {gb_cv_scores[2]:.2f} |
    | 4 | {gb_cv_scores[3]:.2f} |
    | 5 | {gb_cv_scores[4]:.2f} |
    | 平均 | {gb_overall_rmse:.2f} |
    """)
    return GradientBoostingRegressor, gb_params, tscv_gb, gb_cv_scores, gb_overall_rmse, model_gb


@app.cell
def __(X, y, TimeSeriesSplit, mean_squared_error, np, mo):
    # RandomForestRegressor
    from sklearn.ensemble import RandomForestRegressor

    # ハイパーパラメータ
    rf_params = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }

    # TimeSeriesSplit交差検証
    tscv_rf = TimeSeriesSplit(n_splits=5)
    rf_cv_scores = []

    for fold_rf, (train_idx_rf, val_idx_rf) in enumerate(tscv_rf.split(X)):
        X_train_rf, X_val_rf = X[train_idx_rf], X[val_idx_rf]
        y_train_rf, y_val_rf = y[train_idx_rf], y[val_idx_rf]

        model_rf = RandomForestRegressor(**rf_params)
        model_rf.fit(X_train_rf, y_train_rf)

        y_pred_rf = model_rf.predict(X_val_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_val_rf, y_pred_rf))
        rf_cv_scores.append(rmse_rf)

    rf_overall_rmse = np.mean(rf_cv_scores)

    mo.md(f"""
    ### RandomForestRegressor

    ハイパーパラメータ:
    - n_estimators: {rf_params['n_estimators']}
    - max_depth: {rf_params['max_depth']}
    - max_features: {rf_params['max_features']}

    交差検証結果:

    | Fold | RMSE |
    |------|------|
    | 1 | {rf_cv_scores[0]:.2f} |
    | 2 | {rf_cv_scores[1]:.2f} |
    | 3 | {rf_cv_scores[2]:.2f} |
    | 4 | {rf_cv_scores[3]:.2f} |
    | 5 | {rf_cv_scores[4]:.2f} |
    | 平均 | {rf_overall_rmse:.2f} |
    """)
    return RandomForestRegressor, rf_params, tscv_rf, rf_cv_scores, rf_overall_rmse, model_rf


@app.cell
def __(overall_rmse, gb_overall_rmse, rf_overall_rmse, mo, pd):
    # モデル比較サマリー
    model_comparison = pd.DataFrame({
        'モデル': ['線形回帰', 'GradientBoosting', 'RandomForest'],
        'RMSE': [overall_rmse, gb_overall_rmse, rf_overall_rmse],
        '改善率(%)': [
            0,
            (overall_rmse - gb_overall_rmse) / overall_rmse * 100,
            (overall_rmse - rf_overall_rmse) / overall_rmse * 100
        ]
    }).sort_values('RMSE')

    best_model = model_comparison.iloc[0]['モデル']
    best_rmse = model_comparison.iloc[0]['RMSE']
    improvement = model_comparison.iloc[0]['改善率(%)']

    mo.md(f"""
    ### モデル比較サマリー

    {mo.ui.table(model_comparison)}

    ベストモデル: {best_model}（RMSE: {best_rmse:.2f}、改善率: {improvement:.1f}%）
    """)
    return model_comparison, best_model, best_rmse, improvement


@app.cell
def __(mo):
    # セクションヘッダー: シンプルモデル
    mo.md("## 6. シンプルモデルで精度改善")
    return


@app.cell
def __(df_train_fe, LinearRegression, TimeSeriesSplit, mean_squared_error, np, mo):
    # 新しい4特徴量モデル
    # month, temperature, is_otanoshimi, is_bad_weather

    simple_features = ['month', 'temperature', 'is_otanoshimi', 'is_bad_weather']

    X_simple = df_train_fe[simple_features].values
    y_simple = df_train_fe['y'].values

    # TimeSeriesSplit交差検証
    tscv_simple = TimeSeriesSplit(n_splits=5)
    simple_cv_scores = []

    for fold_simple, (train_idx_simple, val_idx_simple) in enumerate(tscv_simple.split(X_simple)):
        X_train_simple, X_val_simple = X_simple[train_idx_simple], X_simple[val_idx_simple]
        y_train_simple, y_val_simple = y_simple[train_idx_simple], y_simple[val_idx_simple]

        model_simple = LinearRegression()
        model_simple.fit(X_train_simple, y_train_simple)

        y_pred_simple = model_simple.predict(X_val_simple)
        rmse_simple = np.sqrt(mean_squared_error(y_val_simple, y_pred_simple))
        simple_cv_scores.append(rmse_simple)

    simple_overall_rmse = np.mean(simple_cv_scores)

    mo.md(f"""
    ### 新しい4特徴量モデル（yearを除く）

    特徴量:
    - month（月）
    - temperature（気温）
    - is_otanoshimi（お楽しみメニュー）
    - is_bad_weather（悪天候フラグ）

    交差検証結果:

    | Fold | RMSE |
    |------|------|
    | 1 | {simple_cv_scores[0]:.2f} |
    | 2 | {simple_cv_scores[1]:.2f} |
    | 3 | {simple_cv_scores[2]:.2f} |
    | 4 | {simple_cv_scores[3]:.2f} |
    | 5 | {simple_cv_scores[4]:.2f} |
    | 平均 | {simple_overall_rmse:.2f} |

    参考: 過去の記録（year含む）RMSE 13.26
    """)
    return simple_features, X_simple, y_simple, tscv_simple, simple_cv_scores, simple_overall_rmse, model_simple


@app.cell
def __(overall_rmse, simple_overall_rmse, mo, pd):
    # 特徴量数による比較
    feature_comparison = pd.DataFrame({
        'モデル': ['4特徴量（シンプル）', '25特徴量（全特徴量）'],
        'RMSE': [simple_overall_rmse, overall_rmse],
        '改善率(%)': [
            0,
            (simple_overall_rmse - overall_rmse) / simple_overall_rmse * 100
        ]
    }).sort_values('RMSE')

    mo.md(f"""
    ### 特徴量数による比較

    {mo.ui.table(feature_comparison)}

    シンプルモデルの方が{'良い' if simple_overall_rmse < overall_rmse else '悪い'}結果
    """)
    return feature_comparison,


if __name__ == "__main__":
    app.run()
