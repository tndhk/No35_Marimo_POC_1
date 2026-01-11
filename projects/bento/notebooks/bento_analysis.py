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
    import lightgbm as lgb
    import re
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    import warnings
    warnings.filterwarnings('ignore')
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return (mo, pl, pd, alt, Path, np, lgb, re,
            TimeSeriesSplit, LinearRegression, Ridge,
            DecisionTreeRegressor, RandomForestRegressor,
            mean_squared_error, optuna)


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


@app.cell
def _(df_train, pl, alt):
    # 相関行列ヒートマップ
    numeric_cols = ['y', 'soldout', 'kcal', 'payday', 'temperature']
    
    # 数値変換とNaN除去
    df_corr = df_train.select([
        pl.col('y'),
        pl.col('soldout').fill_null(0),
        pl.col('kcal').fill_null(pl.col('kcal').median()),
        pl.col('payday').fill_null(0),
        pl.col('temperature').cast(pl.Float64).fill_null(pl.col('temperature').cast(pl.Float64).median())
    ]).to_pandas()
    
    corr_matrix = df_corr.corr()
    
    # 相関行列をlong形式に変換
    corr_data = corr_matrix.reset_index().melt(id_vars='index')
    corr_data.columns = ['var1', 'var2', 'correlation']
    
    chart_corr = alt.Chart(corr_data).mark_rect().encode(
        x=alt.X('var1:N', title=''),
        y=alt.Y('var2:N', title=''),
        color=alt.Color('correlation:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1]), title='相関係数'),
        tooltip=['var1:N', 'var2:N', alt.Tooltip('correlation:Q', format='.3f')]
    ).properties(
        width=400,
        height=400,
        title='相関行列ヒートマップ'
    )
    
    # 相関係数の値をテキストで表示
    chart_corr_text = alt.Chart(corr_data).mark_text(fontSize=12).encode(
        x='var1:N',
        y='var2:N',
        text=alt.Text('correlation:Q', format='.2f'),
        color=alt.condition(
            alt.datum.correlation > 0.5,
            alt.value('white'),
            alt.value('black')
        )
    )
    
    chart_corr + chart_corr_text
    return chart_corr,


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
def _(pd, np, re):
    def extract_menu_keywords(df, col_name='name'):
        """メニュー名からキーワードフラグを作成する"""
        # 抽出したいキーワードリスト
        keywords = {
            'curry': r'カレー',
            'fry': r'フライ|カツ|唐揚げ|天ぷら',
            'hamburg': r'ハンバーグ',
            'fish': r'魚|サバ|鮭|ぶり|ブリ',
            'meat': r'肉|牛|豚|鶏|チキン|ポーク|ビーフ',
            'veg': r'野菜|筑前煮',
        }
        
        for key, pattern in keywords.items():
            df[f'menu_{key}'] = df[col_name].apply(
                lambda x: 1 if re.search(pattern, str(x)) else 0
            )
        return df

    def create_features(df_pd, kcal_median=None, temp_median=None):
        """特徴量を作成する関数"""
        df_fe = df_pd.copy()

        # datetime列をパース
        df_fe['datetime'] = pd.to_datetime(df_fe['datetime'])

        # 日付関連特徴量
        df_fe['year'] = df_fe['datetime'].dt.year
        df_fe['month'] = df_fe['datetime'].dt.month
        df_fe['day'] = df_fe['datetime'].dt.day
        df_fe['dayofweek'] = df_fe['datetime'].dt.dayofweek
        df_fe['weekofyear'] = df_fe['datetime'].dt.isocalendar().week.astype(int)

        # 月初・月ハーフ・月末フラグ
        df_fe['is_beginning_of_month'] = (df_fe['day'] <= 10).astype(int)
        df_fe['is_middle_of_month'] = ((df_fe['day'] > 10) & (df_fe['day'] <= 20)).astype(int)
        df_fe['is_end_of_month'] = (df_fe['day'] > 20).astype(int)

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

        # 欠損値処理
        fill_kcal = kcal_median if kcal_median is not None else df_fe['kcal'].median()
        df_fe['kcal'] = df_fe['kcal'].fillna(fill_kcal)

        df_fe['precipitation'] = df_fe['precipitation'].replace('--', '0')
        df_fe['precipitation'] = pd.to_numeric(df_fe['precipitation'], errors='coerce').fillna(0)

        df_fe['temperature'] = pd.to_numeric(df_fe['temperature'], errors='coerce')
        fill_temp = temp_median if temp_median is not None else df_fe['temperature'].median()
        df_fe['temperature'] = df_fe['temperature'].fillna(fill_temp)

        df_fe['soldout'] = df_fe['soldout'].fillna(0).astype(int)
        df_fe['payday'] = df_fe['payday'].fillna(0).astype(float)
        df_fe['event'] = df_fe['event'].notna().astype(int)
        df_fe['remarks'] = df_fe['remarks'].notna().astype(int)

        # ===== 新規特徴量 =====

        # 交互作用特徴量
        df_fe['week_month'] = df_fe['dayofweek'] * df_fe['month']
        df_fe['week_payday'] = df_fe['dayofweek'] * df_fe['payday']
        df_fe['temp_month'] = df_fe['temperature'] * df_fe['month']

        # 金曜カレーフラグ（金曜日でカレーメニュー）
        df_fe['is_friday_curry'] = (
            (df_fe['dayofweek'] == 4) &
            (df_fe['name'].str.contains('カレー', na=False))
        ).astype(int)

        # 悪天候フラグ
        df_fe['is_bad_weather'] = df_fe['weather'].isin(['雨', '雪', '雷電']).astype(int)

        # 悪天候×低温
        df_fe['is_bad_weather_cold'] = (
            (df_fe['is_bad_weather'] == 1) &
            (df_fe['temperature'] < 15)
        ).astype(int)

        # 季節フラグ
        df_fe['is_summer'] = df_fe['month'].isin([7, 8]).astype(int)
        df_fe['is_winter'] = df_fe['month'].isin([12, 1, 2]).astype(int)

        # お楽しみメニューフラグ
        df_fe['is_special_menu'] = df_fe['remarks'].apply(
            lambda x: 1 if 'お楽しみ' in str(x) or 'スペシャル' in str(x) else 0
        ) if 'y' in df_fe.columns else 0  # 訓練データのみ

        # 時系列特徴量の計算（yが存在する場合）
        if 'y' in df_fe.columns:
            df_fe['lag_1'] = df_fe['y'].shift(1)
            df_fe['lag_5'] = df_fe['y'].shift(5)
            df_fe['rolling_mean_5'] = df_fe['y'].shift(1).rolling(window=5, min_periods=1).mean()
            df_fe['rolling_mean_10'] = df_fe['y'].shift(1).rolling(window=10, min_periods=1).mean()
            df_fe['rolling_std_5'] = df_fe['y'].shift(1).rolling(window=5, min_periods=1).std()
            df_fe['diff_1'] = df_fe['y'].diff(1)
        else:
            # テストデータ（yなし）では、NaNで埋める（後でpredict_recursive内で計算）
            df_fe['lag_1'] = np.nan
            df_fe['lag_5'] = np.nan
            df_fe['rolling_mean_5'] = np.nan
            df_fe['rolling_mean_10'] = np.nan
            df_fe['rolling_std_5'] = np.nan
            df_fe['diff_1'] = np.nan

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

    # 時系列特徴量を明示的に確認
    timeseries_features = ['lag_1', 'lag_5', 'rolling_mean_5', 'rolling_mean_10', 'rolling_std_5', 'diff_1']
    print(f"使用特徴量数: {len(feature_cols)}")
    print(f"時系列特徴量: {[f for f in timeseries_features if f in feature_cols]}")

    return df_train_fe, df_test_fe, feature_cols, train_kcal_median, train_temp_median


# ===== グループE: モデル構築 =====

@app.cell
def _(mo):
    mo.md("""
    ## 4. モデル構築と評価

    **LightGBM** を使用し、時系列データに適した **TimeSeriesSplit** で交差検証を行います。
    """)
    return


@app.cell
def _(
    df_train_fe,
    feature_cols,
    lgb,
    TimeSeriesSplit,
    mean_squared_error,
    np,
    mo
):
    # 学習データ
    X = df_train_fe[feature_cols]
    y = df_train_fe['y']

    # 時系列CVの設定
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # LightGBMパラメータ
    cv_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 7,
        'num_leaves': 31,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1,
        'random_state': 42
    }

    # CVループ
    oof_preds = np.zeros(len(X))
    cv_scores = []
    models = []

    print(f"Starting TimeSeriesSplit CV (n_splits={n_splits})...")

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMRegressor(**cv_params)
        
        # LightGBMのコールバックを使用して早期終了などを制御
        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0) # ログ出力を抑制
        ]

        model.fit(
            X_train_fold, 
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric='rmse',
            callbacks=callbacks
        )

        val_pred = model.predict(X_val_fold)
        oof_preds[val_index] = val_pred
        
        score = np.sqrt(mean_squared_error(y_val_fold, val_pred))
        cv_scores.append(score)
        models.append(model)
        
        print(f"Fold {fold+1} RMSE: {score:.4f}")

    # 全体スコア（検証データが存在する部分のみ）
    # TimeSeriesSplitでは最初のk個のデータは検証に使われないため、0以外の部分で計算
    valid_indices = np.where(oof_preds != 0)[0]
    overall_rmse = np.sqrt(mean_squared_error(y.iloc[valid_indices], oof_preds[valid_indices]))

    mo.md(f"""
    ### LightGBM Cross Validation 結果
    - **Overall RMSE**: {overall_rmse:.4f}
    - 各FoldのRMSE: {', '.join([f'{s:.2f}' for s in cv_scores])}
    """)
    
    return X, y, models, overall_rmse, cv_scores


@app.cell
def _(mo):
    # Optuna最適化の制御UI
    run_optuna = mo.ui.switch(value=False, label="Optunaで最適化を実行")
    n_trials_slider = mo.ui.slider(10, 100, value=30, step=10, label="試行回数")
    mo.hstack([run_optuna, n_trials_slider])
    return run_optuna, n_trials_slider


@app.cell
def _(
    run_optuna,
    n_trials_slider,
    df_train_fe,
    feature_cols,
    lgb,
    TimeSeriesSplit,
    mean_squared_error,
    np,
    optuna,
    mo
):
    # Optuna最適化（スイッチがONの場合のみ実行）
    if run_optuna.value:
        X_opt = df_train_fe[feature_cols]
        y_opt = df_train_fe['y']
        tscv_opt = TimeSeriesSplit(n_splits=5)

        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'verbosity': -1,
                'random_state': 42
            }

            cv_scores_opt = []
            for train_idx, val_idx in tscv_opt.split(X_opt):
                X_tr, X_val = X_opt.iloc[train_idx], X_opt.iloc[val_idx]
                y_tr, y_val = y_opt.iloc[train_idx], y_opt.iloc[val_idx]

                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(0)
                    ]
                )
                pred = model.predict(X_val)
                cv_scores_opt.append(np.sqrt(mean_squared_error(y_val, pred)))
            return np.mean(cv_scores_opt)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials_slider.value, show_progress_bar=True)

        best_params = study.best_params
        best_rmse = study.best_value

        mo.md(f"""
        ### Optuna 最適化結果
        - 試行回数: {n_trials_slider.value}
        - ベストRMSE: {best_rmse:.4f}
        - ベストパラメータ: {best_params}
        """)
    else:
        best_params = None
        mo.md("Optunaスイッチをオンにすると、ハイパーパラメータ最適化を実行します。")

    return best_params,


# ===== グループF: 予測と提出 =====

@app.cell
def _(overall_rmse, mo):
    mo.md(f"""
    ## 5. 最終モデル評価

    ### ベストモデル: LightGBM (TimeSeriesSplit CV)
    - **Cross Validation RMSE**: {overall_rmse:.4f}

    ※ 時系列分割交差検証の結果、上記の精度が得られました。全データで再学習させて提出用予測を作成します。
    """)
    return


@app.cell
def _(X, y, lgb, best_params, mo):
    # 全データでの再学習
    # Optunaで最適化した場合はbest_paramsを使用、そうでなければデフォルト
    default_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 1200,
        'learning_rate': 0.05,
        'max_depth': 7,
        'num_leaves': 31,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1,
        'random_state': 42
    }

    if best_params is not None:
        # Optunaの結果を使用
        final_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            **best_params  # Optunaで最適化されたパラメータを上書き
        }
        mo.md("Optunaで最適化されたパラメータを使用して最終モデルを学習中...")
    else:
        final_params = default_params
        mo.md("デフォルトパラメータを使用して最終モデルを学習中...")

    final_model = lgb.LGBMRegressor(**final_params)
    final_model.fit(X, y)

    print(f"使用パラメータ: {final_params}")

    return final_model,


@app.cell
def _(final_model, feature_cols, alt, pd, mo):
    # 特徴量重要度の可視化
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)

    chart_importance = alt.Chart(importance_df).mark_bar().encode(
        x=alt.X('importance:Q', title='重要度'),
        y=alt.Y('feature:N', sort='-x', title='特徴量'),
        color=alt.Color('importance:Q', scale=alt.Scale(scheme='blues'), legend=None),
        tooltip=['feature:N', 'importance:Q']
    ).properties(
        width=600,
        height=500,
        title='特徴量重要度 (Top 20)'
    )

    mo.md("### 特徴量重要度分析")
    chart_importance
    return chart_importance,


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

    return test_predictions


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


@app.cell
def _(df_test_fe, test_predictions, alt, pd, mo):
    # 予測結果の時系列可視化
    pred_viz_df = pd.DataFrame({
        'datetime': df_test_fe['datetime'],
        'predicted': test_predictions,
        'week': df_test_fe['week']
    })

    chart_pred = alt.Chart(pred_viz_df).mark_line(
        point=True, strokeWidth=2, color='steelblue'
    ).encode(
        x=alt.X('datetime:T', title='日付'),
        y=alt.Y('predicted:Q', title='予測販売数', scale=alt.Scale(zero=False)),
        tooltip=['datetime:T', 'predicted:Q', 'week:N']
    ).properties(
        width=800,
        height=300,
        title='テストデータ予測結果'
    ).interactive()

    mo.md("### 予測結果の可視化")
    chart_pred
    return chart_pred,


@app.cell
def _(test_predictions, np, mo):
    # 予測結果のサマリー統計
    pred_mean = np.mean(test_predictions)
    pred_std = np.std(test_predictions)
    pred_min = np.min(test_predictions)
    pred_max = np.max(test_predictions)

    mo.md(f"""
    ### 予測結果サマリー
    | 統計量 | 値 |
    |--------|-----|
    | 平均 | {pred_mean:.1f} 個 |
    | 標準偏差 | {pred_std:.1f} |
    | 最小 | {pred_min:.1f} 個 |
    | 最大 | {pred_max:.1f} 個 |
    | 予測日数 | {len(test_predictions)} 日 |
    """)
    return


if __name__ == "__main__":
    app.run()
