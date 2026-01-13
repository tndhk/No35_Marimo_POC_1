import marimo

__generated_with = "0.10.19"
app = marimo.App(width="medium")


# =============================================================================
# Phase 1: EDA (Exploratory Data Analysis)
# =============================================================================


@app.cell
def imports():
    """全モジュールのインポート"""
    import marimo as mo
    import polars as pl
    import pandas as pd
    import altair as alt
    import numpy as np
    from pathlib import Path
    from datetime import datetime

    # scikit-learn
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, root_mean_squared_error
    from sklearn.preprocessing import StandardScaler

    # Gradient Boosting
    import lightgbm as lgb
    import xgboost as xgb

    # Optuna
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    # 警告抑制
    import warnings
    warnings.filterwarnings('ignore')
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    return (
        mo, pl, pd, alt, np, Path, datetime,
        TimeSeriesSplit, cross_val_score, Ridge, Lasso,
        RandomForestRegressor, GradientBoostingRegressor,
        mean_squared_error, root_mean_squared_error, StandardScaler,
        lgb, xgb, optuna, TPESampler, MedianPruner, warnings
    )


@app.cell
def load_data(Path, pl):
    """データ読み込み"""
    data_dir = Path(__file__).parent.parent / "data"

    df_train = pl.read_csv(data_dir / "bento_train.csv")
    df_test = pl.read_csv(data_dir / "bento_test.csv")

    return df_train, df_test, data_dir


@app.cell
def title(mo, df_train, df_test):
    """タイトルと概要"""
    mo.md(f"""
    # Bento8 お弁当販売数予測

    ## 概要
    - 訓練データ: {df_train.shape[0]}行 x {df_train.shape[1]}列
    - テストデータ: {df_test.shape[0]}行 x {df_test.shape[1]}列
    - 評価指標: RMSE (Root Mean Squared Error)
    - 目標: 日々のお弁当販売数を予測

    ---
    """)
    return


@app.cell
def eda_header(mo):
    """EDAセクションヘッダー"""
    return mo.md("## Phase 1: EDA (Exploratory Data Analysis)")


@app.cell
def basic_stats(mo, df_train):
    """基本統計量"""
    mo.md("### 1.1 基本統計量")
    return mo.ui.table(df_train.describe())


@app.cell
def null_info(df_train, pl, mo):
    """欠損値確認"""
    null_counts = df_train.select([
        pl.col(col).null_count().alias(col)
        for col in df_train.columns
    ])

    null_df = pl.DataFrame({
        "column": df_train.columns,
        "null_count": null_counts.row(0),
        "null_rate": [n / len(df_train) * 100 for n in null_counts.row(0)]
    })

    return mo.vstack([
        mo.md("### 1.2 欠損値確認"),
        mo.ui.table(null_df)
    ])


@app.cell
def target_stats(df_train, mo):
    """ターゲット変数の統計"""
    y_stats = df_train.select([
        pl.col("y").mean().alias("mean"),
        pl.col("y").std().alias("std"),
        pl.col("y").min().alias("min"),
        pl.col("y").max().alias("max"),
        pl.col("y").median().alias("median")
    ])

    return mo.vstack([
        mo.md("### 1.3 ターゲット変数 (y) の統計"),
        mo.ui.table(y_stats)
    ])


@app.cell
def to_pandas(df_train, df_test):
    """Pandas変換（可視化用）"""
    df_train_pandas = df_train.to_pandas()
    df_test_pandas = df_test.to_pandas()

    # datetime列をdatetime型に変換
    df_train_pandas['datetime'] = pd.to_datetime(df_train_pandas['datetime'])
    df_test_pandas['datetime'] = pd.to_datetime(df_test_pandas['datetime'])

    return df_train_pandas, df_test_pandas


@app.cell
def y_distribution(df_train_pandas, alt, mo):
    """販売数の分布"""
    chart = alt.Chart(df_train_pandas).mark_bar().encode(
        alt.X('y:Q', bin=alt.Bin(maxbins=20), title='販売数'),
        alt.Y('count()', title='頻度')
    ).properties(
        title='販売数の分布',
        width=500,
        height=300
    )

    return mo.vstack([
        mo.md("### 1.4 販売数の分布"),
        mo.ui.altair_chart(chart)
    ])


@app.cell
def week_stats(df_train, pl, mo):
    """曜日別集計"""
    week_agg = df_train.group_by("week").agg([
        pl.col("y").mean().alias("mean_y"),
        pl.col("y").std().alias("std_y"),
        pl.col("y").count().alias("count")
    ]).sort("week")

    return mo.vstack([
        mo.md("### 1.5 曜日別販売数"),
        mo.ui.table(week_agg)
    ])


@app.cell
def weather_stats(df_train, pl, mo):
    """天気別集計"""
    weather_agg = df_train.group_by("weather").agg([
        pl.col("y").mean().alias("mean_y"),
        pl.col("y").std().alias("std_y"),
        pl.col("y").count().alias("count")
    ]).sort("mean_y", descending=True)

    return mo.vstack([
        mo.md("### 1.6 天気別販売数"),
        mo.ui.table(weather_agg)
    ])


@app.cell
def timeseries_plot(df_train_pandas, alt, mo):
    """時系列プロット"""
    chart = alt.Chart(df_train_pandas).mark_line(point=True).encode(
        alt.X('datetime:T', title='日付'),
        alt.Y('y:Q', title='販売数'),
        tooltip=['datetime:T', 'y:Q', 'weather:N']
    ).properties(
        title='販売数の時系列推移',
        width=700,
        height=300
    )

    return mo.vstack([
        mo.md("### 1.7 時系列プロット"),
        mo.ui.altair_chart(chart)
    ])


@app.cell
def correlation_analysis(df_train_pandas, alt, mo, np):
    """相関分析"""
    # 数値列のみ抽出
    numeric_cols = ['y', 'kcal', 'temperature']
    df_numeric = df_train_pandas[numeric_cols].dropna()

    corr_matrix = df_numeric.corr()

    # ヒートマップ用データ整形
    corr_data = []
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            corr_data.append({
                'var1': col1,
                'var2': col2,
                'correlation': corr_matrix.loc[col1, col2]
            })

    corr_df = pd.DataFrame(corr_data)

    chart = alt.Chart(corr_df).mark_rect().encode(
        x='var1:N',
        y='var2:N',
        color=alt.Color('correlation:Q', scale=alt.Scale(scheme='blueorange', domain=[-1, 1])),
        tooltip=['var1', 'var2', 'correlation']
    ).properties(
        title='相関行列',
        width=300,
        height=300
    )

    return mo.vstack([
        mo.md("### 1.8 相関分析"),
        mo.md(f"- 気温と販売数の相関: {corr_matrix.loc['y', 'temperature']:.3f}"),
        mo.md(f"- カロリーと販売数の相関: {corr_matrix.loc['y', 'kcal']:.3f}"),
        mo.ui.altair_chart(chart)
    ])


@app.cell
def eda_summary(mo):
    """EDAの知見まとめ"""
    return mo.md("""
    ### 1.9 EDAの知見まとめ

    - 販売数は約90〜170個の範囲で推移
    - 気温と販売数には負の相関がある（暑いと売れにくい）
    - 曜日によって販売数に差がある
    - 天気も販売数に影響（悪天候で減少傾向）
    - kcal（カロリー）に欠損値あり → 訓練データの中央値で補完

    ---
    """)


# =============================================================================
# Phase 2: Feature Engineering
# =============================================================================


@app.cell
def fe_header(mo):
    """Feature Engineeringセクションヘッダー"""
    return mo.md("## Phase 2: Feature Engineering")


@app.cell
def menu_keywords():
    """メニューキーワード定義"""
    MENU_KEYWORDS = {
        'カレー': ['カレー', 'curry'],
        '唐揚': ['唐揚', 'から揚', '竜田'],
        'ハンバーグ': ['ハンバーグ', 'バーグ'],
        '魚': ['鮭', 'サバ', 'さば', '鯖', 'ブリ', 'アジ', '魚'],
        '肉': ['豚', '鶏', '牛', 'チキン', 'ポーク'],
        'フライ': ['フライ', '揚げ', 'コロッケ', 'カツ'],
        'お楽しみ': ['お楽しみ', 'おたのしみ']
    }
    return MENU_KEYWORDS,


@app.cell
def feature_creator(MENU_KEYWORDS, pd, np):
    """特徴量作成関数（データリーク防止版）"""

    def create_features(df, train_stats=None):
        """
        特徴量を作成する関数

        Parameters:
        -----------
        df : pd.DataFrame
            入力データフレーム
        train_stats : dict, optional
            訓練データから計算した統計量（テストデータ用）
            None の場合は df から計算（訓練データ用）

        Returns:
        --------
        pd.DataFrame : 特徴量を追加したデータフレーム
        dict : 計算した統計量（訓練データの場合）
        """
        df = df.copy()

        # 統計量の計算または適用
        if train_stats is None:
            # 訓練データ: 統計量を計算
            stats = {
                'kcal_median': df['kcal'].median(),
                'temp_median': df['temperature'].median(),
            }
        else:
            # テストデータ: 訓練データの統計量を使用
            stats = train_stats

        # 欠損値補完（データリーク防止: trainの統計量を使用）
        df['kcal_filled'] = df['kcal'].fillna(stats['kcal_median'])
        df['temp_filled'] = df['temperature'].fillna(stats['temp_median'])

        # 日付特徴量
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['weekofyear'] = df['datetime'].dt.isocalendar().week.astype(int)
        df['is_friday'] = (df['dayofweek'] == 4).astype(int)

        # 月の周期性
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # 曜日ワンホット
        week_dummies = pd.get_dummies(df['week'], prefix='week')
        for col in ['week_月', 'week_火', 'week_水', 'week_木', 'week_金']:
            if col not in week_dummies.columns:
                week_dummies[col] = 0
        df = pd.concat([df, week_dummies[['week_月', 'week_火', 'week_水', 'week_木', 'week_金']]], axis=1)

        # 天気特徴量
        weather_categories = ['快晴', '晴れ', '曇', '薄曇', '雨', '雪', '雷電']
        for cat in weather_categories:
            df[f'weather_{cat}'] = (df['weather'] == cat).astype(int)

        # 悪天候フラグ
        df['is_bad_weather'] = df['weather'].isin(['雨', '雪', '雷電']).astype(int)

        # 降水量の数値化
        df['precipitation_num'] = pd.to_numeric(df['precipitation'], errors='coerce').fillna(0)

        # メニュー関連特徴量
        df['name_str'] = df['name'].fillna('').astype(str)
        for menu_name, keywords in MENU_KEYWORDS.items():
            df[f'menu_{menu_name}'] = df['name_str'].apply(
                lambda x: int(any(kw in x for kw in keywords))
            )

        # remarks特徴量
        df['has_remarks'] = df['remarks'].notna().astype(int)

        # イベント・給料日フラグ
        df['event_flag'] = (df['event'].notna() & (df['event'] != '')).astype(int)
        df['payday_flag'] = df['payday'].fillna(0).astype(int)

        # soldoutフラグ（前日完売の影響）
        df['soldout_flag'] = df['soldout'].fillna(0).astype(int)

        return df, stats

    return create_features,


@app.cell
def train_statistics(df_train_pandas, create_features):
    """訓練データ統計量の計算"""
    df_train_fe, train_stats = create_features(df_train_pandas, train_stats=None)
    return df_train_fe, train_stats


@app.cell
def test_features(df_test_pandas, create_features, train_stats):
    """テストデータに特徴量適用（データリーク防止）"""
    df_test_fe, _ = create_features(df_test_pandas, train_stats=train_stats)
    return df_test_fe,


@app.cell
def feature_columns():
    """使用する特徴量カラムの定義"""
    FEATURE_COLS = [
        # 日付関連
        'month', 'dayofweek', 'is_friday', 'month_sin', 'month_cos',
        # 曜日ワンホット
        'week_月', 'week_火', 'week_水', 'week_木', 'week_金',
        # 天気関連
        'weather_快晴', 'weather_晴れ', 'weather_曇', 'weather_薄曇',
        'weather_雨', 'weather_雪', 'is_bad_weather',
        # 数値
        'kcal_filled', 'temp_filled', 'precipitation_num',
        # メニュー
        'menu_カレー', 'menu_唐揚', 'menu_ハンバーグ', 'menu_魚',
        'menu_肉', 'menu_フライ', 'menu_お楽しみ',
        # その他
        'has_remarks', 'event_flag', 'payday_flag', 'soldout_flag'
    ]
    return FEATURE_COLS,


@app.cell
def prepare_xy(df_train_fe, FEATURE_COLS, np):
    """学習データの準備"""
    # 存在する特徴量のみ使用
    available_cols = [col for col in FEATURE_COLS if col in df_train_fe.columns]

    X_train = df_train_fe[available_cols].values
    y_train = df_train_fe['y'].values

    return X_train, y_train, available_cols


@app.cell
def fe_summary(mo, available_cols):
    """特徴量エンジニアリングのまとめ"""
    return mo.md(f"""
    ### 2.1 特徴量まとめ

    - 使用特徴量数: {len(available_cols)}
    - 主な特徴量カテゴリ:
      - 日付関連: month, dayofweek, is_friday, 周期性
      - 曜日ワンホット: week_月 〜 week_金
      - 天気関連: 天気ワンホット, is_bad_weather
      - 数値: kcal_filled, temp_filled, precipitation_num
      - メニュー: menu_カレー, menu_唐揚, etc.
      - その他: has_remarks, event_flag, payday_flag

    ---
    """)


# =============================================================================
# Phase 3: Model Selection & Optimization
# =============================================================================


@app.cell
def model_header(mo):
    """モデル選択セクションヘッダー"""
    return mo.md("## Phase 3: Model Selection & Optimization")


@app.cell
def baseline_models_eval(X_train, y_train, TimeSeriesSplit, np,
                          Ridge, RandomForestRegressor, GradientBoostingRegressor,
                          lgb, xgb, root_mean_squared_error, mo, pd):
    """5モデルベースライン評価"""

    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbosity=0)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            cv_scores.append(rmse)

        mean_rmse = np.mean(cv_scores)
        std_rmse = np.std(cv_scores)
        results.append({
            'Model': name,
            'CV_RMSE_Mean': mean_rmse,
            'CV_RMSE_Std': std_rmse
        })

        # 全データで再学習
        model.fit(X_train, y_train)
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values('CV_RMSE_Mean')

    return results_df, trained_models, tscv


@app.cell
def cv_results_display(mo, results_df):
    """CV結果の表示"""
    return mo.vstack([
        mo.md("### 3.1 ベースラインモデル評価結果"),
        mo.ui.table(results_df)
    ])


@app.cell
def cv_results_chart(results_df, alt, mo):
    """CV結果の可視化"""
    chart = alt.Chart(results_df).mark_bar().encode(
        x=alt.X('Model:N', sort=alt.EncodingSortField(field='CV_RMSE_Mean', order='ascending')),
        y=alt.Y('CV_RMSE_Mean:Q', title='CV RMSE'),
        color=alt.Color('Model:N', legend=None),
        tooltip=['Model', 'CV_RMSE_Mean', 'CV_RMSE_Std']
    ).properties(
        title='モデル別 CV RMSE 比較',
        width=500,
        height=300
    )

    return mo.ui.altair_chart(chart)


@app.cell
def optuna_switch(mo):
    """Optuna最適化スイッチ"""
    optuna_enabled = mo.ui.switch(label="Optuna最適化を実行", value=False)
    return optuna_enabled,


@app.cell
def optuna_config(optuna_enabled, mo):
    """Optuna設定の表示"""
    if optuna_enabled.value:
        return mo.md("""
        ### 3.2 Optuna最適化

        上位3モデル（LightGBM, XGBoost, GradientBoosting）に対して
        ハイパーパラメータ最適化を実行します。

        - 試行回数: 30回
        - 枝刈り: MedianPruner
        """)
    else:
        return mo.md("### 3.2 Optuna最適化\n\n*スイッチをONにすると最適化を実行します*")


@app.cell
def optuna_optimization(optuna_enabled, X_train, y_train, tscv,
                         lgb, xgb, GradientBoostingRegressor,
                         optuna, TPESampler, MedianPruner,
                         root_mean_squared_error, np, pd, mo):
    """Optuna最適化の実行"""

    if not optuna_enabled.value:
        return mo.md(""), {}, pd.DataFrame()

    def objective_lgbm(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'random_state': 42,
            'verbose': -1
        }

        model = lgb.LGBMRegressor(**params)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            cv_scores.append(rmse)

        return np.mean(cv_scores)

    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 42,
            'verbosity': 0
        }

        model = xgb.XGBRegressor(**params)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            cv_scores.append(rmse)

        return np.mean(cv_scores)

    # LightGBM最適化
    study_lgbm = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner()
    )
    study_lgbm.optimize(objective_lgbm, n_trials=30, show_progress_bar=False)

    # XGBoost最適化
    study_xgb = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner()
    )
    study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=False)

    # 最適パラメータでモデル学習
    best_lgbm = lgb.LGBMRegressor(**study_lgbm.best_params, random_state=42, verbose=-1)
    best_lgbm.fit(X_train, y_train)

    best_xgb = xgb.XGBRegressor(**study_xgb.best_params, random_state=42, verbosity=0)
    best_xgb.fit(X_train, y_train)

    optimized_models = {
        'LightGBM_optimized': best_lgbm,
        'XGBoost_optimized': best_xgb
    }

    optuna_results = pd.DataFrame([
        {'Model': 'LightGBM', 'Best_CV_RMSE': study_lgbm.best_value},
        {'Model': 'XGBoost', 'Best_CV_RMSE': study_xgb.best_value}
    ])

    return mo.vstack([
        mo.md("### Optuna最適化結果"),
        mo.ui.table(optuna_results)
    ]), optimized_models, optuna_results


@app.cell
def select_best_model(optuna_enabled, results_df, trained_models, optimized_models, mo):
    """最良モデルの選択"""

    if optuna_enabled.value and optimized_models:
        # Optuna最適化済みモデルから選択
        best_model_name = 'LightGBM_optimized'
        best_model = optimized_models.get('LightGBM_optimized')
    else:
        # ベースラインから最良モデルを選択
        best_model_name = results_df.iloc[0]['Model']
        best_model = trained_models[best_model_name]

    return mo.md(f"""
    ### 3.3 最良モデル

    選択されたモデル: {best_model_name}
    """), best_model, best_model_name


# =============================================================================
# Phase 4: Submission
# =============================================================================


@app.cell
def submit_header(mo):
    """提出セクションヘッダー"""
    return mo.md("## Phase 4: Submission")


@app.cell
def test_prediction(best_model, df_test_fe, available_cols, np):
    """テストデータの予測"""

    # 存在する特徴量のみ使用
    X_test = df_test_fe[available_cols].values

    # 予測
    predictions = best_model.predict(X_test)

    # 整数に丸める
    predictions = np.round(predictions).astype(int)

    # 負の値を0にクリップ
    predictions = np.maximum(predictions, 0)

    return predictions, X_test


@app.cell
def prediction_summary(predictions, mo, np):
    """予測結果のサマリー"""
    return mo.md(f"""
    ### 4.1 予測結果サマリー

    - 予測件数: {len(predictions)}
    - 予測値の範囲: {predictions.min()} 〜 {predictions.max()}
    - 予測値の平均: {np.mean(predictions):.1f}
    - 予測値の標準偏差: {np.std(predictions):.1f}
    """)


@app.cell
def submission_formatter(df_test_fe, predictions, pd):
    """提出ファイルのフォーマット"""

    # 日付をyyyy-m-d形式に変換
    dates = df_test_fe['datetime'].dt.strftime('%Y-%-m-%-d')

    submission_df = pd.DataFrame({
        'datetime': dates,
        'y': predictions
    })

    return submission_df,


@app.cell
def submission_save(submission_df, data_dir, mo):
    """提出ファイルの保存"""

    output_path = data_dir.parent / "submission.csv"
    submission_df.to_csv(output_path, index=False, header=False)

    return mo.md(f"""
    ### 4.2 提出ファイル保存完了

    保存先: `{output_path}`
    """), output_path


@app.cell
def submission_preview(submission_df, mo):
    """提出ファイルのプレビュー"""
    return mo.vstack([
        mo.md("### 4.3 提出ファイルプレビュー（先頭10行）"),
        mo.ui.table(submission_df.head(10))
    ])


@app.cell
def final_summary(mo, best_model_name, results_df):
    """最終サマリー"""
    best_rmse = results_df.iloc[0]['CV_RMSE_Mean']

    return mo.md(f"""
    ---

    ## 最終結果

    - 使用モデル: {best_model_name}
    - CV RMSE: {best_rmse:.4f}
    - 提出ファイル: submission.csv（40行）

    ### 次のステップ

    1. SIGNATEに提出してLBスコアを確認
    2. CV-LB乖離を分析
    3. 必要に応じて特徴量追加・モデル調整
    """)


if __name__ == "__main__":
    app.run()
