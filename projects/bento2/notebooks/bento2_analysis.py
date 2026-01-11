import marimo

__generated_with = "0.19.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import numpy as np
    import altair as alt
    import jpholiday
    from pathlib import Path
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit
    # import lightgbm as lgb
    import optuna
    return (
        HistGradientBoostingRegressor,
        Path,
        RandomForestRegressor,
        Ridge,
        StandardScaler,
        TimeSeriesSplit,
        jpholiday,
        mean_squared_error,
        mo,
        np,
        optuna,
        pd,
        pl,
    )


@app.cell
def _(Path, mo, pl):
    mo.md("# Bento2 お弁当販売数予測 - 高精度モデル構築")

    # データ読み込み
    data_dir = Path("/Users/takahiko_tsunoda/work/dev/No35_marimo_poc1/projects/bento2/data")
    train_path = data_dir / "bento_train.csv"
    test_path = data_dir / "bento_test.csv"

    df_train_raw = pl.read_csv(train_path)
    df_test_raw = pl.read_csv(test_path)

    mo.md(f"訓練データサイズ: {df_train_raw.shape}")
    return df_test_raw, df_train_raw


@app.cell
def _(df_train_raw, mo, pl):
    # 基本EDA
    mo.md("## 1. データ概要")

    null_counts = df_train_raw.null_count()
    total_rows = len(df_train_raw)

    null_info = pl.DataFrame({
        "カラム": list(null_counts.columns),
        "欠損数": [null_counts[col][0] for col in null_counts.columns],
        "欠損率(%)": [round(null_counts[col][0] * 100 / total_rows, 2) for col in null_counts.columns]
    }).filter(pl.col("欠損数") > 0)

    mo.md("### 欠損値があるカラム")
    return (null_info,)


@app.cell
def _(null_info):
    null_info
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. フェーズ 1: シンプルベースライン (RMSE 13目標)
    """)
    return


@app.cell
def _(pd):
    def preprocess_simple(df_pl):
        df = df_pl.to_pandas()
        df['datetime'] = pd.to_datetime(df['datetime'])

        # コア4特徴量
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
        df['is_friday'] = (df['datetime'].dt.dayofweek == 4).astype(int)

        # 欠損値埋め（シンプルに中央値）
        df['temperature'] = df['temperature'].fillna(df['temperature'].median())

        return df
    return (preprocess_simple,)


@app.cell
def _(df_test_raw, df_train_raw, preprocess_simple):
    train_simple = preprocess_simple(df_train_raw)
    test_simple = preprocess_simple(df_test_raw)

    features_v1 = ['year', 'month', 'temperature', 'is_friday']
    return features_v1, train_simple


@app.cell
def _(
    Ridge,
    StandardScaler,
    TimeSeriesSplit,
    features_v1,
    mean_squared_error,
    mo,
    np,
    train_simple,
):
    # シンプルなRidge回帰での評価
    X_v1 = train_simple[features_v1]
    y_v1 = train_simple['y']

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores_v1 = []

    for train_idx_v1, val_idx_v1 in tscv.split(X_v1):
        X_tr_v1, X_val_v1 = X_v1.iloc[train_idx_v1], X_v1.iloc[val_idx_v1]
        y_tr_v1, y_val_v1 = y_v1.iloc[train_idx_v1], y_v1.iloc[val_idx_v1]

        # 線形モデルの場合は標準化が重要
        scaler_v1 = StandardScaler()
        X_tr_scaled_v1 = scaler_v1.fit_transform(X_tr_v1)
        X_val_scaled_v1 = scaler_v1.transform(X_val_v1)

        model_v1 = Ridge(alpha=1.0)
        model_v1.fit(X_tr_scaled_v1, y_tr_v1)
        pred_v1 = model_v1.predict(X_val_scaled_v1)
        cv_scores_v1.append(np.sqrt(mean_squared_error(y_val_v1, pred_v1)))

    mean_rmse_v1 = np.mean(cv_scores_v1)
    print(f'Phase 1 (Ridge) Mean CV RMSE: {mean_rmse_v1:.4f}')
    mo.md(f"### フェーズ1 (Ridge) CV RMSE: {mean_rmse_v1:.4f}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. フェーズ 2: 特徴量実験 (RMSE 11目標)
    """)
    return


@app.cell
def _(jpholiday, pd, re):
    def extract_keywords(name):
        res = []
        if re.search(r'カレー', str(name)): res.append('curry')
        if re.search(r'フライ|カツ|唐揚げ', str(name)): res.append('fry')
        if re.search(r'ハンバーグ', str(name)): res.append('hamburg')
        return res

    def preprocess_advanced(df_pl):
        import re
        df = df_pl.to_pandas()
        df['datetime'] = pd.to_datetime(df['datetime'])

        # 基本特徴量
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['is_friday'] = (df['dayofweek'] == 4).astype(int)

        # 祝日特徴量 (jpholiday)
        df['is_holiday'] = df['datetime'].apply(lambda x: 1 if jpholiday.is_holiday(x) else 0)
        df['is_next_holiday'] = df['datetime'].apply(lambda x: 1 if jpholiday.is_holiday(x + pd.Timedelta(days=1)) else 0)

        # 気象
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
        df['temperature'] = df['temperature'].fillna(df['temperature'].median())
        df['precipitation'] = df['precipitation'].replace('--', '0')
        df['precipitation'] = pd.to_numeric(df['precipitation'], errors='coerce').fillna(0)
        df['is_bad_weather'] = (df['precipitation'] > 0).astype(int)

        # その他フラグ
        df['payday'] = df['payday'].fillna(0).astype(int)
        df['event'] = df['event'].notna().astype(int)

        # メニュー
        df['is_curry'] = df['name'].apply(lambda x: 1 if r'カレー' in str(x) else 0)
        df['is_special'] = df['remarks'].apply(lambda x: 1 if re.search(r'お楽しみ|スペシャル|手作り', str(x)) else 0)

        # フェーズ3/4/5用: ラグ特徴量の初期化
        # 訓練データは真値から計算できるが、テストデータは空または仮の値
        if 'y' in df.columns:
            df['lag_1'] = df['y'].shift(1).fillna(df['y'].mean())
            df['lag_7'] = df['y'].shift(7).fillna(df['y'].mean())
            df['rolling_mean_5'] = df['y'].shift(1).rolling(window=5, min_periods=1).mean().fillna(df['y'].mean())
        else:
            df['lag_1'] = 0.0
            df['lag_7'] = 0.0
            df['rolling_mean_5'] = 0.0

        return df
    return (preprocess_advanced,)


@app.cell
def _(df_test_raw, df_train_raw, preprocess_advanced):
    train_adv = preprocess_advanced(df_train_raw)
    test_adv = preprocess_advanced(df_test_raw)

    features_v2 = [
        'year', 'month', 'temperature', 'is_friday', 
        'dayofweek', 'is_curry', 'is_bad_weather', 'event', 'payday',
        'is_holiday', 'is_next_holiday'
    ]
    return features_v2, test_adv, train_adv


@app.cell
def _(
    Ridge,
    StandardScaler,
    TimeSeriesSplit,
    features_v2,
    mean_squared_error,
    mo,
    np,
    train_adv,
):
    # 特徴量追加後のRidge回帰での評価
    X_v2 = train_adv[features_v2]
    y_v2 = train_adv['y']

    tscv_v2 = TimeSeriesSplit(n_splits=5)
    cv_scores_v2 = []

    for train_idx_v2, val_idx_v2 in tscv_v2.split(X_v2):
        X_tr_v2, X_val_v2 = X_v2.iloc[train_idx_v2], X_v2.iloc[val_idx_v2]
        y_tr_v2, y_val_v2 = y_v2.iloc[train_idx_v2], y_v2.iloc[val_idx_v2]

        scaler_v2 = StandardScaler()
        X_tr_scaled_v2 = scaler_v2.fit_transform(X_tr_v2)
        X_val_scaled_v2 = scaler_v2.transform(X_val_v2)

        model_v2_fold = Ridge(alpha=1.0)
        model_v2_fold.fit(X_tr_scaled_v2, y_tr_v2)

        pred_v2 = model_v2_fold.predict(X_val_scaled_v2)
        cv_scores_v2.append(np.sqrt(mean_squared_error(y_val_v2, pred_v2)))

    mean_rmse_v2 = np.mean(cv_scores_v2)
    print(f'Phase 2 (Ridge) Mean CV RMSE: {mean_rmse_v2:.4f}')
    mo.md(f"### フェーズ2 (Ridge) CV RMSE: {mean_rmse_v2:.4f}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. フェーズ 3: 高度なモデル (RandomForest + ラグ特徴量)
    """)
    return


@app.cell
def _(
    RandomForestRegressor,
    TimeSeriesSplit,
    features_v2,
    mean_squared_error,
    mo,
    np,
    train_adv,
):
    # フェーズ3: RandomForestでの評価
    # ラグ特徴量を特徴量リストに加える
    features_v3 = features_v2 + ['lag_1', 'rolling_mean_5']
    X_v3 = train_adv[features_v3]
    y_v3 = train_adv['y']

    tscv_v3 = TimeSeriesSplit(n_splits=5)
    cv_scores_v3 = []

    for train_idx_v3, val_idx_v3 in tscv_v3.split(X_v3):
        X_tr_v3, X_val_v3 = X_v3.iloc[train_idx_v3], X_v3.iloc[val_idx_v3]
        y_tr_v3, y_val_v3 = y_v3.iloc[train_idx_v3], y_v3.iloc[val_idx_v3]

        # 線形ではないので標準化なしでもOK
        model_v3_fold = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model_v3_fold.fit(X_tr_v3, y_tr_v3)

        pred_v3 = model_v3_fold.predict(X_val_v3)
        cv_scores_v3.append(np.sqrt(mean_squared_error(y_val_v3, pred_v3)))

    mean_rmse_v3 = np.mean(cv_scores_v3)
    print(f'Phase 3 (RF) Mean CV RMSE: {mean_rmse_v3:.4f}')
    mo.md(f"### フェーズ3 (RandomForest) CV RMSE: {mean_rmse_v3:.4f}")
    return (features_v3,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. 予測と提出ファイルの生成
    """)
    return


@app.cell
def _(RandomForestRegressor, features_v3, test_adv, train_adv):
    # 最終モデルを全データで学習 (フェーズ3のセットを使用)
    X_final = train_adv[features_v3]
    y_final = train_adv['y']

    final_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    final_model.fit(X_final, y_final)

    # テストデータ予測
    # 注: テストデータのラグは簡易的に訓練データ末尾の値で埋める
    X_test = test_adv[features_v3].copy()
    X_test['lag_1'] = y_final.iloc[-1]
    X_test['rolling_mean_5'] = y_final.tail(5).mean()

    test_preds = final_model.predict(X_test)
    return (test_preds,)


@app.cell
def _(Path, pd, test_adv, test_preds):
    # 提出用データの作成
    submission = pd.DataFrame({
        'datetime': test_adv['datetime'].dt.strftime('%Y-%-m-%-d'),
        'prediction': test_preds
    })

    sub_path = Path("/Users/takahiko_tsunoda/work/dev/No35_marimo_poc1/projects/bento2/data/submission.csv")
    submission.to_csv(sub_path, index=False, header=False)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. 予測と提出ファイルの生成 (フェーズ 4 アルティメット)
    """)
    return


@app.cell
def _(
    HistGradientBoostingRegressor,
    TimeSeriesSplit,
    features_v2,
    mean_squared_error,
    mo,
    np,
    optuna,
    pd,
    test_adv,
    train_adv,
):
    # フェーズ4: HistGradientBoostingRegressor + Optuna + 再帰的予測

    features_v4 = features_v2 + ['lag_1', 'rolling_mean_5']
    X_v4 = train_adv[features_v4]
    y_v4 = train_adv['y']

    # --- Optuna で最適化 ---
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 1.0, log=True),
            'random_state': 42
        }

        tscv_opt = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for tr_idx, val_idx in tscv_opt.split(X_v4):
            X_tr, X_val = X_v4.iloc[tr_idx], X_v4.iloc[val_idx]
            y_tr, y_val = y_v4.iloc[tr_idx], y_v4.iloc[val_idx]

            model = HistGradientBoostingRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            cv_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

        return np.mean(cv_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value

    print(f"Phase 4 (HistGBM+Optuna) Best CV RMSE: {best_score:.4f}")
    mo.md(f"### フェーズ4 (HistGBM+Optuna) Best CV RMSE: {best_score:.4f}")
    mo.md(f"Best Params: {best_params}")

    # --- 最終モデル学習 ---
    final_model_v4 = HistGradientBoostingRegressor(**best_params)
    final_model_v4.fit(X_v4, y_v4)

    # --- 再帰的予測ロジック ---
    # テストデータを1行ずつ予測し、結果を次のlag特徴量として使う

    # 訓練データの最後のy値と移動平均計算用のバッファ
    y_history_v4 = list(y_v4.values)

    test_preds_v4 = []

    # テストデータの各行について
    for i_v4 in range(len(test_adv)):
        row_v4 = test_adv.iloc[i_v4].copy()

        # ラグ特徴量を現在のhistoryから計算
        # lag_1
        last_val_v4 = y_history_v4[-1]

        # rolling_mean_5 (直近5個の平均)
        last_5_vals_v4 = y_history_v4[-5:]
        rolling_mean_5_v4 = np.mean(last_5_vals_v4)

        # データフレーム内の値を更新しないで、モデル入力用の配列を直接作る方が安全だが
        # ここではわかりやすくrowの特徴量を上書きする
        # 注意: pandasの行コピーに対する操作

        # 特徴量ベクトル作成 (features_v4の順序を守る)
        # そのためにデータフレームを作成
        current_input_df_v4 = pd.DataFrame([row_v4]) # 1行のDF
        current_input_df_v4['lag_1'] = last_val_v4
        current_input_df_v4['rolling_mean_5'] = rolling_mean_5_v4

        # 必要なカラムだけ抽出
        X_current_v4 = current_input_df_v4[features_v4]

        # 予測
        pred_val_v4 = final_model_v4.predict(X_current_v4)[0]

        # 負の値はあり得ないのでクリップ
        if pred_val_v4 < 0: pred_val_v4 = 0

        test_preds_v4.append(pred_val_v4)
        y_history_v4.append(pred_val_v4) # 予測値を履歴に追加して次回のラグ計算に使う
    return (test_preds_v4,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. 予測と提出ファイルの生成 (フェーズ 5 起死回生)
    """)
    return


@app.cell
def _(
    RandomForestRegressor,
    TimeSeriesSplit,
    features_v2,
    mean_squared_error,
    mo,
    np,
    optuna,
    pd,
    test_adv,
    train_adv,
):
    # フェーズ5: RandomForest + Optuna + 再帰的予測 + Lag7 + Special

    features_v5 = features_v2 + ['lag_1', 'lag_7', 'rolling_mean_5', 'is_special']
    X_v5 = train_adv[features_v5]
    y_v5 = train_adv['y']

    # --- Optuna で最適化 ---
    def objective_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'random_state': 42
        }

        tscv_opt = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for tr_idx, val_idx in tscv_opt.split(X_v5):
            X_tr, X_val = X_v5.iloc[tr_idx], X_v5.iloc[val_idx]
            y_tr, y_val = y_v5.iloc[tr_idx], y_v5.iloc[val_idx]

            model = RandomForestRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            cv_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

        return np.mean(cv_scores)

    study_v5 = optuna.create_study(direction='minimize')
    study_v5.optimize(objective_rf, n_trials=50, show_progress_bar=True)

    best_params_v5 = study_v5.best_params
    best_score_v5 = study_v5.best_value

    print(f"Phase 5 (RF+Optuna) Best CV RMSE: {best_score_v5:.4f}")
    mo.md(f"### フェーズ5 (RF+Optuna) Best CV RMSE: {best_score_v5:.4f}")
    mo.md(f"Best Params: {best_params_v5}")

    # --- 最終モデル学習 ---
    final_model_v5 = RandomForestRegressor(**best_params_v5)
    final_model_v5.fit(X_v5, y_v5)

    # --- 再帰的予測ロジック ---
    y_history_v5 = list(y_v5.values)
    test_preds_v5 = []

    for i_v5 in range(len(test_adv)):
        row_v5 = test_adv.iloc[i_v5].copy()

        # historyから特徴量計算
        last_val_v5 = y_history_v5[-1]

        # lag_7 (7日前, ない場合はとりあえず平均や直近で埋めるが、historyは十分長いはず)
        # テストデータの最初は訓練データの末尾に続くのでindex指定に注意
        if len(y_history_v5) >= 7:
            val_lag7_v5 = y_history_v5[-7]
        else:
            val_lag7_v5 = np.mean(y_history_v5) # fallback

        last_5_vals_v5 = y_history_v5[-5:]
        rolling_mean_5_v5 = np.mean(last_5_vals_v5)

        current_input_df_v5 = pd.DataFrame([row_v5])
        current_input_df_v5['lag_1'] = last_val_v5
        current_input_df_v5['lag_7'] = val_lag7_v5
        current_input_df_v5['rolling_mean_5'] = rolling_mean_5_v5
        # is_specialはpreprocessですでに計算済み

        X_current_v5 = current_input_df_v5[features_v5]

        pred_val_v5 = final_model_v5.predict(X_current_v5)[0]
        if pred_val_v5 < 0: pred_val_v5 = 0

        test_preds_v5.append(pred_val_v5)
        y_history_v5.append(pred_val_v5)
    return (test_preds_v5,)


@app.cell
def _(Path, pd, test_adv, test_preds_v4):
    # 提出用データの作成 (フェーズ4)
    submission_v4 = pd.DataFrame({
        'datetime': test_adv['datetime'].dt.strftime('%Y-%-m-%-d'),
        'prediction': test_preds_v4
    })

    sub_path_v4 = Path("/Users/takahiko_tsunoda/work/dev/No35_marimo_poc1/projects/bento2/data/submission_v4.csv")
    submission_v4.to_csv(sub_path_v4, index=False, header=False)
    return (sub_path_v4,)


@app.cell
def _(mo, sub_path_v4):
    mo.md(f"""
    提出ファイル(v4)を作成しました: `{sub_path_v4}`
    """)
    return


@app.cell
def _(Path, pd, test_adv, test_preds_v5):
    # 提出用データの作成 (フェーズ5)
    submission_v5 = pd.DataFrame({
        'datetime': test_adv['datetime'].dt.strftime('%Y-%-m-%-d'),
        'prediction': test_preds_v5
    })

    sub_path_v5 = Path("/Users/takahiko_tsunoda/work/dev/No35_marimo_poc1/projects/bento2/data/submission_v5.csv")
    submission_v5.to_csv(sub_path_v5, index=False, header=False)
    return (sub_path_v5,)


@app.cell
def _(mo, sub_path_v5):
    mo.md(f"""
    提出ファイル(v5)を作成しました: `{sub_path_v5}`
    """)
    return


if __name__ == "__main__":
    app.run()
