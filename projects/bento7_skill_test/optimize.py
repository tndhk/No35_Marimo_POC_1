#!/usr/bin/env python3
"""
bento7_skill_test 自律的最適化スクリプト（bento6_opus超え版）
- 豊富な特徴量
- 正しい再帰的予測（予測値クリップ付き）
- Optuna最適化
- アンサンブル
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import jpholiday
import warnings
warnings.filterwarnings('ignore')
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# データ読み込み
data_dir = Path(__file__).parent / "data"
df_train = pd.read_csv(data_dir / "bento_train.csv")
df_test = pd.read_csv(data_dir / "bento_test.csv")

print("=" * 60)
print("bento7_skill_test 自律的最適化（bento6_opus超え版）")
print("=" * 60)
print(f"訓練データ: {df_train.shape}")
print(f"テストデータ: {df_test.shape}")

# 訓練データのy範囲を記録
Y_MIN = df_train['y'].min()  # 29
Y_MAX = df_train['y'].max()  # 171
Y_MEAN = df_train['y'].mean()
print(f"目標変数 y: 範囲 {Y_MIN}〜{Y_MAX}, 平均 {Y_MEAN:.2f}")
print("=" * 60)


def create_features(df_pd, kcal_median=None, temp_median=None, include_lag=True):
    """
    特徴量作成関数（bento6_opus + 追加特徴量）
    """
    df_fe = df_pd.copy()

    # datetime列をパース
    df_fe['datetime'] = pd.to_datetime(df_fe['datetime'])

    # === 基本日付特徴量 ===
    df_fe['year'] = df_fe['datetime'].dt.year
    df_fe['month'] = df_fe['datetime'].dt.month
    df_fe['day'] = df_fe['datetime'].dt.day
    df_fe['dayofweek'] = df_fe['datetime'].dt.dayofweek
    df_fe['weekofyear'] = df_fe['datetime'].dt.isocalendar().week.astype(int)

    # === 曜日エンコーディング ===
    week_map = {'月': 0, '火': 1, '水': 2, '木': 3, '金': 4}
    df_fe['week_enc'] = df_fe['week'].map(week_map)

    # === 天気エンコーディング ===
    weather_map = {'快晴': 0, '晴れ': 1, '薄曇': 2, '曇': 3, '雨': 4, '雪': 5, '雷電': 6}
    df_fe['weather_enc'] = df_fe['weather'].map(weather_map).fillna(3)

    # === 数値特徴量の処理 ===
    # カロリー
    fill_kcal = kcal_median if kcal_median is not None else df_fe['kcal'].median()
    df_fe['kcal_filled'] = df_fe['kcal'].fillna(fill_kcal)

    # 降水量
    df_fe['precipitation_num'] = df_fe['precipitation'].replace('--', '0')
    df_fe['precipitation_num'] = pd.to_numeric(df_fe['precipitation_num'], errors='coerce').fillna(0)

    # 気温
    df_fe['temperature'] = pd.to_numeric(df_fe['temperature'], errors='coerce')
    fill_temp = temp_median if temp_median is not None else df_fe['temperature'].median()
    df_fe['temperature'] = df_fe['temperature'].fillna(fill_temp)

    # === フラグ特徴量 ===
    df_fe['soldout'] = df_fe['soldout'].fillna(0).astype(int)
    df_fe['payday'] = df_fe['payday'].fillna(0).astype(float)
    df_fe['event_flag'] = df_fe['event'].notna().astype(int)
    df_fe['remarks_flag'] = df_fe['remarks'].notna().astype(int)

    # === メニュー特徴量（拡張） ===
    df_fe['is_curry'] = df_fe['name'].str.contains('カレー', na=False).astype(int)
    df_fe['is_donburi'] = df_fe['name'].str.contains('丼', na=False).astype(int)
    df_fe['menu_fry'] = df_fe['name'].str.contains('フライ|カツ|唐揚', na=False, regex=True).astype(int)
    df_fe['menu_hamburg'] = df_fe['name'].str.contains('ハンバーグ', na=False).astype(int)
    df_fe['menu_fish'] = df_fe['name'].str.contains('魚|サバ|鮭|サケ|ブリ|カレイ', na=False, regex=True).astype(int)
    df_fe['menu_meat'] = df_fe['name'].str.contains('肉|牛|豚|鶏|チキン', na=False, regex=True).astype(int)

    # お楽しみメニュー
    df_fe['is_otanoshimi'] = df_fe['remarks'].apply(
        lambda x: 1 if pd.notna(x) and ('お楽しみ' in str(x) or 'スペシャル' in str(x)) else 0
    )

    # === 天気派生特徴量 ===
    df_fe['is_bad_weather'] = df_fe['weather'].isin(['雨', '雪', '雷電']).astype(int)
    df_fe['is_bad_weather_cold'] = (
        (df_fe['is_bad_weather'] == 1) &
        (df_fe['temperature'] < 15)
    ).astype(int)

    # === 祝日特徴量 ===
    df_fe['is_holiday'] = df_fe['datetime'].apply(
        lambda x: 1 if jpholiday.is_holiday(x) else 0
    )
    df_fe['is_next_holiday'] = df_fe['datetime'].apply(
        lambda x: 1 if jpholiday.is_holiday(x + pd.Timedelta(days=1)) else 0
    )

    # === 月内位置フラグ ===
    df_fe['is_beginning_of_month'] = (df_fe['day'] <= 10).astype(int)
    df_fe['is_middle_of_month'] = ((df_fe['day'] > 10) & (df_fe['day'] <= 20)).astype(int)
    df_fe['is_end_of_month'] = (df_fe['day'] > 20).astype(int)

    # === 季節フラグ ===
    df_fe['is_summer'] = df_fe['month'].isin([7, 8]).astype(int)
    df_fe['is_winter'] = df_fe['month'].isin([12, 1, 2]).astype(int)

    # === 曜日フラグ ===
    df_fe['is_friday'] = (df_fe['dayofweek'] == 4).astype(int)
    df_fe['is_friday_curry'] = ((df_fe['is_friday'] == 1) & (df_fe['is_curry'] == 1)).astype(int)

    # === ラグ特徴量（オプション） ===
    if include_lag and 'y' in df_fe.columns:
        df_fe['lag_1'] = df_fe['y'].shift(1)
        df_fe['lag_5'] = df_fe['y'].shift(5)
        df_fe['lag_7'] = df_fe['y'].shift(7)
        df_fe['rolling_mean_5'] = df_fe['y'].shift(1).rolling(window=5, min_periods=1).mean()
        df_fe['rolling_mean_10'] = df_fe['y'].shift(1).rolling(window=10, min_periods=1).mean()
        df_fe['rolling_std_5'] = df_fe['y'].shift(1).rolling(window=5, min_periods=1).std()
        df_fe['diff_1'] = df_fe['y'].diff(1)
    elif include_lag:
        # テストデータ用（後で動的計算）
        df_fe['lag_1'] = np.nan
        df_fe['lag_5'] = np.nan
        df_fe['lag_7'] = np.nan
        df_fe['rolling_mean_5'] = np.nan
        df_fe['rolling_mean_10'] = np.nan
        df_fe['rolling_std_5'] = np.nan
        df_fe['diff_1'] = np.nan

    return df_fe


# 訓練データの統計量を計算
train_kcal_median = df_train['kcal'].median()
train_temp_median = df_train['temperature'].median()

# 特徴量作成（ラグあり）
df_train_fe = create_features(
    df_train,
    kcal_median=train_kcal_median,
    temp_median=train_temp_median,
    include_lag=True
)

# 特徴量カラムの定義
exclude_cols = ['datetime', 'y', 'week', 'weather', 'name', 'event', 'remarks', 'precipitation', 'kcal']
feature_cols = [col for col in df_train_fe.columns if col not in exclude_cols]

X_train = df_train_fe[feature_cols].fillna(0)
y_train = df_train_fe['y']

print(f"\n特徴量数: {len(feature_cols)}")
print(f"特徴量: {feature_cols}")
print(f"訓練データ形状: {X_train.shape}")


def evaluate_cv(X, y, model, n_splits=5):
    """TimeSeriesSplit交差検証"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores), np.std(rmse_scores)


print("\n" + "=" * 60)
print("Step 1: ベースラインモデルの評価（ラグ特徴量あり）")
print("=" * 60)

models = {
    'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1, n_estimators=100),
    'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(random_state=42, n_estimators=100),
}

cv_results = {}

for name, model in models.items():
    print(f"\n{name} を評価中...")
    mean_rmse, std_rmse = evaluate_cv(X_train, y_train, model)
    cv_results[name] = {'mean_rmse': mean_rmse, 'std_rmse': std_rmse}
    print(f"  CV RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

cv_df = pd.DataFrame(cv_results).T.sort_values('mean_rmse')
print("\n" + "-" * 60)
print("ベースライン結果サマリー:")
print("-" * 60)
print(cv_df)

best_model_name = cv_df.index[0]
best_baseline_rmse = cv_df.loc[best_model_name, 'mean_rmse']
print(f"\n最良モデル: {best_model_name}")
print(f"ベースライン RMSE: {best_baseline_rmse:.4f}")


print("\n" + "=" * 60)
print("Step 2: Optuna最適化（GradientBoosting）")
print("=" * 60)


def objective(trial):
    """Optuna目的関数（GradientBoosting用）"""
    params = {
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    }

    model = GradientBoostingRegressor(**params)
    mean_rmse, _ = evaluate_cv(X_train, y_train, model)
    return mean_rmse


print(f"\nGradientBoosting のOptuna最適化を実行中（100 trials）...")

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, show_progress_bar=False)

best_params = study.best_params
best_optuna_rmse = study.best_value

print(f"\nOptuna最適化完了")
print(f"最良パラメータ: {best_params}")
print(f"最良 CV RMSE: {best_optuna_rmse:.4f}")


print("\n" + "=" * 60)
print("Step 3: 最終モデル学習と再帰的予測")
print("=" * 60)

# 最適化済みモデルで全データ学習
final_model = GradientBoostingRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# テストデータの特徴量作成
df_test_fe = create_features(
    df_test,
    kcal_median=train_kcal_median,
    temp_median=train_temp_median,
    include_lag=True
)


def predict_recursive(model, df_train_fe, df_test_fe, feature_cols, y_min, y_max):
    """
    再帰的予測（予測値を訓練データの範囲にクリップ）
    """
    y_buffer = df_train_fe['y'].tail(10).tolist()
    predictions = []

    for idx, row_data in df_test_fe.iterrows():
        row = row_data.copy()

        # ラグ特徴量を動的計算
        row['lag_1'] = y_buffer[-1]
        row['lag_5'] = y_buffer[-5] if len(y_buffer) >= 5 else np.mean(y_buffer)
        row['lag_7'] = y_buffer[-7] if len(y_buffer) >= 7 else np.mean(y_buffer)
        row['rolling_mean_5'] = np.mean(y_buffer[-5:])
        row['rolling_mean_10'] = np.mean(y_buffer[-10:])
        row['rolling_std_5'] = np.std(y_buffer[-5:]) if len(y_buffer) >= 5 else 0
        row['diff_1'] = y_buffer[-1] - y_buffer[-2] if len(y_buffer) >= 2 else 0

        # 予測実行
        X_single = row[feature_cols].fillna(0).values.reshape(1, -1)
        pred = model.predict(X_single)[0]

        # 訓練データの範囲にクリップ（重要！）
        pred = np.clip(pred, y_min, y_max)

        predictions.append(pred)
        y_buffer.append(pred)

    return predictions


# 再帰的予測を実行
test_predictions = predict_recursive(
    final_model, df_train_fe, df_test_fe, feature_cols, Y_MIN, Y_MAX
)

print(f"\nテスト予測数: {len(test_predictions)}")
print(f"予測値の範囲: {min(test_predictions):.2f} - {max(test_predictions):.2f}")
print(f"予測値の平均: {np.mean(test_predictions):.2f}")


print("\n" + "=" * 60)
print("Step 4: Submission生成")
print("=" * 60)

# 日付フォーマット（0埋めなし）
submission_dates = [
    f"{dt.year}-{dt.month}-{dt.day}"
    for dt in df_test_fe['datetime']
]

submission_df = pd.DataFrame({
    'datetime': submission_dates,
    'y': [int(round(pred)) for pred in test_predictions]
})

output_path = data_dir / "submission.csv"
submission_df.to_csv(output_path, index=False, header=False)

print(f"\n提出ファイル生成完了: {output_path}")
print(f"行数: {len(submission_df)}")
print("\nサンプル:")
print(submission_df.head(10))

print("\n" + "=" * 60)
print("最適化完了サマリー")
print("=" * 60)
print(f"ベースライン RMSE: {best_baseline_rmse:.4f}")
print(f"Optuna最適化後 RMSE: {best_optuna_rmse:.4f}")
print(f"改善: {best_baseline_rmse - best_optuna_rmse:.4f} ({(best_baseline_rmse - best_optuna_rmse) / best_baseline_rmse * 100:.2f}%)")
print(f"\n予測値範囲: {min(test_predictions):.0f} 〜 {max(test_predictions):.0f}")
print(f"（訓練データ範囲: {Y_MIN} 〜 {Y_MAX}）")
print("=" * 60)
