
__generated_with = "0.19.1"

# %%
import marimo as mo
import polars as pl
import pandas as pd
import numpy as np
import altair as alt
import re
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
# import lightgbm as lgb
import optuna

# %%
mo.md("# Bento2 お弁当販売数予測 - 高精度モデル構築")

# データ読み込み
data_dir = Path("/Users/takahiko_tsunoda/work/dev/No35_marimo_poc1/projects/bento2/data")
train_path = data_dir / "bento_train.csv"
test_path = data_dir / "bento_test.csv"

df_train_raw = pl.read_csv(train_path)
df_test_raw = pl.read_csv(test_path)

mo.md(f"訓練データサイズ: {df_train_raw.shape}")

# %%
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

# %%
null_info

# %%
mo.md("## 2. フェーズ 1: シンプルベースライン (RMSE 13目標)")

# %%
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

# %%
train_simple = preprocess_simple(df_train_raw)
test_simple = preprocess_simple(df_test_raw)

features_v1 = ['year', 'month', 'temperature', 'is_friday']

# %%
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

# %%
mo.md("## 3. フェーズ 2: 特徴量実験 (RMSE 11目標)")

# %%
def extract_keywords(name):
    res = []
    if re.search(r'カレー', str(name)): res.append('curry')
    if re.search(r'フライ|カツ|唐揚げ', str(name)): res.append('fry')
    if re.search(r'ハンバーグ', str(name)): res.append('hamburg')
    return res

def preprocess_advanced(df_pl):
    df = df_pl.to_pandas()
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 基本特徴量
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['is_friday'] = (df['dayofweek'] == 4).astype(int)

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
    df['is_curry'] = df['name'].apply(lambda x: 1 if 'カレー' in str(x) else 0)

    # フェーズ3用: ラグ特徴量 (yが存在する場合のみ)
    if 'y' in df.columns:
        df['lag_1'] = df['y'].shift(1)
        df['rolling_mean_5'] = df['y'].shift(1).rolling(window=5).mean()
        # 欠損値は全体の平均等で埋める
        df['lag_1'] = df['lag_1'].fillna(df['y'].mean())
        df['rolling_mean_5'] = df['rolling_mean_5'].fillna(df['y'].mean())
    else:
        # テストデータ用（実際には予測値を再帰的に使うべきだが、まずは直近訓練データ末尾等で代用）
        df['lag_1'] = 0 
        df['rolling_mean_5'] = 0

    return df

# %%
train_adv = preprocess_advanced(df_train_raw)
test_adv = preprocess_advanced(df_test_raw)

features_v2 = [
    'year', 'month', 'temperature', 'is_friday', 
    'dayofweek', 'is_curry', 'is_bad_weather', 'event', 'payday'
]

# %%
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

# %%
mo.md("## 4. フェーズ 3: 高度なモデル (RandomForest + ラグ特徴量)")

# %%
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

# %%
mo.md("## 5. 予測と提出ファイルの生成")

# %%
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

# %%
# 提出用データの作成
submission = pd.DataFrame({
    'datetime': test_adv['datetime'].dt.strftime('%Y-%-m-%-d'),
    'prediction': test_preds
})

sub_path = Path("/Users/takahiko_tsunoda/work/dev/No35_marimo_poc1/projects/bento2/data/submission.csv")
submission.to_csv(sub_path, index=False, header=False)

# %%
mo.md(f"提出ファイルを作成しました: `{sub_path}`")
