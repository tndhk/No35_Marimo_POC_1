
__generated_with = "0.19.1"

# %%
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

# %%
# 訓練データロード
data_path_train = Path(__file__).parent.parent / "data" / "bento_train.csv"
df_train = pl.read_csv(data_path_train)

# %%
# テストデータロード
data_path_test = Path(__file__).parent.parent / "data" / "bento_test.csv"
df_test = pl.read_csv(data_path_test)

# %%
# タイトルと概要
mo.md(f"""
# お弁当販売数予測 - EDA & モデル構築

訓練データ: {df_train.shape[0]} 行 × {df_train.shape[1]} 列 (2013-11-18〜2014-09-30)

テストデータ: {df_test.shape[0]} 行 × {df_test.shape[1]} 列 (2014-10-01〜2014-11-28)

評価指標: RMSE（Root Mean Squared Error）
""")

# %%
mo.md("""
## 1. 基本統計とデータ理解
""")

# %%
# 基本統計量
train_stats = df_train.describe()
train_stats

# %%
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

# %%
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

# %%
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

# %%
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

# %%
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

# %%
mo.md("""
## 2. データ可視化
""")

# %%
# Pandas変換（可視化用）
df_train_pandas = df_train.to_pandas()
df_test_pandas = df_test.to_pandas()
df_train_pandas['datetime'] = pd.to_datetime(df_train_pandas['datetime'])
df_test_pandas['datetime'] = pd.to_datetime(df_test_pandas['datetime'])

# %%
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

# %%
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

# %%
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

# %%
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

# %%
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

# %%
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

# %%
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

# %%
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

# %%
mo.md("""
## 3. 特徴量エンジニアリング

モデル学習のための特徴量を作成します。
""")

# %%
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

    return df_fe

# %%
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

# %%
mo.md("""
## 4. モデル構築と評価

**LightGBM** を使用し、時系列データに適した **TimeSeriesSplit** で交差検証を行います。
""")

# %%
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


# %%
# 旧モデル（線形回帰・Ridge）のセルは削除されました

# %%
# 旧モデル（決定木・RF）のセルは削除されました

# %%
# 旧モデル（GBR）のセルは削除されました

# %%
mo.md(f"""
## 5. 最終モデル評価

### ベストモデル: LightGBM (TimeSeriesSplit CV)
- **Cross Validation RMSE**: {overall_rmse:.4f}

※ 時系列分割交差検証の結果、上記の精度が得られました。全データで再学習させて提出用予測を作成します。
""")

# %%
# 全データでの再学習
# CVで良さそうなパラメータを使用（ここでは固定）
final_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 1200, # 全データなので少し増やす
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

final_model = lgb.LGBMRegressor(**final_params)
final_model.fit(X, y)


# %%
# テストデータへの予測
X_test = df_test_fe[feature_cols]

test_predictions = final_model.predict(X_test)

# 負の値を0にクリップ（販売数は0以上）
test_predictions = [max(0, pred) for pred in test_predictions]

# %%
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
