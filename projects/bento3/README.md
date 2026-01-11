# Bento3 販売数予測プロジェクト

お弁当販売数を予測するコンペティション用の分析プロジェクト。

## 戦略

- 目標RMSE: 10以下（既存スコア27から大幅改善）
- シンプルな特徴量（15個）で過学習を回避
- ラグ特徴量を使わず、再帰予測の誤差累積を回避
- 正則化モデル（Ridge）とRandomForestで比較

## 環境構築

### 1. プロジェクトディレクトリへ移動

```bash
cd projects/bento3
```

### 2. 仮想環境の作成と有効化

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. パッケージのインストール

```bash
pip install -r requirements.txt
```

### 4. marimoの起動

```bash
marimo edit notebooks/bento3_analysis.py
```

## 特徴量（15個）

基本特徴量:
- year, month, dayofweek, temperature

フラグ特徴量:
- is_friday, is_special_menu（お楽しみメニュー）, payday, event_flag, is_rainy, month_end

メニュー分類:
- menu_curry, menu_fry, menu_hamburg, menu_fish, menu_meat

## 分析の流れ

1. EDA（データ理解・可視化）
2. ベースラインモデル（線形回帰、基本4特徴量のみ）
3. 改良モデル（Ridge、RandomForest）
4. モデル比較と最良モデル選択
5. 提出ファイル生成

## データリーク防止

- テストデータの統計量は使わない
- 訓練データの中央値をテストデータにも適用

## 提出ファイル

- ファイル名: submission.csv
- 形式: ヘッダーなし、日付（yyyy-m-d形式、ゼロ埋めなし）と予測値

## 期待される結果

- ベースライン（基本4特徴量）: RMSE 13程度
- 改良モデル（全15特徴量）: RMSE 10以下を目指す
