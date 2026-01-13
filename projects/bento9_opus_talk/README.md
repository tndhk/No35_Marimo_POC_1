# bento9_opus_talk - お弁当販売数予測

## 概要

カフェフロアで販売されるお弁当の日次販売数を予測する機械学習プロジェクト。

## データ

- 訓練データ: `data/bento_train.csv`（2013-11-18 〜 2014-09-30）
- テストデータ: `data/bento_test.csv`（2014-10-01 〜 2014-11-28）
- 評価指標: RMSE（Root Mean Squared Error）

## 特徴量

- 日付特徴: year, month, day, weekday, is_holiday
- カテゴリカル: week_encoded, weather_encoded
- 数値: soldout, payday, kcal, precipitation, temperature

## モデル

1. Ridge回帰（ベースライン）
2. LightGBM（メインモデル、5-fold TimeSeriesSplit CV）

## 実行方法

```bash
# 依存パッケージインストール
cd projects/bento9_opus_talk
pip install -r requirements.txt

# marimoノートブック起動
cd notebooks
marimo edit bento9_analysis.py
```

## 結果

- 予測ファイル: `submission.csv`（ヘッダーなし、フォーマット: yyyy-m-d,prediction）

## 注意事項

- データリーク防止: テストデータの欠損値補完には訓練データの統計量を使用
- 日付フォーマット: 1桁の日は0埋めしない（例: 2014-10-1）
