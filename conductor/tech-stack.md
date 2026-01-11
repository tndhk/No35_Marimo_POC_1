# Technology Stack

## Core Technologies
- **Python**: 開発および分析の主要言語。
- **marimo**: インタラクティブなノートブック環境。リアクティブな挙動を活かした分析ダッシュボードの構築に使用。

## Data Engineering & Science
- **Polars**: 高速なデータフレーム操作（推奨）。
- **Pandas**: 互換性や特定のライブラリ連携のためのデータ操作。
- **scikit-learn**: 機械学習アルゴリズムと評価。
- **LightGBM**: 勾配ブースティング決定木を用いた高精度な予測モデル。

## Visualization
- **Altair**: 宣言的な統計可視化。marimoとの親和性が高く、インタラクティブなグラフ作成に使用。

## Architecture & Infrastructure
- **Monorepo**: `projects/` ディレクトリ配下に独立した分析プロジェクトを配置。
- **Virtual Environments**: 各プロジェクトが `requirements.txt` を持ち、個別の `.venv` で依存関係を管理。
