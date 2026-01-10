# CLAUDE.md

このファイルは、本リポジトリで作業する際のガイドラインを提供します。

## プロジェクト概要

複数のデータ分析プロジェクトを管理する marimo ベースのモノレポ構成。

**技術スタック**:
- **marimo**: インタラクティブノートブック環境
- **polars-lts-cpu**: 高速データフレーム処理
- **altair**: 宣言的なグラフ可視化
- **scikit-learn**: 機械学習・予測モデル構築

## プロジェクト構造

```
.
├── projects/
│   ├── bento/                  # お弁当販売数予測
│   │   ├── data/              # 訓練/テストデータ
│   │   ├── notebooks/         # marimoノートブック
│   │   └── requirements.txt   # プロジェクト固有の依存関係
│   │
│   └── titanic/                # Titanic生存予測
│       ├── data/
│       ├── notebooks/
│       └── requirements.txt
├── .venv/                      # 共通仮想環境
├── README.md                   # 実行ガイド
└── PR_REVIEW.md                # 過去のレビュー記録
```

## 開発ガイドライン

### 1. プロジェクトの独立性
- 各プロジェクトは `projects/<name>/` 配下で完結させる。
- 依存ライブラリの追加時は、該当プロジェクトの `requirements.txt` を更新する。

### 2. データリーク（Data Leakage）の防止
- **重要**: テストデータの欠損値補完やスケーリングには、**必ず訓練データの統計量（中央値、平均など）を使用する**こと。
- テストデータ自身の中央値で補完することは、将来の情報をカンニングすることになり、モデルの評価を歪めるため厳禁。

### 3. marimo セル設計
- **変数再定義の禁止**: 同じ変数を複数のセルで定義してはいけない。
- **依存関係の明示**: セル関数の引数（例: `def _(df_pandas):`）で他のセルへの依存関係を表現する。
- **データ変換の一元化**: `to_pandas()` などの重い変換や共通データ処理は、専用のセルで一度だけ行い、結果を他のセルで共有する。

## 実行コマンド

### 環境有効化とパッケージインストール

```bash
source .venv/bin/activate
# プロジェクトに応じたインストール
pip install -r projects/bento/requirements.txt
```

### アプリの起動

```bash
# お弁当分析
marimo edit projects/bento/notebooks/bento_analysis.py

# Titanic分析
marimo edit projects/titanic/notebooks/titanic_analysis.py
```

## 実装パターン

### 堅牢な欠損値補完（データリーク防止）

```python
# 1. 訓練データで統計量を計算
train_median = df_train["column"].median()

# 2. 訓練・テスト両方に同じ値を適用
df_train_fe = df_train.with_columns(pl.col("column").fill_null(train_median))
df_test_fe = df_test.with_columns(pl.col("column").fill_null(train_median))
```

### marimo での可視化フロー

1. Polars でデータ集計・加工
2. `df.to_pandas()` で一度だけ Pandas に変換
3. Altair でグラフ作成

## 修正時のチェックリスト

- [ ] 変更が他のプロジェクト（`bento` vs `titanic`）に影響していないか？
- [ ] テストデータの処理に訓練データの統計量を使っているか？（データリークがないか）
- [ ] marimo のセル間で変数の重複定義がないか？
- [ ] パス参照が `Path(__file__).parent.parent / "data"` の形式で正しく記述されているか？