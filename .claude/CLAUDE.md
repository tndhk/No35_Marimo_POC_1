# CLAUDE.md

このファイルは、本リポジトリで作業する際のガイドラインを提供します。

## プロジェクト概要

複数のデータ分析プロジェクトを管理する marimo ベースのモノレポ構成。

**技術スタック**:
- **marimo**: インタラクティブノートブック環境（詳細は marimoスキルを参照）
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

---

## 開発ガイドライン

### 1. プロジェクトの独立性
- 各プロジェクトは `projects/<name>/` 配下で完結させる
- 依存ライブラリの追加時は、該当プロジェクトの `requirements.txt` を更新する

### 2. データリーク（Data Leakage）の防止
- **重要**: テストデータの欠損値補完やスケーリングには、**必ず訓練データの統計量（中央値、平均など）を使用する**こと
- テストデータ自身の中央値で補完することは、将来の情報をカンニングすることになり、モデルの評価を歪めるため厳禁

### 3. marimo開発
- marimoの詳細なガイドラインは、グローバルスキル（~/.claude/skills/marimo/skill.md）を参照
- 本プロジェクトでは Polars を優先的に使用する（`polars-lts-cpu`）

---

## 実行コマンド

### 環境有効化とパッケージインストール

```bash
cd projects/bento
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### アプリの起動

```bash
# プロジェクトディレクトリ内で実行
marimo edit notebooks/bento_analysis.py
```

---

## 修正時のチェックリスト

- [ ] 変更が他のプロジェクト（`bento` vs `titanic`）に影響していないか？
- [ ] テストデータの処理に訓練データの統計量を使っているか？（データリークがないか）
- [ ] marimo のセル間で変数の重複定義がないか？
- [ ] パス参照が `Path(__file__).parent.parent / "data"` の形式で正しく記述されているか？
- [ ] 可視化で `plt.gca()` を最後の式として使用しているか？（matplotlibの場合）
- [ ] UI要素の値アクセスを定義とは別のセルで行っているか？
