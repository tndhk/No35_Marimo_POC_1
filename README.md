# marimo Data Analysis Platform

marimoを使ったインタラクティブなデータ分析プラットフォーム。

## 環境構築

### 1. 仮想環境の作成と有効化

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. パッケージのインストール

```bash
pip install -r requirements.txt
```

### 3. データの準備

分析用のデータファイルを `data/` ディレクトリに配置してください。

## 使用方法

### marimoアプリの起動

```bash
source .venv/bin/activate
marimo edit notebooks/<notebook_name>.py
```

ブラウザが自動的に開き、インタラクティブな分析環境が利用できます。

### 別のポートで実行

```bash
marimo edit --port 2719 notebooks/<notebook_name>.py
```

## 使用技術

- **marimo**: インタラクティブなPythonノートブック環境
- **polars-lts-cpu**: 高速データフレーム処理（Apple Silicon互換版）
- **altair**: 宣言的なグラフ可視化
- **pandas, pyarrow**: データ形式変換

## プロジェクト構造

```
.
├── .venv/                      # Python仮想環境
├── .claude/
│   └── CLAUDE.md              # 開発者向けガイド
├── data/
│   └── *.csv                  # 分析用データファイル
├── notebooks/
│   └── *.py                   # marimoアプリケーション
├── .gitignore                 # Git除外設定
├── requirements.txt           # パッケージ依存関係
└── README.md                  # このファイル
```

## 新しい分析の追加

### 新規ノートブックの作成

```bash
# テンプレートとして既存のノートブックをコピー
cp notebooks/example_analysis.py notebooks/my_analysis.py

# 必要に応じてカスタマイズして起動
marimo edit notebooks/my_analysis.py
```

### marimoセル設計のベストプラクティス

- セル関数の引数で依存関係を明示的に表現
- 同じ変数を複数のセルで定義しない（共有データは専用セルで一度だけ処理）
- `return` で出力変数を明示的に指定

詳細は `.claude/CLAUDE.md` を参照。

## 開発者向け情報

環境セットアップ、トラブルシューティング、アーキテクチャの詳細については `.claude/CLAUDE.md` を参照してください。

## ライセンス

MIT License
