# marimo Data Analysis Platform

marimoを使ったインタラクティブなデータ分析プラットフォーム。

## プロジェクト構成

本リポジトリは複数の分析プロジェクトを管理するモノレポ構成になっています。

```
.
├── projects/
│   ├── bento/                  # お弁当販売数予測
│   │   ├── data/              # データファイル
│   │   ├── notebooks/         # marimoノートブック
│   │   └── requirements.txt   # 依存ライブラリ
│   │
│   └── titanic/                # Titanic生存予測
│       ├── data/
│       ├── notebooks/
│       └── requirements.txt
├── .venv/                      # Python仮想環境（共有）
├── .claude/                    # 開発者向け設定
└── README.md                   # このファイル
```

## 環境構築

### 1. 仮想環境の作成と有効化

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. パッケージのインストール

分析したいプロジェクトに合わせて依存ライブラリをインストールします。

**例: Bentoプロジェクト（お弁当販売予測）の場合**
```bash
pip install -r projects/bento/requirements.txt
```

**例: Titanicプロジェクトの場合**
```bash
pip install -r projects/titanic/requirements.txt
```

## 使用方法

### marimoアプリの起動

プロジェクトごとのノートブックを指定して起動します。

**Bento分析の起動**
```bash
source .venv/bin/activate
marimo edit projects/bento/notebooks/bento_analysis.py
```

**Titanic分析の起動**
```bash
source .venv/bin/activate
marimo edit projects/titanic/notebooks/titanic_analysis.py
```

## 使用技術

- **marimo**: インタラクティブなPythonノートブック環境
- **polars**: 高速データフレーム処理
- **altair**: 宣言的なグラフ可視化
- **scikit-learn**: 機械学習（Bentoプロジェクト等）

## 新しい分析プロジェクトの追加

新しい分析テーマを開始する場合は、`projects/` 配下に新しいディレクトリを作成します。

```bash
# 1. プロジェクトディレクトリの作成
mkdir -p projects/new_project/data
mkdir -p projects/new_project/notebooks

# 2. requirements.txtの作成（基本セット）
echo -e "marimo>=0.9.0\npolars-lts-cpu>=0.20.0\naltair>=5.2.0" > projects/new_project/requirements.txt

# 3. ノートブックの作成と起動
marimo edit projects/new_project/notebooks/analysis.py
```

## 開発者向け情報

環境セットアップ、トラブルシューティング、アーキテクチャの詳細については `.claude/CLAUDE.md` を参照してください。

## ライセンス

MIT License