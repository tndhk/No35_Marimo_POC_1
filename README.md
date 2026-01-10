# marimo Data Analysis Platform

marimoを使ったインタラクティブなデータ分析プラットフォーム。

## プロジェクト構成

本リポジトリは複数の分析プロジェクトを管理するモノレポ構成になっています。
各プロジェクトは完全に独立しており、それぞれで仮想環境を構築します。

```
. 
├── projects/
│   ├── bento/                  # お弁当販売数予測
│   │   ├── .venv/             # プロジェクト専用の仮想環境
│   │   ├── data/              # データファイル
│   │   ├── notebooks/         # marimoノートブック
│   │   └── requirements.txt   # 依存ライブラリ
│   │
│   └── titanic/                # Titanic生存予測
│       ├── .venv/             # プロジェクト専用の仮想環境
│       ├── data/              # データファイル
│       ├── notebooks/         # marimoノートブック
│       └── requirements.txt   # 依存ライブラリ
├── .claude/                    # 開発者向け設定
└── README.md                   # このファイル
```

## 環境構築と実行

各プロジェクトのディレクトリに移動して環境をセットアップしてください。

### 例：Bentoプロジェクト（お弁当販売予測）

1. **プロジェクトディレクトリへ移動**
   ```bash
   cd projects/bento
   ```

2. **仮想環境の作成と有効化**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **パッケージのインストール**
   ```bash
   pip install -r requirements.txt
   ```

4. **marimoの起動**
   ```bash
   marimo edit notebooks/bento_analysis.py
   ```

### 例：Titanicプロジェクト

1. **プロジェクトディレクトリへ移動**
   ```bash
   cd projects/titanic
   ```

2. **セットアップと起動**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   marimo edit notebooks/titanic_analysis.py
   ```

## 使用技術

- **marimo**: インタラクティブなPythonノートブック環境
- **polars**: 高速データフレーム処理
- **altair**: 宣言的なグラフ可視化
- **scikit-learn**: 機械学習

## 新しい分析プロジェクトの追加

新しい分析テーマを開始する場合は、`projects/` 配下に新しいディレクトリを作成します。

```bash
# 1. プロジェクトディレクトリの作成
mkdir -p projects/new_project/data
mkdir -p projects/new_project/notebooks

# 2. requirements.txtの作成
echo -e "marimo>=0.9.0\npolars-lts-cpu>=0.20.0\naltair>=5.2.0" > projects/new_project/requirements.txt

# 3. 仮想環境の作成
cd projects/new_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. 開発開始
marimo edit notebooks/analysis.py
```

## 開発者向け情報

アーキテクチャの詳細やベストプラクティスについては `.claude/CLAUDE.md` を参照してください。

## ライセンス

MIT License