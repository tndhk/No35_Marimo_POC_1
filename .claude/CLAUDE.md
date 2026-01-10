# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

marimo を使ったインタラクティブなデータ分析環境。Titanicデータセットを使った探索的データ分析（EDA）プロジェクト。

**技術スタック**:
- marimo: インタラクティブノートブック環境
- polars-lts-cpu: データ処理（Apple Silicon互換版）
- altair: グラフ可視化
- pandas, pyarrow: データ変換

## 環境構築と実行

### 初回セットアップ

```bash
# 仮想環境の作成と有効化
python3 -m venv .venv
source .venv/bin/activate

# パッケージのインストール
pip install -r requirements.txt

# データのダウンロード（初回のみ）
curl -L "https://docs.google.com/spreadsheets/d/1cJdocFdLpZ7hNUi4fTFvQK4ovIWYruqQ_nCuXR_w3Lw/export?format=csv&gid=0" \
  -o data/titanic.csv
```

### アプリの実行

```bash
source .venv/bin/activate
marimo edit notebooks/titanic_analysis.py
```

ブラウザが自動的に開き、インタラクティブな分析環境が利用できます（デフォルトポート: 2718）。

### コード検証

```bash
# Pythonシンタックスチェック
source .venv/bin/activate
python -m py_compile notebooks/titanic_analysis.py
```

## プロジェクト構造

```
.
├── data/
│   └── titanic.csv              # Titanicデータセット（891行 × 12列）
├── notebooks/
│   └── titanic_analysis.py      # marimoアプリケーション（メインコード）
├── requirements.txt             # Python依存パッケージ
├── README.md                    # ユーザー向けドキュメント
└── .claude/
    └── CLAUDE.md               # このファイル
```

## marimoアプリの構造

`notebooks/titanic_analysis.py` は複数のセルで構成されています：

### セル設計のポイント

**marimoの制約事項**:
- 同じ変数を複数のセルで定義することは禁止（依存関係管理のため）
- セル関数の引数が依存関係を定義（`def _(arg1, arg2)` → これらのセルに依存）

### セル構成

1. **インポート** (行1-13)
   - marimo, polars, altair, Path をインポート

2. **データロード** (行16-21)
   - `data/titanic.csv` を Polars DataFrameで読み込み
   - 相対パスは `Path(__file__).parent.parent / "data"` で解決

3. **基本分析** (行24-139)
   - データセットの概要
   - 基本統計量
   - 欠損値チェック（注意：Polars→Pandas変換が必要）
   - 生存率の計算（Polars集計）

4. **可視化** (行142-239)
   - **重要**: pandas変換は専用セル（150-154行）で一度だけ実行
   - 4つのグラフセルが `df_pandas` を共通で使用
   - 各グラフセルは異なる出力変数（`chart_sex`, `chart_class`, `chart_age`, `chart_fare_age`）を定義

5. **分析サマリー** (行242-256)
   - 発見事項のまとめ

## よくある落とし穴と解決方法

### 1. Apple Silicon + Rosetta環境でのクラッシュ

**症状**: marimoが起動してもPythonカーネルが予期せず終了

**原因**: x86_64 Pythonで通常の `polars` パッケージを実行時にセグメンテーションフォルト

**対応**:
```bash
# requirements.txt が polars-lts-cpu を使用していることを確認
grep "polars" requirements.txt
# 出力: polars-lts-cpu>=0.20.0

# 古いpolarsがインストールされている場合
pip uninstall -y polars polars-runtime-32
pip install polars-lts-cpu
```

### 2. marimoの変数再定義エラー

**症状**: `This cell redefines variables from other cells` エラー

**原因**: 複数のセルで同じ変数を定義しようとした

**対応パターン**: pandas変換セルを一元化
```python
# 【悪い例】
@app.cell
def _(df):
    df_pandas = df.to_pandas()  # セル1で定義
    ...

@app.cell
def _(df):
    df_pandas = df.to_pandas()  # セル2で定義 → エラー！

# 【正しい例】
@app.cell
def _(df):
    # pandas変換は専用セルで一度だけ
    df_pandas = df.to_pandas()
    return df_pandas,

@app.cell
def _(alt, df_pandas):  # df ではなく df_pandas を受け取る
    chart = alt.Chart(df_pandas).mark_bar().encode(...)
    return chart,
```

### 3. PyArrow/Pandas不足のエラー

**症状**: `ModuleNotFoundError: No module named 'pyarrow'` または Altairがpandasを見つからない

**原因**: Polars → Pandas変換やAltair使用時に必須

**対応**:
```bash
# requirements.txt に以下が含まれていることを確認
grep -E "pyarrow|pandas" requirements.txt

# 不足の場合
pip install pyarrow pandas
```

## データ処理の実装パターン

### Polarsを使った集計

```python
# 生存率の計算例
survival_by_sex = (
    df.group_by("Sex")
    .agg([
        (pl.col("Survived") == 1).sum().alias("生存者数"),
        pl.count().alias("総人数")
    ])
    .with_columns([
        (pl.col("生存者数") / pl.col("総人数") * 100).alias("生存率(%)")
    ])
)
```

**Polarsの利点**: x86_64環境でもlts-cpuなら安定、高速データ処理

### Polars → Pandas → Altair の変換フロー

```python
# 1. Polars DataFrameをロード
df = pl.read_csv("data.csv")

# 2. 専用セルでPandas変換（一度だけ）
df_pandas = df.to_pandas()  # PyArrow経由で変換

# 3. 複数のグラフセルから df_pandas を使用
chart = alt.Chart(df_pandas).mark_bar().encode(...)
```

## 修正・改善時の注意点

### セルの追加・削除時

- **新しいセルを追加**: 依存するセルを考慮し、関数引数で明示的に依存関係を示す
- **セルの結合**: 変数の再定義がないか確認
- **セルの削除**: 依存するセルがないか確認

### グラフの追加

```python
# 新しいグラフを追加する場合
@app.cell
def _(alt, df_pandas):  # df_pandas のみを引数に
    # 新しいグラフロジック
    new_chart = alt.Chart(df_pandas).mark_bar().encode(...)
    return new_chart,  # 戻り値を明示的に指定
```

### データ処理の変更

- Polarsで処理 → Pandas変換は専用セル → グラフセルで使用という流れを維持
- 欠損値フィルタリングは `dropna(subset=[...])` で明示的に実施
- Polarsの型チェック: `df.schema` で確認

## トラブルシューティング参考資料

詳細なトラブルシューティング情報は `README.md` に記載：

- Apple Silicon + Rosettaでのクラッシュ対応
- marimoの変数再定義エラー
- パッケージ依存関係の問題
- marimoが起動しない場合
- セル実行エラーのデバッグ

## 推奨される開発フロー

1. **ローカルテスト**: `python -m py_compile` でシンタックスチェック
2. **アプリ起動**: `marimo edit notebooks/titanic_analysis.py`
3. **ブラウザ確認**: セル実行順序、エラー表示、グラフレンダリング
4. **依存関係確認**: marimoのUI上で「Cell Dependencies」を表示

## ローカルGITの初期化（将来使用）

このプロジェクトは現在Gitで管理されていません。将来管理する場合：

```bash
git init
git add .
git commit -m "Initial commit: marimo Titanic analysis app"
```

`.gitignore` には以下を追加推奨：
```
.venv/
.marimo/
__pycache__/
*.pyc
.DS_Store
```
