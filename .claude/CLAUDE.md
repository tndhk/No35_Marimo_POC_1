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

---

## marimo 基礎知識

marimo は従来のノートブックとは異なるリアクティブなノートブック環境です：

- **自動実行**: 依存関係のあるセルは、依存変数が変更されると自動的に再実行される
- **変数の一意性**: セル間で同じ変数を再定義することはできない
- **DAG構造**: ノートブックは有向非巡回グラフ（DAG）を形成する
- **自動表示**: セルの最後の式は自動的に表示される（Jupyterと同様）
- **リアクティブUI**: UI要素は自動的に更新をトリガーする

---

## 開発ガイドライン

### 1. プロジェクトの独立性
- 各プロジェクトは `projects/<name>/` 配下で完結させる
- 依存ライブラリの追加時は、該当プロジェクトの `requirements.txt` を更新する

### 2. データリーク（Data Leakage）の防止
- **重要**: テストデータの欠損値補完やスケーリングには、**必ず訓練データの統計量（中央値、平均など）を使用する**こと
- テストデータ自身の中央値で補完することは、将来の情報をカンニングすることになり、モデルの評価を歪めるため厳禁

### 3. marimo セル設計
- **変数再定義の禁止**: 同じ変数を複数のセルで定義してはいけない
- **依存関係の明示**: セル関数の引数（例: `def _(df_pandas):`）で他のセルへの依存関係を表現する
- **データ変換の一元化**: `to_pandas()` などの重い変換や共通データ処理は、専用のセルで一度だけ行い、結果を他のセルで共有する

### 4. コード要件
- すべてのコードは完全で実行可能であること
- 一貫したコーディングスタイルを維持する
- 説明的な変数名と有用なコメントを含める
- 最初のセルで必ず `import marimo as mo` を含むすべてのモジュールをインポートする
- ノートブックの依存関係グラフに循環がないことを確認する
- markdownセルにはコメントを含めない
- SQLセルにはコメントを含めない

---

## marimo ベストプラクティス

### データ処理
- 本プロジェクトでは Polars を優先的に使用する（`polars-lts-cpu`）
- 適切なデータバリデーションを実装する
- 欠損値を適切に処理する
- セルの最後の式の変数は自動的にテーブルとして表示される

### 可視化
- **matplotlib**: `plt.show()` ではなく `plt.gca()` を最後の式として使用する
- **plotly**: figureオブジェクトを直接返す
- **altair**: chartオブジェクトを直接返す
- 適切なラベル、タイトル、カラースキームを含める
- 適切な場合はインタラクティブな可視化を使用する

### UI要素
- UI要素の値には `.value` 属性でアクセスする（例: `slider.value`）
- UI要素は1つのセルで作成し、後続のセルで参照する
- `mo.hstack()`, `mo.vstack()`, `mo.tabs()` で直感的なレイアウトを作成する
- コールバックよりもリアクティブな更新を優先する（marimoが自動的にリアクティビティを処理）
- 関連するUI要素はグループ化して整理する

### SQLとDuckDB
- DuckDBを使用する場合は、marimoのSQLセルを優先する
- SQLセルは `_df = mo.sql(query)` で開始する
- `mo.sql()` を使用するセルにはコメントを追加しない

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

### インタラクティブ散布図の例

```python
# Cell 1
import marimo as mo
import matplotlib.pyplot as plt
import numpy as np

# Cell 2
n_points = mo.ui.slider(10, 100, value=50, label="Number of points")
n_points

# Cell 3
x = np.random.rand(n_points.value)
y = np.random.rand(n_points.value)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.7)
plt.title(f"Scatter plot with {n_points.value} points")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.gca()
```

### Altairチャートと選択の例

```python
# Cell 1
import marimo as mo
import altair as alt
import pandas as pd

# Cell 2
cars_df = pd.read_csv('https://raw.githubusercontent.com/vega/vega-datasets/master/data/cars.json')
_chart = alt.Chart(cars_df).mark_point().encode(
    x='Horsepower',
    y='Miles_per_Gallon',
    color='Origin',
)

chart = mo.ui.altair_chart(_chart)
chart

# Cell 3
chart.value
```

---

## 利用可能なUI要素

### 入力系
- `mo.ui.button(value=None, kind='primary')` - ボタン
- `mo.ui.run_button(label=None, tooltip=None, kind='primary')` - 実行ボタン
- `mo.ui.checkbox(label='', value=False)` - チェックボックス
- `mo.ui.date(value=None, label=None, full_width=False)` - 日付選択
- `mo.ui.dropdown(options, value=None, label=None, full_width=False)` - ドロップダウン
- `mo.ui.file(label='', multiple=False, full_width=False)` - ファイル選択
- `mo.ui.number(value=None, label=None, full_width=False)` - 数値入力
- `mo.ui.radio(options, value=None, label=None, full_width=False)` - ラジオボタン
- `mo.ui.slider(start, stop, value=None, label=None, full_width=False, step=None)` - スライダー
- `mo.ui.range_slider(start, stop, value=None, label=None, full_width=False, step=None)` - 範囲スライダー
- `mo.ui.text(value='', label=None, full_width=False)` - テキスト入力
- `mo.ui.text_area(value='', label=None, full_width=False)` - テキストエリア

### データ表示系
- `mo.ui.table(data, columns=None, on_select=None, sortable=True, filterable=True)` - テーブル
- `mo.ui.data_explorer(df)` - データエクスプローラー
- `mo.ui.dataframe(df)` - データフレーム表示

### チャート系
- `mo.ui.altair_chart(altair_chart)` - Altairチャート
- `mo.ui.plotly(plotly_figure)` - Plotlyチャート

### レイアウト系
- `mo.ui.tabs(elements: dict[str, mo.ui.Element])` - タブ
- `mo.ui.array(elements: list[mo.ui.Element])` - 配列
- `mo.ui.form(element: mo.ui.Element, label='', bordered=True)` - フォーム
- `mo.ui.refresh(options: List[str], default_interval: str)` - リフレッシュ

---

## レイアウトとユーティリティ関数

- `mo.md(text)` - markdownを表示
- `mo.stop(predicate, output=None)` - 条件付きで実行を停止
- `mo.Html(html)` - HTMLを表示
- `mo.image(image)` - 画像を表示
- `mo.hstack(elements)` - 要素を水平に配置
- `mo.vstack(elements)` - 要素を垂直に配置
- `mo.tabs(elements)` - タブインターフェースを作成

---

## トラブルシューティング

よくある問題と解決策：

- **循環依存**: コードを再構成して依存関係グラフの循環を除去する
- **UI要素の値へのアクセス**: 定義とは別のセルにアクセスを移動する
- **可視化が表示されない**: 可視化オブジェクトが最後の式であることを確認する

---

## 修正時のチェックリスト

- [ ] 変更が他のプロジェクト（`bento` vs `titanic`）に影響していないか？
- [ ] テストデータの処理に訓練データの統計量を使っているか？（データリークがないか）
- [ ] marimo のセル間で変数の重複定義がないか？
- [ ] パス参照が `Path(__file__).parent.parent / "data"` の形式で正しく記述されているか？
- [ ] 可視化で `plt.gca()` を最後の式として使用しているか？（matplotlibの場合）
- [ ] UI要素の値アクセスを定義とは別のセルで行っているか？
