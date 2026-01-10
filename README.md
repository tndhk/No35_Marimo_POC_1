# Marimo Titanic Analysis

marimoを使ったTitanicデータセットの探索的データ分析プロジェクト。

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

### 3. データのダウンロード

```bash
curl -L "https://docs.google.com/spreadsheets/d/1cJdocFdLpZ7hNUi4fTFvQK4ovIWYruqQ_nCuXR_w3Lw/export?format=csv&gid=0" -o data/titanic.csv
```

## 使用方法

marimoアプリの起動:

```bash
marimo edit notebooks/titanic_analysis.py
```

ブラウザが自動的に開き、インタラクティブな分析環境が利用できます。

### ポート指定

別のポートでアプリを実行する場合:

```bash
marimo edit --port 2719 notebooks/titanic_analysis.py
```

## 使用技術

- **marimo**: インタラクティブなPythonノートブック環境
- **polars**: 高速なデータフレーム操作ライブラリ
- **altair**: 宣言的な可視化ライブラリ

## プロジェクト構造

```
.
├── .venv/                      # Python仮想環境
├── data/
│   └── titanic.csv             # Titanicデータセット
├── notebooks/
│   └── titanic_analysis.py     # Titanic分析アプリケーション
├── requirements.txt            # パッケージ依存関係
└── README.md                   # このファイル
```

## 分析内容

### データの概要

- 乗客数: 891人
- データカラム数: 11列
- 生存者: 342人（38.4%）

### 実施する分析

1. **基本統計量**
   - 年齢、運賃などの数値データの統計情報

2. **欠損値の確認**
   - 各カラムの欠損値数と割合

3. **生存率の分析**
   - 全体生存率
   - 性別ごとの生存率
   - 客室クラスごとの生存率

4. **可視化**
   - 性別・クラス別の生存者数（棒グラフ）
   - 年齢分布と生存状況（ヒストグラム）
   - 年齢と運賃の関係（インタラクティブ散布図）

## 重要な発見

上記の分析から以下の傾向が読み取れます:

- **性別による生存率の差**: 女性の生存率が男性より顕著に高い（当時の「Women and children first」ルール）
- **客室クラスによる差**: 上位クラスの乗客の生存率が高い
- **年齢による影響**: 子どもや高齢者で異なるパターンが見られる
- **運賃と生存率**: 高い運賃（上位クラス）の乗客の生存率が高い傾向

## トラブルシューティング

### 環境確認の基本

```bash
# 仮想環境が有効化されていることを確認
which python
# 期待値: /path/to/project/.venv/bin/python

# パッケージがインストールされていることを確認
pip list | grep -E "marimo|polars|altair|pyarrow|pandas"

# データファイルの確認
ls -lh data/titanic.csv
head -3 data/titanic.csv
```

### 【重要】Apple Silicon + Rosettaでのクラッシュ（macOS M1/M2/M3）

**症状**: marimoが起動してもPythonカーネルが予期せず終了する

**原因**: x86_64 Pythonで通常の `polars` パッケージを実行すると、CPU互換性の問題でセグメンテーションフォルトが発生

**解決方法**:

1. requirements.txtが `polars-lts-cpu` になっていることを確認
   ```bash
   grep "polars" requirements.txt
   # 出力: polars-lts-cpu>=0.20.0
   ```

2. `polars` と `polars-runtime-32` がインストールされていないか確認
   ```bash
   pip list | grep polars
   ```

3. 古いpolarsが残っている場合はアンインストール
   ```bash
   pip uninstall -y polars polars-runtime-32
   pip install polars-lts-cpu
   ```

**推奨事項**: ネイティブArm64 Pythonの使用（根本解決）
```bash
# Homebrewでネイティブ版Pythonをインストール
brew install python@3.13  # Apple Silicon対応版が自動選択される
```

### 【重要】marimoの変数再定義エラー

**症状**: `This cell redefines variables from other cells` エラーが表示される

**原因**: marimoでは同じ変数を複数のセルで定義することが許可されていません（依存関係管理のため）

**解決方法**:
- 共通データ変換は専用セルで一度だけ実行
- 各セルは異なる出力変数を定義
- 例: pandas変換は1つのセル `df_pandas = df.to_pandas()` で行い、複数のグラフセルがそれを使用

**このプロジェクトの例**:
```python
# 【共通セル】可視化用データの準備
@app.cell
def _(df):
    df_pandas = df.to_pandas()
    return df_pandas,

# 【グラフセル1】このセルでは df_pandas を引数として受け取る
@app.cell
def _(alt, df_pandas):  # df ではなく df_pandas を使用
    chart = alt.Chart(df_pandas).mark_bar().encode(...)
    return chart,
```

### 依存パッケージの確認

このプロジェクトは以下のパッケージに依存しています：

| パッケージ | 用途 | 注記 |
|-----------|------|------|
| marimo | ノートブック環境 | |
| polars-lts-cpu | データ処理 | Apple Silicon互換版を使用 |
| altair | グラフ可視化 | |
| pyarrow | Polars↔Pandas変換 | **必須** |
| pandas | Altair用データフォーマット | **必須** |

**不足時のエラー例**:
```python
# ModuleNotFoundError: No module named 'pyarrow'
df_pandas = df.to_pandas()  # PyArrowが必要

# NameError: Pandas not available
alt.Chart(df_pandas)  # Altairはpandasベース
```

### marimoが起動しない場合

1. **ポート競合の場合**
   ```bash
   # ポート2718がすでに使用中の場合
   marimo edit --port 2719 notebooks/titanic_analysis.py
   ```

2. **キャッシュ問題の場合**
   ```bash
   rm -rf .marimo
   marimo edit notebooks/titanic_analysis.py
   ```

3. **CPUチェックスキップが必要な場合**（x86_64環境での警告が出ている場合）
   ```bash
   # 環境変数を設定（一時的）
   POLARS_SKIP_CPU_CHECK=1 marimo edit notebooks/titanic_analysis.py
   ```

### グラフが表示されない場合

- ブラウザのキャッシュをクリア: `Cmd+Shift+R` (macOS) / `Ctrl+Shift+R` (Linux/Windows)
- JavaScriptが有効になっているか確認
- ブラウザのコンソール（DevTools）でエラーを確認

### セル実行エラーが出た場合

1. **欠損値関連のエラー**: `dropna(subset=[...])` を使用して事前フィルタリング
2. **型エラー**: Polarsの型を確認 `df.schema`
3. **メモリ不足**: データサイズを確認 `df.shape` `df.estimated_size()`

### 推奨されない対応

❌ **`POLARS_SKIP_CPU_CHECK=1` を常用する**
- 根本解決ではなく、エラーを隠蔽するだけ
- クラッシュのリスクが残る

❌ **通常の `polars` パッケージをApple Siliconで使用する**
- CPU互換性の問題が発生

✓ **推奨**: polars-lts-cpu またはネイティブArm64 Pythonを使用

## 拡張の可能性

このプロジェクトはさらに以下のように拡張できます:

- **機械学習モデルの追加**: scikit-learnで生存予測モデルを構築
- **パラメータの動的変更**: `mo.ui.slider()` でビン数を調整可能に
- **複数データセットの比較**: タブやドロップダウンで切り替え
- **エクスポート機能**: HTMLやPDFでのレポート生成

## ライセンス

MIT License

## 参考

- [Marimo Documentation](https://marimo.io)
- [Polars Documentation](https://docs.pola-rs.com)
- [Altair Documentation](https://altair-viz.github.io)
- [Titanic Dataset](https://www.kaggle.com/c/titanic)
