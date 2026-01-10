import marimo

__generated_with = "0.19.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    from pathlib import Path
    return Path, alt, mo, pl


@app.cell
def _(Path, pl):
    # データロード
    data_path = Path(__file__).parent.parent / "data" / "titanic.csv"
    df = pl.read_csv(data_path)
    return (df,)


@app.cell
def _(df, mo):
    # データセットの概要
    mo.md(f"""
    # Titanic データセット分析

    データサイズ: {df.shape[0]} 行 × {df.shape[1]} 列
    """)
    return


@app.cell
def _(mo):
    # 基本統計量
    mo.md("""
    ## 基本統計量
    """)
    return


@app.cell
def _(df):
    # 基本統計量の表示
    stats_display = df.describe()
    stats_display
    return


@app.cell
def _(mo):
    # 欠損値チェック
    mo.md("""
    ## 欠損値確認
    """)
    return


@app.cell
def _(df, pl):
    # 欠損値の数と割合を計算
    null_counts = df.null_count()
    total_rows = df.shape[0]

    # 表示用に整形
    null_info = pl.DataFrame({
        "カラム": list(null_counts.columns),
        "欠損数": [null_counts[col][0] for col in null_counts.columns],
        "欠損率(%)": [
            round(null_counts[col][0] * 100 / total_rows, 2)
            for col in null_counts.columns
        ]
    })

    null_info
    return


@app.cell
def _(mo):
    # 生存率の計算
    mo.md("""
    ## 生存率の計算
    """)
    return


@app.cell
def _(df, mo, pl):
    # 全体の生存率
    total_survived = df.filter(pl.col("Survived") == 1).shape[0]
    total_passengers = df.shape[0]
    survival_rate = total_survived / total_passengers * 100

    mo.md(f"""
    **全体の生存率:** {survival_rate:.1f}% ({total_survived}/{total_passengers}人)
    """)
    return


@app.cell
def _(df, mo, pl):
    # 性別ごとの生存率
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

    mo.md("### 性別ごとの生存率")
    survival_by_sex
    return


@app.cell
def _(df, mo, pl):
    # 客室クラスごとの生存率
    survival_by_class = (
        df.group_by("Pclass")
        .agg([
            (pl.col("Survived") == 1).sum().alias("生存者数"),
            pl.count().alias("総人数")
        ])
        .with_columns([
            (pl.col("生存者数") / pl.col("総人数") * 100).alias("生存率(%)")
        ])
        .sort("Pclass")
    )

    mo.md("### 客室クラスごとの生存率")
    survival_by_class
    return


@app.cell
def _(mo):
    mo.md("""
    ## 可視化
    """)
    return


@app.cell
def _(df):
    # pandas DataFrameへの変換（可視化用）
    df_pandas = df.to_pandas()
    return df_pandas,


@app.cell
def _(alt, df_pandas):
    # 性別・生存状況の可視化
    chart_sex = alt.Chart(df_pandas).mark_bar().encode(
        x=alt.X('Sex:N', title='性別'),
        y=alt.Y('count()', title='人数'),
        color=alt.Color('Survived:N',
                       scale=alt.Scale(domain=[0, 1],
                                     range=['#e74c3c', '#2ecc71']),
                       title='生存')
    ).properties(
        width=400,
        height=300,
        title='性別ごとの生存者数'
    )

    chart_sex
    return chart_sex,


@app.cell
def _(alt, df_pandas):
    # クラス別の生存率
    chart_class = alt.Chart(df_pandas).mark_bar().encode(
        x=alt.X('Pclass:O', title='客室クラス'),
        y=alt.Y('count()', title='人数'),
        color=alt.Color('Survived:N',
                       scale=alt.Scale(domain=[0, 1],
                                     range=['#e74c3c', '#2ecc71']),
                       title='生存')
    ).properties(
        width=400,
        height=300,
        title='客室クラスごとの生存者数'
    )

    chart_class
    return chart_class,


@app.cell
def _(alt, df_pandas):
    # 年齢分布のヒストグラム
    chart_age = alt.Chart(df_pandas.dropna(subset=['Age'])).mark_bar(opacity=0.7).encode(
        x=alt.X('Age:Q', bin=alt.Bin(maxbins=30), title='年齢'),
        y=alt.Y('count()', title='人数'),
        color=alt.Color('Survived:N',
                       scale=alt.Scale(domain=[0, 1],
                                     range=['#e74c3c', '#2ecc71']),
                       title='生存')
    ).properties(
        width=700,
        height=300,
        title='年齢分布と生存状況'
    )

    chart_age
    return chart_age,


@app.cell
def _(alt, df_pandas):
    # 運賃と年齢の散布図（生存状況別）
    # 欠損値と異常値を除外
    df_filtered = df_pandas.dropna(subset=['Age', 'Fare'])
    df_filtered = df_filtered[df_filtered['Fare'] > 0]

    chart_fare_age = alt.Chart(df_filtered).mark_circle(size=60).encode(
        x=alt.X('Age:Q', title='年齢'),
        y=alt.Y('Fare:Q', title='運賃', scale=alt.Scale(type='log')),
        color=alt.Color('Survived:N',
                       scale=alt.Scale(domain=[0, 1],
                                     range=['#e74c3c', '#2ecc71']),
                       title='生存'),
        tooltip=['PassengerId', 'Name', 'Age', 'Fare', 'Pclass', 'Sex', 'Survived']
    ).properties(
        width=700,
        height=400,
        title='年齢と運賃の関係（生存状況別）'
    ).interactive()

    chart_fare_age
    return chart_fare_age,


@app.cell
def _(mo):
    mo.md("""
    ## 分析サマリー

    上記のグラフから以下の傾向が読み取れます:

    - **性別による生存率の差**: 女性の生存率が男性より顕著に高い
    - **客室クラスによる差**: 上位クラス（1等、2等）の乗客の生存率が高い傾向
    - **年齢による影響**: 子どもや高齢者で異なる生存率パターンが見られる
    - **運賃と生存率**: 運賃が高い（上位クラスに多い）乗客の生存率が高い傾向

    これらは当時の「Women and children first」という救助優先ルールが実施されたことを示唆しています。
    """)
    return


if __name__ == "__main__":
    app.run()
