import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def pivot_table(df, columns, rows, values, sort, aggfun):

    pivot = pd.pivot_table(df, values=values, index=columns, columns=rows, aggfunc=aggfun)

    if sort in ["ascending", "descending"]:
        if columns != []:
            pivot = pivot.sort_values(by=pivot.columns[0], ascending=(sort == "ascending"))
        else:
            pivot = pivot.T.sort_values(by=rows[0], ascending=(sort == "ascending"))
            pivot = pivot.T

    return pivot

def pivot_barchart(df, columns, rows, values, sort, agg):
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot = pd.pivot_table(df, values=values, index=columns + rows, aggfunc=agg)
    if sort in ["ascending", "descending"]:
        pivot = pivot.sort_values(by=pivot.columns[0], ascending=(sort == "ascending"))

    if len(columns) == 2 and len(rows) == 0 and len(values) > 0:
        sns.barplot(data=pivot, x=columns[0], y=values[0], hue=columns[1], ax=ax, palette="bright")
        ax.set_title(f"{values[0]} by {columns[0]}" + (f" and {columns[1]}" if columns[1] else ""))

    elif len(columns) == 0 and len(rows) == 2 and len(values) > 0:
        sns.barplot(pivot, x=rows[0], y=values[0], hue=rows[1], ax=ax, palette="bright")
        ax.set_title(f"{values[0]} by {rows[0]}" + (f" and {rows[1]}" if rows[1] else ""))

    elif len(columns) == 1 and len(rows) == 1 and len(values) > 0:
        sns.barplot(pivot, x=columns[0], y=values[0], hue=rows[0], ax=ax, palette="bright")
        ax.set_title(f"{values[0]} by {columns[0]}" + (f" and {rows[0]}" if rows[0] else ""))

    elif len(columns) == 1 and len(rows) == 0 and len(values) > 0:
        sns.barplot(pivot, x=columns[0], y=values[0], ax=ax, palette="bright")
        ax.set_title(f"{values[0]} by {columns[0]}")

    elif len(columns) == 0 and len(rows) == 1 and len(values) > 0:
        sns.barplot(df, x=rows[0], y=values[0], ax=ax, palette="bright")
        ax.set_title(f"{values[0]} by {rows[0]}")

    else:
        return "⚠️ Dimensions not supported!"

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    return fig

def pivot_barhchart(df, columns, rows, values, sort, agg):
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = pd.pivot_table(df, values=values, index=columns + rows, aggfunc=agg)
    if sort in ["ascending", "descending"]:
        pivot = pivot.sort_values(by=pivot.columns[0], ascending=(sort == "ascending"))

    if len(columns) == 2 and len(rows) == 0 and len(values) > 0:
        sns.barplot(pivot, y=columns[0], x=values[0], hue=columns[1], ax=ax, palette='bright')
        ax.set_title(f"{values[0]} by {columns[0]}" + (f" and {columns[1]}" if columns[1] else ""))

    elif len(columns) == 0 and len(rows) == 2 and len(values) > 0:
        sns.barplot(pivot, y=rows[0], x=values[0], hue=rows[1], ax=ax, palette="bright")
        ax.set_title(f"{values[0]} by {rows[0]}" + (f" and {rows[1]}" if rows[1] else ""))

    elif len(columns) == 1 and len(rows) == 1 and len(values) > 0:
        sns.barplot(pivot, y=columns[0], x=values[0], hue=rows[0], ax=ax, palette="bright")
        ax.set_title(f"{values[0]} by {columns[0]}" + (f" and {rows[0]}" if rows[0] else ""))

    elif len(columns) == 1 and len(rows) == 0 and len(values) > 0:
        sns.barplot(pivot, y=columns[0], x=values[0], ax=ax, palette="bright")
        ax.set_title(f"{values[0]} by {columns[0]}")

    elif len(columns) == 0 and len(rows) == 1 and len(values) > 0:
        sns.barplot(pivot, y=rows[0], x=values[0], ax=ax, palette="bright")
        ax.set_title(f"{values[0]} by {rows[0]}")

    else:
        return "⚠️ Dimensions not supported!"

    return fig

def pivot_linechart(df, columns, rows, values, sort, agg):
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot = pd.pivot_table(df, values=values, index=columns + rows, aggfunc=agg)

    if len(columns) == 2 and len(rows) == 0 and len(values) > 0:
        sns.lineplot(pivot, x=columns[0], y=values[0], hue=columns[1], ax=ax, palette="bright")
        ax.set_title(f"{values[0]} by {columns[0]}" + (f" and {columns[1]}" if columns[1] else ""))

    elif len(columns) == 0 and len(rows) == 2 and len(values) > 0:
        sns.lineplot(pivot, x=rows[0], y=values[0], hue=rows[1], ax=ax, palette="bright")
        ax.set_title(f"{values[0]} by {rows[0]}" + (f" and {rows[1]}" if rows[1] else ""))

    elif len(columns) == 1 and len(rows) == 1 and len(values) > 0:
        sns.lineplot(pivot, x=columns[0], y=values[0], hue=rows[0], ax=ax, palette="bright")
        ax.set_title(f"{values[0]} by {columns[0]}" + (f" and {rows[0]}" if rows[0] else ""))

    elif len(columns) == 1 and len(rows) == 0 and len(values) > 0:
        sns.lineplot(pivot, x=columns[0], y=values[0], ax=ax, palette="bright")
        ax.set_title(f"{values[0]} by {columns[0]}")

    elif len(columns) == 0 and len(rows) == 1 and len(values) > 0:
        sns.lineplot(pivot, x=rows[0], y=values[0], ax=ax, palette="bright")
        ax.set_title(f"{values[0]} by {rows[0]}")

    else:
        return "⚠️ Dimensions not supported!"

    return fig

def pivot_piechart(df, columns, rows, values):
    fig, ax = plt.subplots(figsize=(12, 12))
    colors = sns.color_palette('bright')

    if len(columns) == 1 and len(rows) == 0 and len(values) > 0:
        data = df.groupby(columns[0])[values[0]].sum()
        ax.pie(data, labels=data.index, colors=colors, autopct='%.0f%%', startangle=90, explode=[0.1] * len(data))
        ax.set_title(f"{values[0]} by {columns[0]}")

    elif len(columns) == 0 and len(rows) == 1 and len(values) > 0:
        data = df.groupby(rows[0])[values[0]].sum()
        ax.pie(data, labels=data.index, colors=colors, autopct='%.0f%%', startangle=90, explode=[0.1] * len(data))
        ax.set_title(f"{values[0]} by {rows[0]}")

    else:
        return "⚠️ Dimensions not supported!"

    return fig
