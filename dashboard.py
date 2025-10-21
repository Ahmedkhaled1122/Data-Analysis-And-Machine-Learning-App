import numpy as np
import pandas as pd
import plotly.express as px

def bar_chart(df, x, y, sort, color=None):
    df_grouped = df.groupby([x, color], as_index=False)[y].sum() if color else df.groupby(x, as_index=False)[y].sum()
    if sort in ["ascending", "descending"]:
        df_grouped = df_grouped.sort_values(by=y, ascending=(sort == "ascending"))

    title = f'{y} by {x}' + (f" and {color}" if color else "")
    fig = px.bar(df_grouped, x=x, y=y, color=color, title=title, barmode='group')
    fig.update_layout(xaxis_title=x, yaxis_title=y, template="plotly_white", margin=dict(l=20, r=2, t=21))
    return fig

def barh_chart(df, x, y, sort, color=None):
    df_grouped = df.groupby([x, color], as_index=False)[y].sum() if color else df.groupby(x, as_index=False)[y].sum()

    if sort in ["ascending", "descending"]:
        df_grouped = df_grouped.sort_values(by=y, ascending=(sort == "ascending"))

    title = f'{y} by {x}' + (f" and {color}" if color else "")
    fig = px.bar(df_grouped, x=y, y=x, color=color, title=title, orientation="h", barmode='group')
    fig.update_layout(xaxis_title=y, yaxis_title=x, template="plotly_white", margin=dict(l=20, r=2, t=21))
    return fig

def line_chart(df, x, y, color=None):
    group = [x, color] if color else x
    df_grouped = df.groupby(group, as_index=False)[y].sum()

    title = f'{y} by {x}' + (f" and {color}" if color else "")
    fig = px.line(df_grouped, x=x, y=y, color=color, title=title, markers=True)
    fig.update_layout(xaxis_title=x, yaxis_title=y, template="plotly_white", margin=dict(l=20, r=2, t=21))
    return fig

def scatter_chart(df, x, y, color=None):
    title = f"{y} vs {x}" + (f" by {color}" if color else "")
    fig = px.scatter(df, x=x, y=y, color=color, title=title)
    fig.update_layout(xaxis_title=x, yaxis_title=y, template="plotly_white", margin=dict(l=20, r=2, t=21))
    return fig

def violin_chart(df, x, y, color=None):
    title = f'{y} by {x}' + (f" and {color}" if color else "")
    fig = px.violin(df, x=x, y=y, color=color, title=title, box=True, points="all")
    fig.update_layout(xaxis_title=x, yaxis_title=y, template="plotly_white", margin=dict(l=20, r=2, t=21))
    return fig

def histogram_chart(df, x, color=None):
    title = f"Distribution of {x}" + (f" by {color}" if color else "")
    fig = px.histogram(df, x=x, color=color, barmode="overlay", title=title)
    fig.update_layout(xaxis_title=x, yaxis_title="Count", template="plotly_white", margin=dict(l=20, r=2, t=21))
    return fig

def pie_chart(df, names, values):
    title = f"{values} distribution by {names}"
    fig = px.pie(df, names=names, values=values, title=title)
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=2, t=21))
    return fig

def donut_chart(df, names, values):
    title = f"{values} distribution by {names}"
    fig = px.pie(df, names=names, values=values, hole=0.4, title=title)
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=2, t=21))
    return fig

def treemap_chart(df, path, values, color=None):
    title = f"{values} treemap by {' - '.join(path)}"
    fig = px.treemap(df, path=path, values=values, color=color, title=title)
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=2, t=21))
    return fig

def sunburst_chart(df, path, values, color=None):
    title = f"{values} sunburst by {' - '.join(path)}"
    fig = px.sunburst(df, path=path, values=values, color=color, title=title)
    fig.update_traces(textinfo="label+percent entry")
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=2, t=21))
    return fig
