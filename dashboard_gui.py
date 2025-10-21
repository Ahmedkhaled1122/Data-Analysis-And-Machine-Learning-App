import numpy as np
import pandas as pd
import streamlit as st
import dashboard as db

def dashboard_page():
    st.set_page_config(page_title="DashBoard", layout="wide")
    st.title("DashBoard")

def dashboard_gui(df, k):
    with st.sidebar:
        slicers = st.multiselect(f'Select Slicers DashBoard {k}', df.select_dtypes(exclude=['number']).columns, key=f'slicer_{k}')
        for slicer in slicers:
            st.subheader(slicer + " slicer")
            cols = st.columns(2)
            selected_countries = []
            for i, slice in enumerate(df[slicer].unique()):
                col = cols[i%2]
                with col:
                    if st.checkbox(slice, value=True, key=f"check_box_{slicer}_{k}_{i}"):
                        selected_countries.append(slice)

            df = df[df[slicer].isin(selected_countries)]


    col1, col2, col3, col4 = st.columns(4)

    cols = df.columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns

    def format_value(val, scale):
        if not isinstance(val, (int, float, np.integer, np.floating)):
            return val
        if scale == "K":
            return str(round(val / 1e3, 2)) + "K"
        elif scale == "M":
            return str(round(val / 1e6, 2)) + "M"
        elif scale == "B":
            return str(round(val / 1e9, 2)) + "B"
        else:
            return str(round(val, 2))

    def render_card(df, key, k):
        column = st.selectbox(f"Select Column ({key})", cols, key=f"col_{key}_{k}")
        
        col1, col2 = st.columns(2)
        if column in numeric_cols:
            with col1:
                agg = st.selectbox(f"Agg of {column}", ["sum", "mean", "max", "min", "count", "nunique", "mode"], key=f"agg_{key}_{k}")
            value = getattr(df[column], agg)()
        else:
            with col1:
                agg = st.selectbox(f"Agg of {column}", ["count", "nunique", "mode"], key=f"agg_{key}_{k}")
            if agg == "mode":
                modes = df[column].mode()
                value = ", ".join(map(str, modes)) if not modes.empty else "N/A"
            else:
                value = getattr(df[column], agg)()
        
        with col2:
            scale = st.selectbox(f"Format Number ({key})", ["Natural", "K", "M", "B"], key=f"scale_{key}_{k}")
        value = format_value(value, scale)
        return f"{agg} of {column}", value

    def render_chart(df, key, k):
        chart = st.selectbox(
            f'Select Chart ({key})',
            ['Choose Chart', 'Bar', 'Barh', 'Line', 'Scatter', 'Histogram', 'Violin', 'Pie', 'Donut', 'Treemap', 'Sunburst'],
            key=f"chart_{key}_{k}"
        )
        if chart == 'Choose Chart':
            return

        if chart in ['Bar', 'Barh']:
            xaxis = st.selectbox("Select X-axis", df.columns, key=f"xaxis_{key}_{k}")
            yaxis = st.selectbox("Select Y-axis", df.select_dtypes(include=['number']).columns, key=f"yaxis_{key}_{k}")
            col1, col2 = st.columns(2)
            with col1:
                color = st.selectbox("Based On (optional)", [None] + [col for col in df.columns if col not in xaxis], key=f"color_{key}_{k}")
            with col2:
                sort = st.selectbox('Sort', ['Sorted By', 'descending', 'ascending'], key=f'sort_{key}_{k}')
        
            if chart == "Bar":
                fig = db.bar_chart(df, x=xaxis, y=yaxis, color=color, sort=sort)
            elif chart == 'Barh':
                fig = db.barh_chart(df, x=xaxis, y=yaxis, color=color, sort=sort)

        elif chart in ['Line', 'Scatter', 'Violin']:
            xaxis = st.selectbox("Select X-axis", df.columns, key=f"xaxis_{key}_{k}")
            yaxis = st.selectbox("Select Y-axis", df.select_dtypes(include=['number']).columns, key=f"yaxis_{key}_{k}")
            color = st.selectbox("Based On (optional)", [None] + [col for col in df.columns if col not in xaxis], key=f"color_{key}_{k}")

            if chart == "Line":
                fig = db.line_chart(df, x=xaxis, y=yaxis, color=color)
            elif chart == "Scatter":
                fig = db.scatter_chart(df, x=xaxis, y=yaxis, color=color)
            elif chart == "Violin":
                fig = db.violin_chart(df, x=xaxis, y=yaxis, color=color)

        elif chart == "Histogram":
            col = st.selectbox("Select Column", df.select_dtypes(include=['number']).columns, key=f"hist_{key}_{k}")
            fig = db.histogram_chart(df, x=col)

        elif chart == "Pie":
            names = st.selectbox("Names", df.select_dtypes(exclude=['number']).columns, key=f"pie_names_{key}_{k}")
            values = st.selectbox("Values", df.select_dtypes(include=['number']).columns, key=f"pie_values_{key}_{k}")
            fig = db.pie_chart(df, names=names, values=values)

        elif chart == "Donut":
            names = st.selectbox("Names", df.select_dtypes(exclude=['number']).columns, key=f"donut_names_{key}_{k}")
            values = st.selectbox("Values", df.select_dtypes(include=['number']).columns, key=f"donut_values_{key}_{k}")
            fig = db.donut_chart(df, names=names, values=values)

        elif chart == "Treemap":
            path = st.multiselect("Hierarchy", df.select_dtypes(exclude=['number']).columns, default=df.select_dtypes(exclude=['number']).columns[0], key=f"treemap_path_{key}_{k}")
            values = st.selectbox("Values", df.select_dtypes(include=['number']).columns, key=f"treemap_values_{key}_{k}")
            fig = db.treemap_chart(df, path=path, values=values)

        elif chart == "Sunburst":
            path = st.multiselect("Hierarchy", df.select_dtypes(exclude=['number']).columns, df.select_dtypes(exclude=['number']).columns[0], key=f"sunburst_path_{key}_{k}")
            values = st.selectbox("Values", df.select_dtypes(include=['number']).columns, key=f"sunburst_values_{key}_{k}")
            if not path:
                st.warning("Please select at least one hierarchy level.")
                return
            else:
                fig = db.sunburst_chart(df, path=path, values=values)

        return fig


    with col1:
        st.subheader('Card 1')
        label1, value1 = render_card(df, "Card1", k)
    with col2:
        st.subheader('Card 2')
        label2, value2 = render_card(df, "Card2", k)
    with col3:
        st.subheader('Card 3')
        label3, value3 = render_card(df, "Card3", k)
    with col4:
        st.subheader('Card 4')
        label4, value4 = render_card(df, "Card4", k)


    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader('Chart 1')
        chart1 = render_chart(df, 'chart1', k)
    with col2:
        st.subheader('Chart 2')
        chart2 = render_chart(df, 'chart2', k)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader('Chart 3')
        chart3 = render_chart(df, 'chart3', k)
    with col2:
        st.subheader('Chart 4')
        chart4 = render_chart(df, 'chart4', k)
    with col3:
        st.subheader('Chart 5')
        chart5 = render_chart(df, 'chart5', k)


    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label=label1, value=value1)
    with col2:
        st.metric(label=label2, value=value2)
    with col3:
        st.metric(label=label3, value=value3)
    with col4:
        st.metric(label=label4, value=value4)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(chart1, key=f'chart1_{k}', use_container_width=True) if chart1 else st.stop()
    with col2:
        st.plotly_chart(chart2, key=f'chart2_{k}', use_container_width=True) if chart2 else st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(chart3, key=f'chart3_{k}', use_container_width=True) if chart3 else st.stop()
    with col2:
        st.plotly_chart(chart4, key=f'chart4_{k}', use_container_width=True) if chart4 else st.stop()
    with col3:
        st.plotly_chart(chart5, key=f'chart5_{k}', use_container_width=True) if chart5 else st.stop()

def dashboard(df):
    count = st.number_input("How many DashBoards?", min_value=1, max_value=50, value=1)

    for i in range(1, count+1):
        with st.expander(f"DashBoard {i}"):
            dashboard_gui(df, i)