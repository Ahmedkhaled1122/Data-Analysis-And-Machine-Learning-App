import numpy as np
import pandas as pd
import streamlit as st
import analysis as ana

def EDA_page():
    st.set_page_config(page_title="Analysis and Visualization", layout="wide")
    st.title("Analysis and Visualization")

def pivot_table(df, key):
    st.header("Pivot Table")
    col1, col2, col3, col4 = st.columns(4)

    non_numeric_cols1 = df.select_dtypes(exclude=['number']).columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    with col1:
        columns = st.multiselect('Select columns', options=non_numeric_cols1, key=f'columns_{key}')
        non_numeric_cols2 = [col for col in non_numeric_cols1 if col not in columns]
    
    with col2:
        rows = st.multiselect('Select rows', options=non_numeric_cols2, key=f'rows_{key}')
    
    with col3:
        values = st.multiselect('Select values', options=numeric_cols, key=f'values_{key}')
        if values:
            agg = st.selectbox('Select aggregate function', options=['count', 'max', 'mean', 'median', 'min', 'std', 'sum', 'var'], key=f'agg_{key}')
    
    with col4:
        sort = st.selectbox('Sort Privot Table', ['Sorted By', 'descending', 'ascending'], key=f'sort_{key}')
    
    if (columns == [] and rows == []) or values == []:
        st.stop()
    
    pivot = ana.pivot_table(df, columns, rows, values, sort, agg)
    return pivot, columns, rows, values, sort, agg

def pivot_chart(df, columns, rows, values, sort, agg, key):
    st.subheader('Vizualization of Pivot Table')
    fig = ana.pivot_barhchart(df, columns, rows, values, sort, agg)
    if fig == '⚠️ Dimensions not supported!':
        st.warning(f'The Pivot Table Have {fig}')
    else:
        chart = st.selectbox('Choose Chart', options=["Bar", "Barh", "Line", "Pie"], key=f'chart_{key}')
        if chart == 'Bar':
            fig = ana.pivot_barchart(df, columns, rows, values, sort, agg)
            st.pyplot(fig)
        if chart == 'Barh':
            fig = ana.pivot_barhchart(df, columns, rows, values, sort, agg)
            st.pyplot(fig)
        if chart == 'Line':
            fig = ana.pivot_linechart(df, columns, rows, values, sort, agg)
            st.pyplot(fig)
        if chart == 'Pie':
            fig = ana.pivot_piechart(df, columns, rows, values)
            if fig =='⚠️ Dimensions not supported!':
                st.warning(f'The Pivot Table Have {fig}')
            else:
                st.pyplot(fig)

def pivot_table_analysis(df):
    count = st.number_input("How many Pivot Tables?", min_value=1, max_value=50, value=1)

    for i in range(1, count+1):
        with st.expander(f"Pivot Table {i}"):
            pivot, columns, rows, values, sort, agg = pivot_table(df, i)
            st.markdown(pivot.to_html(), unsafe_allow_html=True)
            pivot_chart(df, columns, rows, values, sort, agg, i)