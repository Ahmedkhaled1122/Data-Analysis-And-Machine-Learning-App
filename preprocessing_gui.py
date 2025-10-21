import numpy as np
import pandas as pd
import streamlit as st
import preprocessing as pp
from streamlit_sortables import sort_items

def preprocessing_page():
    st.set_page_config(page_title="Preprocessing", layout="wide")
    st.title("Preprocessing")

def sort_values(df, col):
        sort_item = sort_items(df[col].unique().tolist(), direction="vertical", key=f'sort_item_{col}')
        return sort_item

def encoding(df):
    st.header("Encoding")
    non_numerical_columns = df.select_dtypes(exclude=['number']).columns
    columns = st.multiselect('Select Columns for Encoding', options=non_numerical_columns, default=non_numerical_columns)

    for col in columns:
        st.markdown(f"### Column: {col} -> unique values {df[col].nunique()}")
        encode = st.selectbox('Select Encoder', options=['label encoder', 'ordinal encoder', 'one hot encoder', 'frequency encoder'], key=f'encoder_{col}')

        if encode == 'label encoder':
            df = pp.label_encoder(df, col)
            st.success(f'Label Encoding applied successfully to column: {col}')

        elif encode == 'ordinal encoder':
            sort = sort_values(df, col)
            df = pp.ordinal_encoder(df, col, sort)
            st.success(f'Ordinal Encoding applied successfully to column: {col}')


        elif encode == 'one hot encoder':
            df = pp.one_hot_encoder(df, col)
            st.success(f'One Hot Encoding applied successfully to column: {col}')

        else:
            df = pp.frequency_encoder(df, col)
            st.success(f'Frequency Encoding applied successfully to column: {col}')

    st.dataframe(df, height=213)
    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    return df


def show_skewness(df):
    continuous_columns = pp.detect_skewness(df, 0)
    continuous_columns_copy = df[continuous_columns].copy()

    st.header('Handle Skewness')

    cols = st.columns(4)
    for i, col in enumerate(continuous_columns):
        with cols[i%4]:
            fig = pp.histogram(df, col)
            st.pyplot(fig)

    continuous_columns_skewness = pp.detect_skewness(df, 1)

    select_columns = st.multiselect('Select Columns To Handle Skewness', options=continuous_columns, default=continuous_columns_skewness)
    select_method = st.selectbox('Select transformation method', options=['Log Transformation', 'Box Cox Transformation', 'Yeojohnson Transformation'])
    if select_method == 'Log Transformation':
        for col in select_columns:
            df = pp.log_transform(df, col)
    elif select_method == 'Box Cox Transformation':
        for col in select_columns:
            if df[col].min() > 0:
                df = pp.box_cox_transform(df, col)
    else:
        for col in select_columns:
            df = pp.yeojohnson_transform(df, col)
        
    st.success(f'Skewness handling completed by {select_method} for {select_columns}')

    col1, col2 = st.columns(2)
    for col in select_columns:
        with col1:
            st.subheader('Before Skewness Handling')
            fig = pp.histogram(continuous_columns_copy, col)
            st.pyplot(fig)

        with col2:
            st.subheader('After Skewness Handling')
            fig = pp.histogram(df, col)
            st.pyplot(fig)

    return df

def scaler(df):
    st.subheader('Feature Scaling')
    numeric_cols = df.select_dtypes(include=['number']).columns
    col1, col2 = st.columns(2)
    with col1:
        columns = st.multiselect('Select columns to scale', options=numeric_cols, default=numeric_cols)
    with col2:
        method = scale_method = st.selectbox("Select scaling method", options=['Normalization (MinMaxScaler)', 'Standardization (StandardScaler)'])
    
    for col in columns:
        if method == 'Normalization (MinMaxScaler)':
            pp.min_max_scaler(df, col)
        else:
            pp.standerd_scaler(df, col)

    st.success(f'Scaler handling completed by {method} for {columns}')
    st.dataframe(df, height=213)
    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df

def correlation(df):
    st.header('Correlations')
    target_column = st.selectbox('Select Target Column', options=df.select_dtypes(include='number').columns)
    corrs = {}
    for col in df.select_dtypes(include='number').columns:

        corrs [col] = pp.detect_corr(df, target_column, col)

    corrs = pd.Series(corrs).sort_values(ascending=True)
    cols = st.columns(3)

    st.subheader('Correlations per columns')
    chunks = [corrs.iloc[i::3] for i in range(3)]

    cols = st.columns(3)

    for col, chunk in zip(cols, chunks):
        col.table(chunk.to_frame(name="Correlations Per Column"))
    
    default_columns = corrs[corrs.abs() < 0.1].index.tolist() + df.select_dtypes(exclude='number').columns.tolist()
    columns_to_drop = st.multiselect("Drop Unnecessary Columns And Low Correlation With Target", options=df.columns, default=default_columns)

    df = pp.delete_low_corr(df, columns_to_drop)

    st.success('successfully Droped Unnecessary Columns And Low Correlation With Target')
    st.dataframe(df, height=213)
    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    return df
