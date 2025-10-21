import pandas as pd
import numpy as np
import streamlit as st
import data_loader as dl
import os

def home_page():
    st.set_page_config(page_title="Data Analysis And Machine Learning Project", layout="wide")
    st.title("Data Analysis And Machine Learning Project")

def sidebar():
    st.sidebar.subheader('Data Input')
    upload_files = st.sidebar.file_uploader('Upload your dataset', type=['csv', 'xlsx'], accept_multiple_files=True)
    if upload_files:
        st.sidebar.success(f"Uploaded {len(upload_files)} files Successfully!")
        return upload_files

def get_data(upload_files):
    if not upload_files:
        st.stop()

    dfs = {}
    for file in upload_files:
        df_file = dl.load_data(file)
        dfs[os.path.splitext(file.name)[0]] = df_file

    return dfs

def display_datasets(dfs):
    for name, df in dfs.items():
        st.subheader(name)
        st.dataframe(df, height=213)
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")


def rename_columns(dfs):
    for name, df in dfs.items():
        st.subheader(f"Rename Columns of '{name}'")
        new_col = {}
        row_col = st.columns(7)
        for i, col in enumerate(df.columns):
            with row_col[i % 7]:
                new_name = st.text_input(col, value=col, key=f"{name}_{col}")
                new_col[col] = new_name

        dfs[name] = df.rename(columns=new_col)
        st.success(f"Columns renamed successfully ✅")
    return dfs

def change_datatype(dfs):
    for name, df in dfs.items():
        st.subheader(f'Count Of Values For {name}')
        value_count = dl.value_count(df)

        chunks = [value_count.iloc[i::3] for i in range(3)]

        cols = st.columns(3)
        
        for col, chunk in zip(cols, chunks):
            col.table(chunk.to_frame(name="Values Per Columns"))

        st.subheader(f"Change DataType For {name}")
        new_datatype = {}
        row_col = st.columns(7)
        for i, col in enumerate(df.columns):
            with row_col[i%7]:
                new_type = st.text_input(col, value=df.dtypes[col], key=f"{i}_{name}_{df.dtypes[col]}")
                new_datatype[col] = new_type
        
        dfs[name] = dl.change_datatype(df, new_datatype)
        st.success(f"Change datatype Cuccessfully ✅")
    return dfs


def append_datasets_gui(dfs):
    if len(dfs) >= 2:
        st.subheader('Append Datasets')
        name_files = st.multiselect('Select files to Append', dfs.keys())
        if len(name_files) <= 1:
            st.stop()
        if len(name_files) >= 2:
            select_df = [dfs[key] for key in name_files]
            df = dl.append_datasets(select_df)
            st.dataframe(df, height=213)
            st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.success("Datasets appended successfully ✅")
            return df, name_files
    return None

def joins_datasets(dfs, k):
    if len(dfs) >=2:
        st.subheader('Join Datasets')
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            type_join = st.selectbox('Select type of Join' ,[ 'Full Outer Join', 'Left Join', 'Right Join', 'Inner Join', 'Index Join'], key=f'type_join_{k}')
        with col2:
            select_df1 = st.selectbox('Select first dataset', dfs.keys(), key=f'select_df1_{k}')
        with col3:
            select_df2 = st.selectbox('Select second dataset', [key for key in dfs.keys() if key != select_df1], key=f'select_df2_{k}')
        if type_join != 'Index Join':
            with col4:
                left_join = st.selectbox('Select join column from first dataset', dfs[select_df1].columns, key=f'left_join_{k}')
            with col5:
                right_join = st.selectbox('Select join column from second dataset', dfs[select_df2].columns, key=f'right_join_{k}')
        try:
            if type_join == 'Full Outer Join':
                df = dl.cross_join(dfs[select_df1], dfs[select_df2], left_join, right_join)
            if type_join == 'Left Join':
                df = dl.left_join(dfs[select_df1], dfs[select_df2], left_join, right_join)
            if type_join == 'Right Join':
                df = dl.right_join(dfs[select_df1], dfs[select_df2], left_join, right_join)
            if type_join == 'Inner Join':
                df = dl.inner_join(dfs[select_df1], dfs[select_df2], left_join, right_join)
            if type_join == 'Index Join':
                df = dl.index_join(dfs[select_df1], dfs[select_df2])
        except Exception as e:
            st.error(f"❌ Error during join operation: {str(e)}")
            st.stop()

        if df is not None:
            st.dataframe(df, height=213)
            st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.success("Datasets joins successfully ✅")

    return df, select_df1, select_df2


def append_or_join(dfs):
    if len(dfs) == 1:
        return list(dfs.values())[0]
        
    temp = None
    while(len(dfs) >= 2):
        st.header('Append or Join')
        append_or_join = st.selectbox('Select option', ['Choose Option', 'Append Datasets', 'Join Datasets'], key=f'{len(dfs)}')
        if append_or_join == "Append Datasets":
            df, select_name = append_datasets_gui(dfs)
            for name in select_name:
                dfs.pop(name)
            new_name = st.text_input('Name for appended dataset:', f'appended_{"_".join(select_name)}')
            dfs[new_name] = df
            st.success(f"Dataset saved successfully as: **{new_name}** ✅")

        if append_or_join == "Join Datasets":
            df, df1, df2 = joins_datasets(dfs, len(dfs))
            dfs.pop(df1)
            dfs.pop(df2)
            new_name = st.text_input('Name for joined dataset:', f'joined_{df1}_{df2}')
            dfs[new_name] = df
            st.success(f"Dataset saved successfully as: **{new_name}** ✅")
        
        if append_or_join == 'Choose Option':
            st.stop()
            
        temp = new_name

    return dfs[temp]