import pandas as pd
import numpy as np
import streamlit as st
import data_loader_gui as dlg
import data_cleaning_gui as dcg
import analysis_gui as anag
import dashboard_gui as dbg
import preprocessing_gui as ppg
import model_gui as mg

page = st.sidebar.radio("Choose Page", ["Home", "Data Cleaning", "Analysis and Visualization", 'DashBoard', 'PreProcessing', 'Machine Learning'])

if page == 'Home':
    dlg.home_page()

    upload_files = dlg.sidebar()
    dfs = dlg.get_data(upload_files)
    dlg.display_datasets(dfs)

    dfs = dlg.rename_columns(dfs)
    dfs = dlg.change_datatype(dfs)

    if len(dfs) >= 2:
        df = dlg.append_or_join(dfs)
    else:
        df = list(dfs.values())[0]

    st.session_state["load_df"] = df

if page == 'Data Cleaning':

    dcg.data_cleaning_page()
    if 'load_df' in st.session_state:
        df = st.session_state['load_df']
    else:
        st.warning('Finish all operation in home page ⚠️')
        st.stop()

    temp = dcg.missing_values(df)
    if temp != 'No missing values':
        label ='With Missing Values >= 40 %'
        default_missing = dcg.default_missing_values_selected(df)
        df = dcg.drop_columns_gui(df, default_missing, 1, label )
        df_copy = df.copy()
        df = dcg.handle_missing_value(df_copy)
        st.dataframe(df, height=213)


    df = dcg.duplicated_values(df)

    dcg.display_variance(df)
    label ='With Low Variance < 1'
    default_variance = dcg.default_low_variance_selected(df)
    df = dcg.drop_columns_gui(df, default_variance, 2, label )

    dcg.handle_outliers(df)

    st.session_state['Data_Cleaning'] = df

if page == 'Analysis and Visualization':
    anag.EDA_page()
    if 'Data_Cleaning' in st.session_state:
        df = st.session_state['Data_Cleaning']
    else:
        st.warning('Finish all operation in Data Cleaning page ⚠️')
        st.stop()

    st.dataframe(df, height=213)
    anag.pivot_table_analysis(df)

if page == 'DashBoard':
    dbg.dashboard_page()
    if 'Data_Cleaning' in st.session_state:
        df = st.session_state['Data_Cleaning']
    else:
        st.warning('Finish all operation in Data Cleaning page ⚠️')
        st.stop()

    dbg.dashboard(df)

if page == 'PreProcessing':
    ppg.preprocessing_page()
    if 'Data_Cleaning' in st.session_state:
        df = st.session_state['Data_Cleaning']
    else:
        st.warning('Finish all operation in Data Cleaning page ⚠️')
        st.stop()

    df_copy = df.copy()
    df = ppg.encoding(df_copy)
    df = ppg.show_skewness(df)
    df = ppg.scaler(df)
    df = ppg.correlation(df)

    st.session_state['Preprocessing'] = df

if page == 'Machine Learning':
    if 'Preprocessing' in st.session_state:
        df = st.session_state['Preprocessing']
    else:
        st.warning('Finish all operation in Preprocessing page ⚠️')
        st.stop()
    mg.model_page()
    mg.prepare_data(df)

