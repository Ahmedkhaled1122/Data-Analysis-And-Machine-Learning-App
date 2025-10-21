import numpy as np
import pandas as pd
import streamlit as st
import data_cleaning as dc

def data_cleaning_page():
    st.set_page_config(page_title="Data Cleaning", layout="wide")
    st.title("Data Cleaning")

def show_missing_values(df):
    st.subheader('Missing Values Per Columns')
    missing = dc.show_missing_value_persentage(df)
    missing = missing.apply(lambda x: f"{x:.3f} %")
    if missing is not None:
        st.table(missing.to_frame(name="Missing %"))
    return missing

def visual_missing_values(df):
    option = st.radio("Choose way for visualization", ("Matrix", "Heatmap", "Bar Chart"), horizontal=True)
    if option == "Matrix":
        fig = dc.missing_matrix(df)
        st.pyplot(fig)

    if option == "Heatmap":
        fig = dc.missing_heatmap(df)
        st.pyplot(fig)

    if option == "Bar Chart":
        fig = dc.missing_bar(df)
        st.pyplot(fig)

def missing_values(df):
    missing = dc.show_missing_value_persentage(df)
    if missing is not None:
        col1, col2 = st.columns(2)
        with col1:
            show_missing_values(df)
        with col2:
            visual_missing_values(df)
    else:
        st.success("No missing values in the dataset. ✅")
        return "No missing values"


def handle_missing_value(df):
    imputation_settings = {}

    missing_cols = df.columns[df.isnull().any()].tolist()
        
    for miss_col in missing_cols:
        st.markdown(f"### Column: {miss_col}")
        st.write(f"- Current missing values: {df[miss_col].isnull().sum()} ({round(df[miss_col].isnull().mean()*100, 2)}%)")
        
        strategy = None
        fill_value = None
        n_neighbors = None
        max_iter = None
        
        col_type = "numeric" if pd.api.types.is_numeric_dtype(df[miss_col]) else "categorical"
        
        col1, col2 = st.columns(2)
        
        with col1:
            if col_type == "numeric":
                method = st.selectbox(
                    f"Method for {miss_col}",
                    ["Simple Imputer", "KNN Imputer", "Iterative Imputer", "Fill with value"],
                    key=f"method_{miss_col}"
                )
            else:
                method = st.selectbox(
                    f"Method for {miss_col}",
                    ["Simple Imputer", "Fill with value"],
                    key=f"method_{miss_col}"
                )
        
        with col2:
            if method == "Simple Imputer":
                if col_type == "numeric":
                    strategy = st.selectbox(
                        f"Strategy for {miss_col}",
                        ["mean", "median", "most_frequent", "constant"],
                        key=f"strategy_{miss_col}"
                    )
                    if strategy == "constant":
                        fill_value = st.number_input(
                            f"Fill value for {miss_col}",
                            value=0,
                            key=f"fill_{miss_col}"
                        )
                else:
                    strategy = st.selectbox(
                        f"Strategy for {miss_col}",
                        ["most_frequent", "constant"],
                        key=f"strategy_{miss_col}"
                    )
                    if strategy == "constant":
                        fill_value = st.text_input(
                            f"Fill value for {miss_col}",
                            value="UNKNOWN",
                            key=f"fill_{miss_col}"
                        )
            
            elif method == "Fill with value":
                if col_type == "numeric":
                    fill_value = st.number_input(
                        f"Fill value for {miss_col}",
                        value=0,
                        key=f"fill_{miss_col}"
                    )
                else:
                    fill_value = st.text_input(
                        f"Fill value for {miss_col}",
                        value="UNKNOWN",
                        key=f"fill_{miss_col}"
                    )
            
            elif method == "KNN Imputer":
                n_neighbors = st.number_input(
                    f"Number of neighbors for {miss_col}",
                    min_value=1,
                    value=5,
                    key=f"neighbors_{miss_col}"
                )
            
            elif method == "Iterative Imputer":
                max_iter = st.number_input(
                    f"Max iterations for {miss_col}",
                    min_value=1,
                    value=10,
                    key=f"iter_{miss_col}"
                )
        
        imputation_settings[miss_col] = {
            "type": col_type,
            "method": method,
            "strategy": strategy,
            "fill_value": fill_value,
            "n_neighbors": n_neighbors,
            "max_iter": max_iter
        }
        
    simple_numeric = []
    simple_categorical = []
    fill_values = {}
    knn_cols = []
    iterative_cols = []

    for col, settings in imputation_settings.items():
        if settings["method"] == "Simple Imputer":
            if settings["type"] == "numeric":
                simple_numeric.append((col, settings["strategy"], settings["fill_value"]))
            else:
                simple_categorical.append((col, settings["strategy"], settings["fill_value"]))
        elif settings["method"] == "Fill with value":
            fill_values[col] = settings["fill_value"]
        elif settings["method"] == "KNN Imputer":
            knn_cols.append(col)
        elif settings["method"] == "Iterative Imputer":
            iterative_cols.append(col)

    for col, strategy, fill_value in simple_numeric:
        fill_value = None
        df = dc.simple_imputer(df, simple_numeric)

    for col, strategy, fill_value in simple_categorical:
        fill_value = None
        df = dc.simple_imputer(df, simple_categorical)

    for col, value in fill_values.items():
        df[col] = df[col].fillna(value)

    if knn_cols:
        df = dc.KNN_imputer(df, knn_cols, imputation_settings)

    if iterative_cols:
        dc.iterative__imputer(df, iterative_cols, imputation_settings)

    st.success(f"Successfully Handle Missing Vlaues")
    
    return df

def default_missing_values_selected(df):
    missing = dc.show_missing_value_persentage(df)
    missing = missing[missing >= 40]
    return missing.index

def default_low_variance_selected(df):
    vars = dc.calculate_variance(df)
    vars = vars[vars < 1]
    return vars.index

def duplicated_values(df):
    st.subheader('Duplicated Values')
    value_is_dublicate = dc.show_duplicate(df)
    if value_is_dublicate:
        st.warning(f'Count Of Duplicated Values Is {value_is_dublicate}')
        drop_dublicate = st.checkbox('Drop Duplicates')
        if drop_dublicate:
            df = dc.drop_duplicate(df)
            st.success(f"🧹 Removed {value_is_dublicate} duplicate rows successfully!")
    else:
        st.success("No duplicate values found. ✅")
    return df


def display_variance(df):
    st.subheader('Variance of each columns')
    v_df = dc.calculate_variance(df)

    chunks = [v_df.iloc[i::3] for i in range(3)]

    cols = st.columns(3)

    for col, chunk in zip(cols, chunks):
        col.table(chunk.to_frame(name="Variance Per Column"))

def drop_columns_gui(df, default, key, label=None):
    st.subheader(f'Drop Columns {label}')
    select_drop = st.multiselect('Select columns to drop', df.columns, default=default, key=key)
    if select_drop:
        df = dc.drop_columns(df, select_drop)
        st.success(f"Successfully dropped columns: {select_drop}")
    return df

def default_outliers(df):
    default_select = dc.detect_outlier_columns(df)
    return default_select

def handle_outliers(df):
    st.header('Handle Outliers')
    st.subheader('Box Plots Per Columns')
    cols = st.columns(5)
    for i, col in enumerate(df.select_dtypes(include=[np.number])):
        with cols[i%5]:
            fig = dc.box_plot(df, col)
            st.pyplot(fig)
    
    st.subheader("Select Columns For Handle Outliers")
    default_select = dc.detect_outlier_columns(df)
    select_cols_outliers = st.multiselect("Select columns to handle outliers", default_select, default=default_select, key=3)
    df_old_outliers = df[select_cols_outliers].copy()

    if select_cols_outliers:
        col1, col2 = st.columns(2)
        with col1:
            detect_method = st.selectbox('Select to detect outliers', ['IQR', 'ZScore'])
        with col2:
            handle_method = st.selectbox('Select to handle outliers', ['Remove outliers', 'Winsorization', 'Clip outliers'])

        if handle_method == 'Remove outliers':
            mask = pd.Series(True, index=df.index)
            for col in select_cols_outliers:
                if detect_method == 'IQR':
                    lower_bound, upper_bound = dc.upper_lower_IQR(df, col)
                if detect_method == 'ZScore':
                    lower_bound, upper_bound = dc.upper_lower_Zscore(df, col)
                
                mask = mask & (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df = df[mask]

        for col in select_cols_outliers:
            if detect_method == 'IQR':
                lower_bound, upper_bound = dc.upper_lower_IQR(df, col)
            if detect_method == 'ZScore':
                lower_bound, upper_bound = dc.upper_lower_Zscore(df, col)

            if handle_method == 'Winsorization':
                df = dc.winsorize_outliers(df, col)
            if handle_method == 'Clip outliers':
                df = dc.clip_outliers(df, col, lower_bound, upper_bound)
        
        st.success(f"Outlier handling completed for {select_cols_outliers} selected columns!")

        col1, col2 = st.columns(2)
        for col in select_cols_outliers:
            with col1:
                st.subheader('Before Outlier Handling')
                fig = dc.box_plot(df_old_outliers, col)
                st.pyplot(fig)
            with col2:
                st.subheader('After Outlier Handling')
                fig = dc.box_plot(df, col)
                st.pyplot(fig)
