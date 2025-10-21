import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats.mstats import winsorize

def show_missing_value_persentage(df):
    missing = df.isnull().sum() / df.shape[0] * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    return None if missing.empty else missing

def missing_matrix(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    msno.matrix(df[df.columns[df.isnull().any()]], ax=ax, color=(0, 0.5, 0.5))
    ax.set_title("Missing Values Matrix")
    return fig

def missing_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    msno.heatmap(df[df.columns[df.isnull().any()]], ax=ax, cmap='viridis')
    ax.set_title("Missing Values Heatmap")
    return fig

def missing_bar(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    msno.bar(df[df.columns[df.isnull().any()]], ax=ax, color='teal')
    ax.set_title("Missing Values Bar Chart")
    return fig


def simple_imputer(df, simple_numeric):
    for col, strategy, fill_value in simple_numeric:
        if strategy == "constant":
                imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        else:
            imputer = SimpleImputer(strategy=strategy)
        df[[col]] = imputer.fit_transform(df[[col]])
    return df


def KNN_imputer(df, col, imputation_settings):
    n_neighbors = imputation_settings[col[0]]["n_neighbors"]
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df[col] = knn_imputer.fit_transform(df[col])
    return df

def iterative__imputer(df, iterative_cols, imputation_settings):
    max_iter = imputation_settings[iterative_cols[0]]["max_iter"]
    imputer = IterativeImputer(max_iter=max_iter, random_state=0)
    df[iterative_cols] = imputer.fit_transform(df[iterative_cols])
    return df

def fill_value(df, col, value):
    df[col] = df[col].fillna(value)
    return df


def show_duplicate(df):
    duplicate = df.duplicated().sum()
    return duplicate

def drop_duplicate(df):
    df = df.drop_duplicates()
    return df

def calculate_variance(df):
    return df.var(numeric_only=True).sort_values()

def drop_columns(df, select_drop):
    df = df.drop(select_drop, axis=1)
    return df

def box_plot(df, col):
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.boxplot(x=df[col], ax=ax, color='teal')
    ax.set_xlabel(None)
    ax.set_title(f"Boxplot for {col}")
    return fig

def detect_outlier_columns(df):
    outlier_cols = []
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if ((df[col] < lower_bound) | (df[col] > upper_bound)).any():
            outlier_cols.append(col)
    return outlier_cols


def upper_lower_IQR(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound

def upper_lower_Zscore(df, col):
    lower_bound = df[col].mean() - 3 * df[col].std()
    upper_bound = df[col].mean() + 3 * df[col].std()
    return lower_bound, upper_bound

def remove_outliers(df, col, lower_bound, upper_bound):
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def winsorize_outliers(df, col):
    df[col] = winsorize(df[col], limits=[0.05, 0.05])
    return df

def clip_outliers(df, col, lower_bound, upper_bound):
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df
