import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox, yeojohnson

def label_encoder(df, column):
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[[column]])
    return df

def ordinal_encoder(df, column, based_on):
    encoder = OrdinalEncoder(categories=[based_on])
    df[column] = encoder.fit_transform(df[[column]])
    return df

def one_hot_encoder(df, column):
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded = encoder.fit_transform(df[[column]])

    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))

    df = pd.concat([df.drop(columns=[column]), encoded_df], axis=1)
    return df

def frequency_encoder(df, column):
    df[column] = df[column].map(df[column].value_counts())
    return df

def histogram(df, col):
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.histplot(x=df[col], ax=ax, color='teal', kde=True)
    ax.set_xlabel(None)
    ax.set_title(f"Histogram for {col}")
    return fig

def detect_skewness(df, skew=1):
    continuous_columns = []

    if skew != 1:
        for column in df.select_dtypes(include=['number']).columns:
            if df[column].nunique() > 20:
                continuous_columns.append(column)
    else:
        for column in df.select_dtypes(include=['number']).columns:
            if df[column].nunique() > 20 and abs(df[column].skew()) > 0.5:
                continuous_columns.append(column)
    
    return continuous_columns

def log_transform(df, column):
    df[column] = np.log1p(df[column])
    return df

def box_cox_transform(df, column):
    transformed_data, _ = boxcox(df[column], lmbda=None)
    df[column] = pd.Series(transformed_data, index=df.index)
    return df

def yeojohnson_transform(df, column):
    transformed_data, _ = yeojohnson(df[column])
    df[column] = pd.Series(transformed_data, index=df.index)
    return df

def min_max_scaler(df, column):
    scale = MinMaxScaler()
    df[column] = scale.fit_transform(df[[column]])
    return df

def standerd_scaler(df, column):
    scale = StandardScaler()
    df[column] = scale.fit_transform(df[[column]])
    return df

def detect_corr(df, lable, column):
    corr = df[lable].corr(df[column])
    return corr

def delete_low_corr(df, columns_to_drop):
    # correlations = df[columns_to_drop]
    df.drop(columns=columns_to_drop, inplace=True, axis=1)
    return df
