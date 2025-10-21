import pandas as pd

def load_data(file_path):
    file_name = file_path.name

    if file_name.endswith('.csv'):
        df = pd.read_csv(file_path, index_col=False)
    elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        df = pd.read_excel(file_path, index_col=False)
    
    return df

def append_datasets(df):
    df = pd.concat(list(df), ignore_index=True)
    return df

def value_count(df):
    return df.nunique().sort_values()

def change_datatype(df, convert_dict):
    df = df.astype(convert_dict)
    return df

def inner_join(df1, df2, left_join, right_join):
    df = pd.merge(df1, df2, left_on=left_join, right_on=right_join, how='inner')
    return df

def cross_join(df1, df2, left_join, right_join):
    df = pd.merge(df1, df2,left_on=left_join, right_on=right_join, how='outer')
    return df

def left_join(df1, df2, left_join, right_join):
    df = pd.merge(df1, df2, left_on=left_join, right_on=right_join, how='left')
    return df

def right_join(df1, df2, left_join, right_join):
    df = pd.merge(df1, df2, left_on=left_join, right_on=right_join, how='right')
    return df

def index_join(df1, df2):
    df = pd.merge(df1, df2, left_index=True, right_index=True)
    return df
