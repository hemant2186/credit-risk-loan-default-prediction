import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def handle_missing_values(df):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df

def drop_high_missing_columns(df, threshold=0.4):
    missing_ratio = df.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > threshold].index
    return df.drop(columns=drop_cols)
