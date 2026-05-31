import numpy as np

def create_features(df):
    df = df.copy()

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]

    df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365
    df["EMPLOYMENT_YEARS"] = (-df["DAYS_EMPLOYED"]) / 365
    df["EMPLOYMENT_AGE_RATIO"] = df["EMPLOYMENT_YEARS"] / df["AGE_YEARS"]

    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    return df
