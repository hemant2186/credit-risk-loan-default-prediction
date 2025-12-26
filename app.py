import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
model = joblib.load("models/credit_risk_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

st.title("💳 Credit Risk / Loan Default Prediction")
st.write("Predict loan default risk using a data-driven ML model")

st.sidebar.header("Applicant Information")

def user_input():
    amt_income = st.sidebar.number_input("Total Income", min_value=10000)
    amt_credit = st.sidebar.number_input("Credit Amount", min_value=10000)
    amt_annuity = st.sidebar.number_input("Annuity Amount", min_value=1000)
    days_birth = st.sidebar.number_input("Age (Years)", min_value=18)
    emp_years = st.sidebar.number_input("Employment Duration (Years)", min_value=0)
    fam_members = st.sidebar.number_input("Family Members", min_value=1)

    data = {
        "AMT_INCOME_TOTAL": amt_income,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_annuity,
        "DAYS_BIRTH": -days_birth * 365,
        "DAYS_EMPLOYED": -emp_years * 365,
        "CNT_FAM_MEMBERS": fam_members
    }

    return pd.DataFrame([data])

input_df = user_input()

if st.button("Predict Risk"):
    df = input_df.copy()

    # Feature engineering (same as training)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365
    df["EMPLOYMENT_YEARS"] = (-df["DAYS_EMPLOYED"]) / 365
    df["EMPLOYMENT_AGE_RATIO"] = df["EMPLOYMENT_YEARS"] / df["AGE_YEARS"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]

    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)

    df_scaled = scaler.transform(df)
    prob = model.predict_proba(df_scaled)[0][1]

    st.subheader("🔍 Prediction Result")
    st.write(f"**Default Probability:** {prob:.2f}")

    if prob < 0.3:
        st.success("Low Risk — Loan Approved")
    elif prob < 0.6:
        st.warning("Medium Risk — Manual Review Needed")
    else:
        st.error("High Risk — Loan Rejected")
