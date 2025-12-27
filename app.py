import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Risk / Loan Default Prediction",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Risk / Loan Default Prediction")
st.write(
    """
    This application predicts the **probability of loan default** using a
    **pre-trained machine learning model** built on real-world banking data.

    The system is designed to **support data-driven loan approval decisions**
    while balancing **risk management and business growth**, with a strong
    emphasis on **interpretability and transparency**.
    """
)

# --------------------------------------------------
# Load trained artifacts (NO DATASET REQUIRED)
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    """
    Load pre-trained model artifacts.
    Training is performed offline. Only model artifacts
    are used during deployment (industry best practice).
    """
    model = joblib.load("models/credit_risk_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# --------------------------------------------------
# Sidebar – User Input
# --------------------------------------------------
st.sidebar.header("🧾 Applicant Information")

def get_user_input():
    income = st.sidebar.number_input(
        "Total Annual Income", min_value=10000, value=300000, step=10000
    )
    credit = st.sidebar.number_input(
        "Loan Amount", min_value=10000, value=500000, step=10000
    )
    annuity = st.sidebar.number_input(
        "Annual Annuity", min_value=1000, value=30000, step=1000
    )
    age = st.sidebar.number_input(
        "Age (years)", min_value=18, max_value=80, value=30
    )
    employment_years = st.sidebar.number_input(
        "Employment Duration (years)", min_value=0.0, value=5.0, step=0.5
    )
    family_members = st.sidebar.number_input(
        "Number of Family Members", min_value=1, value=3
    )

    data = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "DAYS_BIRTH": -age * 365,
        "DAYS_EMPLOYED": -employment_years * 365,
        "CNT_FAM_MEMBERS": family_members
    }

    return pd.DataFrame([data])

input_df = get_user_input()

# --------------------------------------------------
# Feature Engineering (MUST MATCH TRAINING LOGIC)
# --------------------------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Financial ratios
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]

    # Age & employment features
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365
    df["EMPLOYMENT_YEARS"] = (-df["DAYS_EMPLOYED"]) / 365
    df["EMPLOYMENT_AGE_RATIO"] = df["EMPLOYMENT_YEARS"] / df["AGE_YEARS"]

    # Household-level feature
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]

    # Handle invalid values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    return df

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Credit Risk"):
    processed_df = feature_engineering(input_df)

    # Ensure feature consistency with training
    processed_df = pd.get_dummies(processed_df)
    processed_df = processed_df.reindex(columns=feature_columns, fill_value=0)

    # Scale features
    processed_scaled = scaler.transform(processed_df)

    # Predict probability
    probability = model.predict_proba(processed_scaled)[0][1]

    st.subheader("📊 Prediction Result")
    st.write(f"**Probability of Default:** `{probability:.2f}`")

    # --------------------------------------------------
    # Business Decision Logic
    # --------------------------------------------------
    if probability < 0.30:
        st.success("🟢 **Low Risk** — Loan Approved")
        st.write(
            "The applicant shows a low probability of default. "
            "The loan can be approved with standard checks."
        )

    elif probability < 0.60:
        st.warning("🟡 **Medium Risk** — Manual Review Required")
        st.write(
            "The applicant has a moderate risk profile. "
            "Further verification or human review is recommended."
        )

    else:
        st.error("🔴 **High Risk** — Loan Rejected")
        st.write(
            "The applicant shows a high probability of default. "
            "Approving this loan may result in financial loss."
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    """
    **Author:** Hemant Kumar  
    *B.Tech Student | Aspiring Data Analyst*  

    This tool is intended to **assist decision-making** and should not
    replace human judgment.
    """
)
