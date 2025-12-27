import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("💳 Credit Risk / Loan Default Prediction")

st.write(
    """
    This application predicts the **probability of loan default**
    using a machine learning model trained on real-world banking data.

    The system is designed to **support data-driven loan approval decisions**
    while balancing **risk management and business growth**.
    """
)

# --------------------------------------------------
# Load Trained Artifacts (Cached)
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/credit_risk_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# --------------------------------------------------
# Sidebar – User Input (REALISTIC CONSTRAINTS)
# --------------------------------------------------
st.sidebar.header("🧾 Applicant Information")

def get_user_input():
    income = st.sidebar.number_input(
        "Total Annual Income (₹)",
        min_value=100000,
        max_value=5000000,
        value=500000,
        step=25000
    )

    credit = st.sidebar.number_input(
        "Loan Amount Requested (₹)",
        min_value=50000,
        max_value=10000000,
        value=500000,
        step=50000
    )

    annuity = st.sidebar.number_input(
        "Annual EMI / Annuity (₹)",
        min_value=12000,
        max_value=1000000,
        value=60000,
        step=5000
    )

    age = st.sidebar.slider(
        "Age (years)",
        min_value=21,
        max_value=65,
        value=30
    )

    employment_years = st.sidebar.slider(
        "Employment Duration (years)",
        min_value=0,
        max_value=40,
        value=5
    )

    family_members = st.sidebar.slider(
        "Number of Family Members",
        min_value=1,
        max_value=10,
        value=3
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
# Business Validation (BLOCK UNREALISTIC CASES)
# --------------------------------------------------
def validate_inputs(df):
    age = -df["DAYS_BIRTH"].iloc[0] / 365
    employment = -df["DAYS_EMPLOYED"].iloc[0] / 365
    income = df["AMT_INCOME_TOTAL"].iloc[0]
    credit = df["AMT_CREDIT"].iloc[0]
    annuity = df["AMT_ANNUITY"].iloc[0]

    issues = []

    if employment > (age - 18):
        issues.append("Employment duration exceeds realistic working age.")

    if credit > income * 10:
        issues.append("Loan amount is unusually high compared to income.")

    if annuity > income * 0.6:
        issues.append("Annual EMI exceeds 60% of income.")

    return issues

# --------------------------------------------------
# Feature Engineering (Must Match Training)
# --------------------------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
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

# --------------------------------------------------
# Explanation Logic (STABLE & ALWAYS WORKS)
# --------------------------------------------------
def explain_prediction(model, input_df, feature_columns):
    if hasattr(model, "coef_"):
        contributions = model.coef_[0] * input_df.values[0]
    else:
        contributions = model.feature_importances_ * input_df.values[0]

    explanation = pd.Series(contributions, index=feature_columns)
    explanation = explanation.sort_values(key=abs, ascending=False)

    return explanation.head(5)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Credit Risk"):

    issues = validate_inputs(input_df)
    if issues:
        for issue in issues:
            st.warning(issue)
        st.stop()

    processed_df = feature_engineering(input_df)
    processed_df = pd.get_dummies(processed_df)
    processed_df = processed_df.reindex(columns=feature_columns, fill_value=0)

    processed_scaled = scaler.transform(processed_df)
    probability = model.predict_proba(processed_scaled)[0][1]

    st.subheader("📊 Prediction Result")
    st.write(f"**Probability of Default:** `{probability:.2f}`")

    if probability < 0.01:
        st.info(
            "ℹ️ The model predicts an extremely low default risk due to strong "
            "financial indicators. Values are rounded for display."
        )

    # --------------------------------------------------
    # Explanation
    # --------------------------------------------------
    st.subheader("🧠 Why this prediction?")

    top_factors = explain_prediction(model, processed_df, feature_columns)

    for feature, value in top_factors.items():
        direction = "increases" if value > 0 else "decreases"
        st.write(f"- **{feature}** {direction} default risk")

    # --------------------------------------------------
    # Business Decision Recommendation
    # --------------------------------------------------
    st.subheader("🏦 Credit Decision Recommendation")

    if probability < 0.30:
        st.success("🟢 Low Risk — Eligible for Approval")
        st.write(
            "The applicant shows a low probability of default. "
            "Final approval is subject to standard verification checks."
        )

    elif probability < 0.60:
        st.warning("🟡 Medium Risk — Manual Review Recommended")
        st.write(
            "The applicant presents moderate credit risk. "
            "Additional verification is advised."
        )

    else:
        st.error("🔴 High Risk — Approval Not Recommended")
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

    This tool assists decision-making and should not replace human judgment.
    """
)
