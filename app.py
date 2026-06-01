from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import (
    BASE_DIR,
    HOME_CREDIT_DEMO_DATA_FILE,
    HOME_CREDIT_FEATURE_IMPORTANCE_CSV,
    HOME_CREDIT_FEATURE_IMPORTANCE_PNG,
    HOME_CREDIT_MODEL_COMPARISON_PNG,
    HOME_CREDIT_MODEL_REPORT_FILE,
    HOME_CREDIT_THRESHOLD_CSV,
    HOME_CREDIT_THRESHOLD_PNG,
    HOME_CREDIT_THRESHOLD_SUMMARY_FILE,
)


st.set_page_config(
    page_title="CreditRisk AI",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


PREVIEW_COLUMNS = [
    "SK_ID_CURR",
    "CODE_GENDER",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "PREV_APPLICATION_COUNT",
    "BUREAU_RECORD_COUNT",
    "INST_DAYS_PAST_DUE_MEAN",
    "POS_SK_DPD_MEAN",
    "CC_UTILIZATION_MEAN",
]


@st.cache_data(show_spinner=False)
def load_demo_data() -> pd.DataFrame:
    return pd.read_csv(HOME_CREDIT_DEMO_DATA_FILE, compression="gzip")


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return joblib.load(model_path)


def resolve_model_path(path_text: str | None) -> Path:
    if not path_text:
        return BASE_DIR / "models" / "home_credit_xgboost.joblib"

    normalized_path_text = path_text.replace("\\", "/")
    candidate = Path(normalized_path_text)
    if candidate.exists():
        return candidate

    relative_candidate = BASE_DIR / normalized_path_text
    if relative_candidate.exists():
        return relative_candidate

    return candidate


def get_model_or_stop(best_model_path: str | None):
    resolved_path = resolve_model_path(best_model_path)
    if not resolved_path.exists():
        st.error(
            "The trained model artifact is missing from this deployment. "
            f"Expected model file: `{resolved_path}`. "
            "Make sure `models/home_credit_xgboost.joblib` is committed and available in GitHub."
        )
        st.stop()
    return load_model(str(resolved_path))


def get_required_features(model) -> list[str]:
    preprocessor = model.named_steps["preprocessor"]
    required_features: list[str] = []
    for _, _, columns in preprocessor.transformers_:
        required_features.extend(list(columns))
    return required_features


def prepare_scoring_data(input_df: pd.DataFrame, required_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    scoring_df = input_df.copy()
    if "TARGET" in scoring_df.columns:
        scoring_df = scoring_df.drop(columns=["TARGET"])

    missing_features = [column for column in required_features if column not in scoring_df.columns]
    for column in missing_features:
        scoring_df[column] = pd.NA

    return scoring_df[required_features], missing_features


def score_applicants(input_df: pd.DataFrame, model, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    required_features = get_required_features(model)
    scoring_df, missing_features = prepare_scoring_data(input_df, required_features)
    probabilities = model.predict_proba(scoring_df)[:, 1]

    output_df = input_df.copy()
    output_df["default_probability"] = probabilities
    review_floor = min(0.25, threshold / 2)
    output_df["risk_band"] = "Review"
    output_df.loc[output_df["default_probability"] < review_floor, "risk_band"] = "Low"
    output_df.loc[output_df["default_probability"] >= threshold, "risk_band"] = "High"
    output_df["decision"] = output_df["default_probability"].ge(threshold).map(
        {True: "Manual review / decline", False: "Approve / monitor"}
    )
    return output_df.sort_values("default_probability", ascending=False), missing_features


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_summary(scored_df: pd.DataFrame, threshold: float) -> dict[str, str]:
    high_risk_count = int((scored_df["default_probability"] >= threshold).sum())
    return {
        "Applicants": f"{len(scored_df):,}",
        "High Risk": f"{high_risk_count:,}",
        "High Risk Rate": f"{high_risk_count / max(len(scored_df), 1):.1%}",
        "Avg. Risk Score": f"{scored_df['default_probability'].mean():.1%}",
    }


def render_metric_row(summary: dict[str, str]) -> None:
    columns = st.columns(len(summary))
    for column, (label, value) in zip(columns, summary.items()):
        column.metric(label, value)


def render_upload_help(template_df: pd.DataFrame) -> None:
    st.markdown("### Data template")
    st.write(
        "Upload a CSV with borrower-level application and engineered credit-history fields. "
        "Use the template when integrating this with an internal loan-origination export."
    )
    st.download_button(
        "Download scoring template",
        data=dataframe_to_csv_bytes(template_df.head(100).drop(columns=["TARGET"], errors="ignore")),
        file_name="creditrisk_scoring_template.csv",
        mime="text/csv",
    )


def render_dashboard(model_report: dict, threshold_summary: dict, demo_scored_df: pd.DataFrame, threshold: float) -> None:
    st.subheader("Portfolio dashboard")
    render_metric_row(build_summary(demo_scored_df, threshold))

    best_model_name = model_report.get("best_model_by_roc_auc", {}).get("name", "unknown")
    best_model_auc = model_report.get("models", {}).get(best_model_name, {}).get("roc_auc", 0)

    model_col, threshold_col = st.columns(2)
    with model_col:
        st.markdown("### Model health")
        st.metric("Best model", best_model_name)
        st.metric("ROC-AUC", f"{best_model_auc:.3f}")
        st.metric("Training default rate", f"{model_report.get('default_rate', 0):.1%}")

    with threshold_col:
        st.markdown("### Operating policy")
        st.metric("Active threshold", f"{threshold:.2f}")
        st.metric("Best F1 threshold", threshold_summary.get("recommended_threshold_f1", "-"))
        st.metric("Business threshold", threshold_summary.get("recommended_threshold_business", "-"))

    st.markdown("### Highest-risk applicants")
    display_cols = [
        "SK_ID_CURR",
        "default_probability",
        "risk_band",
        "decision",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ]
    existing_cols = [column for column in display_cols if column in demo_scored_df.columns]
    st.dataframe(demo_scored_df[existing_cols].head(25), use_container_width=True)


def render_batch_scoring(model, demo_df: pd.DataFrame, threshold: float) -> None:
    st.subheader("Batch applicant scoring")
    render_upload_help(demo_df)

    uploaded_file = st.file_uploader("Upload applicant CSV", type=["csv"])
    if uploaded_file is None:
        st.info("No upload yet. The demo portfolio below shows the same scoring workflow on sample applicants.")
        input_df = demo_df.drop(columns=["TARGET"], errors="ignore").head(250)
    else:
        input_df = pd.read_csv(uploaded_file)

    scored_df, missing_features = score_applicants(input_df, model, threshold)

    if missing_features:
        st.warning(
            f"{len(missing_features)} model features were missing from the upload and filled as blank values. "
            "For best production accuracy, send the full engineered schema."
        )

    render_metric_row(build_summary(scored_df, threshold))
    st.dataframe(scored_df.head(100), use_container_width=True)
    st.download_button(
        "Download scored applicants",
        data=dataframe_to_csv_bytes(scored_df),
        file_name="creditrisk_scored_applicants.csv",
        mime="text/csv",
    )


def render_single_applicant(model, demo_df: pd.DataFrame, threshold: float) -> None:
    st.subheader("Single applicant review")
    selectable_ids = demo_df["SK_ID_CURR"].astype(int).tolist()
    selected_id = st.selectbox("Applicant ID", selectable_ids, index=0)

    applicant = demo_df.loc[demo_df["SK_ID_CURR"] == selected_id].copy()
    scored_df, _ = score_applicants(applicant.drop(columns=["TARGET"], errors="ignore"), model, threshold)
    scored_row = scored_df.iloc[0]
    target = int(applicant["TARGET"].iloc[0]) if "TARGET" in applicant.columns else None

    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Default probability", f"{scored_row['default_probability']:.1%}")
    result_col2.metric("Decision", scored_row["decision"])
    result_col3.metric("Known outcome", "Defaulted" if target == 1 else "No default")

    st.markdown("### Applicant snapshot")
    existing_cols = [column for column in PREVIEW_COLUMNS if column in applicant.columns]
    st.dataframe(applicant[existing_cols].T.rename(columns={applicant.index[0]: "value"}), use_container_width=True)


def render_analytics(threshold_table: pd.DataFrame) -> None:
    st.subheader("Model analytics")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        if HOME_CREDIT_MODEL_COMPARISON_PNG.exists():
            st.image(str(HOME_CREDIT_MODEL_COMPARISON_PNG), caption="Model comparison", use_container_width=True)
    with chart_col2:
        if HOME_CREDIT_THRESHOLD_PNG.exists():
            st.image(str(HOME_CREDIT_THRESHOLD_PNG), caption="Threshold tradeoff", use_container_width=True)

    st.markdown("### Feature importance")
    if HOME_CREDIT_FEATURE_IMPORTANCE_CSV.exists():
        importance_df = pd.read_csv(HOME_CREDIT_FEATURE_IMPORTANCE_CSV)
        st.dataframe(importance_df, use_container_width=True)
        if HOME_CREDIT_FEATURE_IMPORTANCE_PNG.exists():
            st.image(str(HOME_CREDIT_FEATURE_IMPORTANCE_PNG), caption="Top model features", use_container_width=True)
    else:
        st.info("Run `python -m src.explain_model` to generate feature importance reports.")

    st.markdown("### Threshold table")
    if threshold_table.empty:
        st.info("Run `python -m src.threshold_tuning` to generate threshold tradeoff reports.")
    else:
        st.dataframe(threshold_table, use_container_width=True)


def render_product_notes() -> None:
    st.subheader("How teams can use this")
    st.write(
        "CreditRisk AI is packaged as a lightweight decision-support product for lenders, fintech teams, "
        "and analysts who need repeatable borrower risk scoring."
    )

    use_cases = pd.DataFrame(
        [
            {
                "Workflow": "Pre-screening",
                "User": "Loan officer",
                "Outcome": "Prioritize applications that need deeper review.",
            },
            {
                "Workflow": "Portfolio monitoring",
                "User": "Risk analyst",
                "Outcome": "Identify segments with elevated default probability.",
            },
            {
                "Workflow": "Policy testing",
                "User": "Credit manager",
                "Outcome": "Compare approval impact at different score thresholds.",
            },
        ]
    )
    st.dataframe(use_cases, use_container_width=True, hide_index=True)

    st.markdown("### Production boundaries")
    st.write(
        "This app is a decision-support prototype. A real lending deployment should add authentication, "
        "audit logs, fairness testing, human review, monitoring, and compliance approval before making "
        "binding credit decisions."
    )


def main() -> None:
    st.title("CreditRisk AI")
    st.caption("Batch credit-risk scoring, threshold policy testing, and model explainability for lending teams.")

    if not HOME_CREDIT_MODEL_REPORT_FILE.exists() or not HOME_CREDIT_DEMO_DATA_FILE.exists():
        st.error("Run `python -m src.train_home_credit` first so the model and demo dataset are available.")
        return

    threshold_summary = load_json(HOME_CREDIT_THRESHOLD_SUMMARY_FILE)
    model_report = load_json(HOME_CREDIT_MODEL_REPORT_FILE)
    threshold_table = pd.read_csv(HOME_CREDIT_THRESHOLD_CSV) if HOME_CREDIT_THRESHOLD_CSV.exists() else pd.DataFrame()
    best_model_path = model_report.get("best_model_by_roc_auc", {}).get("path")

    model = get_model_or_stop(best_model_path)
    demo_df = load_demo_data()

    recommended_threshold = threshold_summary.get("recommended_threshold_business", 0.55)
    threshold = st.sidebar.slider("Decision threshold", 0.10, 0.90, float(recommended_threshold), 0.05)
    st.sidebar.caption("Lower thresholds catch more potential defaults. Higher thresholds reduce false alarms.")

    demo_scored_df, _ = score_applicants(demo_df.drop(columns=["TARGET"], errors="ignore"), model, threshold)

    dashboard_tab, batch_tab, applicant_tab, analytics_tab, product_tab = st.tabs(
        ["Dashboard", "Batch Scoring", "Applicant Review", "Analytics", "Product"]
    )

    with dashboard_tab:
        render_dashboard(model_report, threshold_summary, demo_scored_df, threshold)

    with batch_tab:
        render_batch_scoring(model, demo_df, threshold)

    with applicant_tab:
        render_single_applicant(model, demo_df, threshold)

    with analytics_tab:
        render_analytics(threshold_table)

    with product_tab:
        render_product_notes()


if __name__ == "__main__":
    main()
