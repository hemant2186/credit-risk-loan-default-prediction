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
    HOME_CREDIT_THRESHOLD_PNG,
    HOME_CREDIT_THRESHOLD_CSV,
    HOME_CREDIT_THRESHOLD_SUMMARY_FILE,
)


st.set_page_config(
    page_title="Home Credit Risk Demo",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_demo_data() -> pd.DataFrame:
    return pd.read_csv(HOME_CREDIT_DEMO_DATA_FILE, compression="gzip")


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return joblib.load(model_path)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def resolve_model_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.exists():
        return candidate

    relative_candidate = BASE_DIR / path_text
    if relative_candidate.exists():
        return relative_candidate

    return candidate


def main() -> None:
    st.title("Home Credit Default Risk Portfolio Demo")
    st.caption("A recruiter-friendly demo for exploring engineered credit-risk scores on real Kaggle competition data.")

    if not HOME_CREDIT_MODEL_REPORT_FILE.exists() or not HOME_CREDIT_DEMO_DATA_FILE.exists():
        st.error("Run `python -m src.train_home_credit` first so the model and demo dataset are available.")
        return

    threshold_summary = load_json(HOME_CREDIT_THRESHOLD_SUMMARY_FILE)
    model_report = load_json(HOME_CREDIT_MODEL_REPORT_FILE)
    threshold_table = pd.read_csv(HOME_CREDIT_THRESHOLD_CSV) if HOME_CREDIT_THRESHOLD_CSV.exists() else pd.DataFrame()
    best_model_path = model_report.get("best_model_by_roc_auc", {}).get("path")
    if not best_model_path:
        st.error("Best model path not found. Re-run `python -m src.train_home_credit`.")
        return

    demo_df = load_demo_data()
    model = load_model(str(resolve_model_path(best_model_path)))

    recommended_threshold = threshold_summary.get("recommended_threshold_business", 0.35)
    threshold = st.sidebar.slider("Decision threshold", 0.10, 0.90, float(recommended_threshold), 0.05)

    st.sidebar.markdown("### Why threshold matters")
    st.sidebar.write(
        "Lower values flag more applicants as risky. Higher values approve more applicants but can miss more defaults."
    )

    best_model_name = model_report.get("best_model_by_roc_auc", {}).get("name", "unknown")
    best_model_auc = model_report.get("models", {}).get(best_model_name, {}).get("roc_auc", 0)

    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    metrics_col1.metric("Applicants In Demo", f"{len(demo_df):,}")
    metrics_col2.metric("Engineered Features", str(model_report.get("dataset_shape", ["-", "-"])[1]))
    metrics_col3.metric("Default Rate", f"{model_report.get('default_rate', 0):.1%}")
    metrics_col4.metric("Best Model", f"{best_model_name} ({best_model_auc:.3f})")

    st.markdown("### Best model summary")
    st.write(
        f"Current best model: **{best_model_name}**"
    )

    st.markdown("### Candidate-level scoring")
    selectable_ids = demo_df["SK_ID_CURR"].astype(int).tolist()
    selected_id = st.selectbox("Select an applicant ID from the demo set", selectable_ids, index=0)

    applicant = demo_df.loc[demo_df["SK_ID_CURR"] == selected_id].copy()
    target = int(applicant["TARGET"].iloc[0])
    score_df = applicant.drop(columns=["TARGET"])
    probability = float(model.predict_proba(score_df)[0][1])
    predicted_default = int(probability >= threshold)

    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Predicted Default Probability", f"{probability:.1%}")
    result_col2.metric("Threshold Decision", "High Risk" if predicted_default else "Lower Risk")
    result_col3.metric("Actual Target", "Defaulted" if target == 1 else "No Default")

    st.markdown("### Applicant feature snapshot")
    preview_cols = [
        "SK_ID_CURR",
        "CODE_GENDER",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "EXT_SOURCE_2",
        "PREV_APPLICATION_COUNT",
        "BUREAU_RECORD_COUNT",
        "INST_DAYS_PAST_DUE_MEAN",
        "POS_SK_DPD_MEAN",
        "CC_UTILIZATION_MEAN",
    ]
    existing_cols = [col for col in preview_cols if col in applicant.columns]
    st.dataframe(applicant[existing_cols].T.rename(columns={applicant.index[0]: "value"}), use_container_width=True)

    st.markdown("### Threshold analysis")
    if threshold_table.empty:
        st.info("Run `python -m src.threshold_tuning` to generate threshold tradeoff reports.")
    else:
        st.dataframe(threshold_table, use_container_width=True)
        st.line_chart(threshold_table.set_index("threshold")[["precision", "recall", "f1"]])
        if HOME_CREDIT_THRESHOLD_PNG.exists():
            st.image(str(HOME_CREDIT_THRESHOLD_PNG), caption="Threshold tradeoff chart")

    st.markdown("### Model explainability")
    if HOME_CREDIT_FEATURE_IMPORTANCE_CSV.exists():
        importance_df = pd.read_csv(HOME_CREDIT_FEATURE_IMPORTANCE_CSV)
        st.dataframe(importance_df, use_container_width=True)
        if HOME_CREDIT_FEATURE_IMPORTANCE_PNG.exists():
            st.image(
                str(HOME_CREDIT_FEATURE_IMPORTANCE_PNG),
                caption="Top features used by the best model",
                use_container_width=True,
            )
    else:
        st.info("Run `python -m src.explain_model` to generate feature importance reports.")

    st.markdown("### Model comparison")
    if HOME_CREDIT_MODEL_COMPARISON_PNG.exists():
        st.image(
            str(HOME_CREDIT_MODEL_COMPARISON_PNG),
            caption="Portfolio-ready model comparison chart",
            use_container_width=True,
        )

    st.markdown("### Resume-ready project strengths")
    for point in model_report.get("project_strengths", []):
        st.write(f"- {point}")


if __name__ == "__main__":
    main()
