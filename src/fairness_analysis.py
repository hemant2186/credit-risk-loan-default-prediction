from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .config import (
    BASE_DIR,
    HOME_CREDIT_DEMO_DATA_FILE,
    HOME_CREDIT_FAIRNESS_CSV,
    HOME_CREDIT_FAIRNESS_JSON,
    HOME_CREDIT_MODEL_REPORT_FILE,
    HOME_CREDIT_THRESHOLD_SUMMARY_FILE,
    HOME_CREDIT_XGBOOST_MODEL_FILE,
    REPORTS_DIR,
)


GROUP_COLUMNS = [
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS",
]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _resolve_model_path(path_text: str | None) -> Path:
    if not path_text:
        return HOME_CREDIT_XGBOOST_MODEL_FILE

    normalized_path_text = path_text.replace("\\", "/")
    candidate = Path(normalized_path_text)
    if candidate.exists():
        return candidate

    relative_candidate = BASE_DIR / normalized_path_text
    if relative_candidate.exists():
        return relative_candidate

    return HOME_CREDIT_XGBOOST_MODEL_FILE


def _required_features(model) -> list[str]:
    preprocessor = model.named_steps["preprocessor"]
    required_features: list[str] = []
    for _, _, columns in preprocessor.transformers_:
        required_features.extend(list(columns))
    return required_features


def _score_demo_applicants(model, threshold: float) -> pd.DataFrame:
    demo_df = pd.read_csv(HOME_CREDIT_DEMO_DATA_FILE, compression="gzip")
    required_features = _required_features(model)
    scoring_df = demo_df.drop(columns=["TARGET"], errors="ignore").copy()

    for column in required_features:
        if column not in scoring_df.columns:
            scoring_df[column] = pd.NA

    probabilities = model.predict_proba(scoring_df[required_features])[:, 1]
    scored_df = demo_df.copy()
    scored_df["default_probability"] = probabilities
    scored_df["is_high_risk"] = scored_df["default_probability"] >= threshold
    return scored_df


def generate_fairness_summary() -> pd.DataFrame:
    model_report = _load_json(HOME_CREDIT_MODEL_REPORT_FILE)
    threshold_summary = _load_json(HOME_CREDIT_THRESHOLD_SUMMARY_FILE)
    threshold = float(threshold_summary.get("recommended_threshold_business", 0.55))
    model_path = _resolve_model_path(model_report.get("best_model_by_roc_auc", {}).get("path"))
    model = joblib.load(model_path)

    scored_df = _score_demo_applicants(model, threshold)

    rows: list[dict[str, object]] = []
    for column in GROUP_COLUMNS:
        if column not in scored_df.columns:
            continue

        grouped = scored_df.groupby(column, dropna=False)
        for group_value, group_df in grouped:
            if len(group_df) < 25:
                continue
            rows.append(
                {
                    "group_feature": column,
                    "group_value": str(group_value),
                    "applicants": int(len(group_df)),
                    "avg_default_probability": round(float(group_df["default_probability"].mean()), 4),
                    "high_risk_rate": round(float(group_df["is_high_risk"].mean()), 4),
                    "observed_default_rate": round(float(group_df["TARGET"].mean()), 4)
                    if "TARGET" in group_df.columns
                    else None,
                }
            )

    fairness_df = pd.DataFrame(rows).sort_values(["group_feature", "high_risk_rate"], ascending=[True, False])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fairness_df.to_csv(HOME_CREDIT_FAIRNESS_CSV, index=False)

    summary = {
        "threshold": threshold,
        "model_path": model_path.as_posix(),
        "rows": int(len(fairness_df)),
        "note": (
            "This is a portfolio fairness snapshot, not a regulatory fairness audit. "
            "A production lending system would require legally reviewed protected-class definitions, "
            "bias tests, documentation, and human oversight."
        ),
    }
    HOME_CREDIT_FAIRNESS_JSON.write_text(json.dumps(summary, indent=2))

    print("Fairness summary complete")
    print(f"Saved fairness table to: {HOME_CREDIT_FAIRNESS_CSV}")
    print(f"Saved fairness metadata to: {HOME_CREDIT_FAIRNESS_JSON}")
    return fairness_df


if __name__ == "__main__":
    generate_fairness_summary()
