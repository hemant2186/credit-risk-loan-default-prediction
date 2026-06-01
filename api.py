from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import (
    BASE_DIR,
    HOME_CREDIT_DEMO_DATA_FILE,
    HOME_CREDIT_MODEL_REPORT_FILE,
    HOME_CREDIT_XGBOOST_MODEL_FILE,
)


app = FastAPI(
    title="CreditRisk AI API",
    version="1.0.0",
    description="JSON API for borrower default-risk scoring.",
)


class ApplicantBatch(BaseModel):
    applicants: list[dict[str, Any]] = Field(..., min_length=1)
    threshold: float = Field(0.55, ge=0.1, le=0.9)


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


def _load_model():
    report = _load_json(HOME_CREDIT_MODEL_REPORT_FILE)
    model_path = _resolve_model_path(report.get("best_model_by_roc_auc", {}).get("path"))
    return joblib.load(model_path)


MODEL = _load_model()


def _required_features() -> list[str]:
    preprocessor = MODEL.named_steps["preprocessor"]
    required_features: list[str] = []
    for _, _, columns in preprocessor.transformers_:
        required_features.extend(list(columns))
    return required_features


REQUIRED_FEATURES = _required_features()


def _score_frame(input_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    scoring_df = input_df.copy()
    for column in REQUIRED_FEATURES:
        if column not in scoring_df.columns:
            scoring_df[column] = pd.NA

    probabilities = MODEL.predict_proba(scoring_df[REQUIRED_FEATURES])[:, 1]
    output_df = input_df.copy()
    output_df["default_probability"] = probabilities
    output_df["risk_band"] = "Review"
    output_df.loc[output_df["default_probability"] < min(0.25, threshold / 2), "risk_band"] = "Low"
    output_df.loc[output_df["default_probability"] >= threshold, "risk_band"] = "High"
    output_df["decision"] = output_df["default_probability"].ge(threshold).map(
        {True: "Manual review / decline", False: "Approve / monitor"}
    )
    return output_df


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": "home_credit_xgboost",
        "required_features": len(REQUIRED_FEATURES),
    }


@app.get("/schema")
def schema() -> dict[str, object]:
    demo_df = pd.read_csv(HOME_CREDIT_DEMO_DATA_FILE, compression="gzip", nrows=1)
    sample_applicant = demo_df.drop(columns=["TARGET"], errors="ignore").iloc[0].to_dict()
    return {
        "required_features": REQUIRED_FEATURES,
        "sample_applicant": sample_applicant,
    }


@app.post("/score")
def score(batch: ApplicantBatch) -> dict[str, object]:
    input_df = pd.DataFrame(batch.applicants)
    scored_df = _score_frame(input_df, batch.threshold)
    return {
        "threshold": batch.threshold,
        "count": int(len(scored_df)),
        "results": scored_df[
            [
                column
                for column in ["SK_ID_CURR", "default_probability", "risk_band", "decision"]
                if column in scored_df.columns
            ]
        ].to_dict(orient="records"),
    }
