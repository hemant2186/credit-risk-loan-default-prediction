from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from .config import (
    BASE_DIR,
    HOME_CREDIT_FEATURE_IMPORTANCE_CSV,
    HOME_CREDIT_FEATURE_IMPORTANCE_PNG,
    HOME_CREDIT_MODEL_REPORT_FILE,
    HOME_CREDIT_XGBOOST_MODEL_FILE,
    REPORTS_DIR,
)


def _resolve_model_path(path_text: str | None) -> Path:
    if not path_text:
        return HOME_CREDIT_XGBOOST_MODEL_FILE

    candidate = Path(path_text)
    if candidate.exists():
        return candidate

    relative_candidate = BASE_DIR / path_text
    if relative_candidate.exists():
        return relative_candidate

    return HOME_CREDIT_XGBOOST_MODEL_FILE


def _clean_feature_name(feature_name: str) -> str:
    for prefix in ("num__", "cat__"):
        if feature_name.startswith(prefix):
            return feature_name[len(prefix) :]
    return feature_name


def generate_feature_importance(top_n: int = 25) -> pd.DataFrame:
    report = {}
    if HOME_CREDIT_MODEL_REPORT_FILE.exists():
        report = json.loads(HOME_CREDIT_MODEL_REPORT_FILE.read_text())

    model_path = _resolve_model_path(report.get("best_model_by_roc_auc", {}).get("path"))
    pipeline = joblib.load(model_path)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    if not hasattr(classifier, "feature_importances_"):
        raise TypeError("The saved best model does not expose tree-based feature importances.")

    feature_names = [_clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    importances = classifier.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            "Feature name count does not match model importance count. "
            "Re-train the model and then re-run this script."
        )

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(HOME_CREDIT_FEATURE_IMPORTANCE_CSV, index=False)

    plot_df = importance_df.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#386641")
    ax.set_title("Top Model Features")
    ax.set_xlabel("XGBoost feature importance")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(HOME_CREDIT_FEATURE_IMPORTANCE_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Feature importance report complete")
    print(f"Model used: {model_path}")
    print(f"Saved feature importance table to: {HOME_CREDIT_FEATURE_IMPORTANCE_CSV}")
    print(f"Saved feature importance chart to: {HOME_CREDIT_FEATURE_IMPORTANCE_PNG}")
    return importance_df


if __name__ == "__main__":
    generate_feature_importance()
