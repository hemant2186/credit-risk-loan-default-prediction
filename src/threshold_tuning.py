from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import (
    HOME_CREDIT_THRESHOLD_CSV,
    HOME_CREDIT_THRESHOLD_PNG,
    HOME_CREDIT_THRESHOLD_SUMMARY_FILE,
    HOME_CREDIT_MODEL_REPORT_FILE,
    REPORTS_DIR,
)
from .train_home_credit import _build_logistic_pipeline, _get_feature_matrix, build_home_credit_training_dataframe


def _build_threshold_table(y_true: pd.Series, y_proba: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for threshold in [round(x, 2) for x in [i / 100 for i in range(10, 91, 5)]]:
        y_pred = (y_proba >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def tune_thresholds() -> pd.DataFrame:
    df = build_home_credit_training_dataframe()
    X, y, numeric_features, categorical_features = _get_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    report = {}
    if HOME_CREDIT_MODEL_REPORT_FILE.exists():
        report = json.loads(HOME_CREDIT_MODEL_REPORT_FILE.read_text())

    best_model_path = report.get("best_model_by_roc_auc", {}).get("path")
    if best_model_path and pd.notna(best_model_path):
        pipeline = joblib.load(best_model_path)
    else:
        pipeline = _build_logistic_pipeline(numeric_features, categorical_features)
        pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    threshold_table = _build_threshold_table(y_test, y_proba)

    best_f1_row = threshold_table.sort_values(["f1", "precision"], ascending=[False, False]).iloc[0]
    shortlist = threshold_table[threshold_table["precision"] >= 0.20]
    if shortlist.empty:
        business_row = threshold_table.sort_values(["precision", "recall"], ascending=[False, False]).iloc[0]
    else:
        business_row = shortlist.sort_values(["recall", "f1"], ascending=[False, False]).iloc[0]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    threshold_table.to_csv(HOME_CREDIT_THRESHOLD_CSV, index=False)
    threshold_table.plot(x="threshold", y=["precision", "recall", "f1"], figsize=(8, 4), marker="o")
    plt.title("Threshold Tradeoff: Precision vs Recall vs F1")
    plt.ylabel("Score")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(HOME_CREDIT_THRESHOLD_PNG, dpi=160, bbox_inches="tight")
    plt.close()

    summary = {
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 6),
        "model_used": report.get("best_model_by_roc_auc", {}).get("name", "logistic_regression"),
        "recommended_threshold_f1": round(float(best_f1_row["threshold"]), 2),
        "recommended_threshold_business": round(float(business_row["threshold"]), 2),
        "best_f1_metrics": {
            "threshold": round(float(best_f1_row["threshold"]), 2),
            "accuracy": round(float(best_f1_row["accuracy"]), 4),
            "precision": round(float(best_f1_row["precision"]), 4),
            "recall": round(float(best_f1_row["recall"]), 4),
            "f1": round(float(best_f1_row["f1"]), 4),
        },
        "business_metrics": {
            "threshold": round(float(business_row["threshold"]), 2),
            "accuracy": round(float(business_row["accuracy"]), 4),
            "precision": round(float(business_row["precision"]), 4),
            "recall": round(float(business_row["recall"]), 4),
            "f1": round(float(business_row["f1"]), 4),
        },
        "business_note": (
            "Lower thresholds catch more risky borrowers but generate more false alarms. "
            "Higher thresholds are stricter and usually improve precision while lowering recall."
        ),
    }
    HOME_CREDIT_THRESHOLD_SUMMARY_FILE.write_text(json.dumps(summary, indent=2))

    print("Threshold tuning complete")
    print(f"Saved threshold table to: {HOME_CREDIT_THRESHOLD_CSV}")
    print(f"Saved threshold summary to: {HOME_CREDIT_THRESHOLD_SUMMARY_FILE}")
    print(f"Saved threshold chart to: {HOME_CREDIT_THRESHOLD_PNG}")
    print()
    print("Best F1 threshold:")
    print(best_f1_row.to_string())
    print()
    print("Business-friendly threshold:")
    print(business_row.to_string())

    return threshold_table


if __name__ == "__main__":
    tune_thresholds()
