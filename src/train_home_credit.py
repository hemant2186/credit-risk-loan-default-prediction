from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import (
    BASE_DIR,
    HOME_CREDIT_ADVANCED_MODEL_FILE,
    HOME_CREDIT_DEMO_DATA_FILE,
    HOME_CREDIT_MODEL_FILE,
    HOME_CREDIT_MODEL_COMPARISON_PNG,
    HOME_CREDIT_MODEL_REPORT_FILE,
    HOME_CREDIT_THRESHOLD_PNG,
    HOME_CREDIT_XGBOOST_MODEL_FILE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from .home_credit import (
    load_home_credit_application_train,
    load_home_credit_bureau,
    load_home_credit_bureau_balance,
    load_home_credit_credit_card,
    load_home_credit_installments,
    load_home_credit_pos_cash,
    load_home_credit_previous_application,
)


# Keep sklearn predictable on Windows/sandboxed environments.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _prepare_previous_application_features(previous_df: pd.DataFrame) -> pd.DataFrame:
    prev = previous_df.copy()

    numeric_candidates = [
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "HOUR_APPR_PROCESS_START",
        "NFLAG_LAST_APPL_IN_DAY",
        "RATE_DOWN_PAYMENT",
        "DAYS_DECISION",
        "CNT_PAYMENT",
    ]
    existing_numeric = [col for col in numeric_candidates if col in prev.columns]

    aggregated_frames: list[pd.DataFrame] = []

    if existing_numeric:
        numeric_agg = prev.groupby("SK_ID_CURR")[existing_numeric].agg(["mean", "max", "min"])
        numeric_agg.columns = [
            f"PREV_{column}_{stat}".upper()
            for column, stat in numeric_agg.columns.to_flat_index()
        ]
        aggregated_frames.append(numeric_agg)

    categorical_candidates = [
        "NAME_CONTRACT_STATUS",
        "NAME_CONTRACT_TYPE",
        "WEEKDAY_APPR_PROCESS_START",
        "NAME_CASH_LOAN_PURPOSE",
    ]
    existing_categorical = [col for col in categorical_candidates if col in prev.columns]

    for column in existing_categorical:
        counts = (
            prev.groupby(["SK_ID_CURR", column])
            .size()
            .unstack(fill_value=0)
            .add_prefix(f"PREV_{column}_".upper())
        )
        aggregated_frames.append(counts)

    previous_count = prev.groupby("SK_ID_CURR").size().rename("PREV_APPLICATION_COUNT")
    aggregated_frames.append(previous_count.to_frame())

    if not aggregated_frames:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    previous_features = pd.concat(aggregated_frames, axis=1).reset_index()
    return previous_features


def _prepare_bureau_features(bureau_df: pd.DataFrame, bureau_balance_df: pd.DataFrame) -> pd.DataFrame:
    bureau = bureau_df.copy()
    bureau_balance = bureau_balance_df.copy()

    bureau_balance["BUREAU_BALANCE_STATUS_IS_DPD"] = bureau_balance["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)
    bureau_balance["BUREAU_BALANCE_STATUS_IS_WRITEOFF"] = (bureau_balance["STATUS"] == "5").astype(int)

    balance_agg = bureau_balance.groupby("SK_ID_BUREAU").agg(
        BUREAU_BALANCE_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        BUREAU_BALANCE_MONTHS_MIN=("MONTHS_BALANCE", "min"),
        BUREAU_BALANCE_MONTHS_MAX=("MONTHS_BALANCE", "max"),
        BUREAU_BALANCE_DPD_RATE=("BUREAU_BALANCE_STATUS_IS_DPD", "mean"),
        BUREAU_BALANCE_WRITEOFF_RATE=("BUREAU_BALANCE_STATUS_IS_WRITEOFF", "mean"),
    )

    bureau = bureau.merge(balance_agg, on="SK_ID_BUREAU", how="left")
    bureau["BUREAU_IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
    bureau["BUREAU_IS_CLOSED"] = (bureau["CREDIT_ACTIVE"] == "Closed").astype(int)

    numeric_candidates = [
        "CREDIT_DAY_OVERDUE",
        "DAYS_CREDIT",
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "AMT_CREDIT_MAX_OVERDUE",
        "CNT_CREDIT_PROLONG",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_SUM_OVERDUE",
        "DAYS_CREDIT_UPDATE",
        "AMT_ANNUITY",
        "BUREAU_BALANCE_MONTHS_COUNT",
        "BUREAU_BALANCE_MONTHS_MIN",
        "BUREAU_BALANCE_MONTHS_MAX",
        "BUREAU_BALANCE_DPD_RATE",
        "BUREAU_BALANCE_WRITEOFF_RATE",
        "BUREAU_IS_ACTIVE",
        "BUREAU_IS_CLOSED",
    ]
    existing_numeric = [col for col in numeric_candidates if col in bureau.columns]

    aggregated_frames: list[pd.DataFrame] = []

    if existing_numeric:
        numeric_agg = bureau.groupby("SK_ID_CURR")[existing_numeric].agg(["mean", "max", "sum"])
        numeric_agg.columns = [
            f"BUREAU_{column}_{stat}".upper()
            for column, stat in numeric_agg.columns.to_flat_index()
        ]
        aggregated_frames.append(numeric_agg)

    for column in ["CREDIT_ACTIVE", "CREDIT_TYPE"]:
        if column in bureau.columns:
            counts = (
                bureau.groupby(["SK_ID_CURR", column]).size().unstack(fill_value=0).add_prefix(f"BUREAU_{column}_".upper())
            )
            aggregated_frames.append(counts)

    bureau_count = bureau.groupby("SK_ID_CURR").size().rename("BUREAU_RECORD_COUNT")
    aggregated_frames.append(bureau_count.to_frame())

    bureau_features = pd.concat(aggregated_frames, axis=1).reset_index()
    return bureau_features


def _prepare_installments_features(installments_df: pd.DataFrame) -> pd.DataFrame:
    installments = installments_df.copy()
    installments["INST_PAYMENT_GAP"] = installments["AMT_PAYMENT"] - installments["AMT_INSTALMENT"]
    installments["INST_DAYS_PAST_DUE"] = (
        installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    ).clip(lower=0)
    installments["INST_DAYS_EARLY"] = (
        installments["DAYS_INSTALMENT"] - installments["DAYS_ENTRY_PAYMENT"]
    ).clip(lower=0)

    agg = installments.groupby("SK_ID_CURR").agg(
        INST_RECORD_COUNT=("AMT_PAYMENT", "count"),
        INST_PAYMENT_GAP_MEAN=("INST_PAYMENT_GAP", "mean"),
        INST_PAYMENT_GAP_MIN=("INST_PAYMENT_GAP", "min"),
        INST_PAYMENT_GAP_MAX=("INST_PAYMENT_GAP", "max"),
        INST_DAYS_PAST_DUE_MEAN=("INST_DAYS_PAST_DUE", "mean"),
        INST_DAYS_PAST_DUE_MAX=("INST_DAYS_PAST_DUE", "max"),
        INST_DAYS_EARLY_MEAN=("INST_DAYS_EARLY", "mean"),
        INST_AMT_PAYMENT_SUM=("AMT_PAYMENT", "sum"),
        INST_AMT_INSTALMENT_SUM=("AMT_INSTALMENT", "sum"),
        INST_NUM_INSTALMENT_NUMBER_MAX=("NUM_INSTALMENT_NUMBER", "max"),
    )
    return agg.reset_index()


def _prepare_pos_cash_features(pos_cash_df: pd.DataFrame) -> pd.DataFrame:
    pos = pos_cash_df.copy()
    pos["POS_IS_DPD"] = (pos["SK_DPD"] > 0).astype(int)
    pos["POS_IS_DPD_DEF"] = (pos["SK_DPD_DEF"] > 0).astype(int)

    aggregated_frames: list[pd.DataFrame] = [
        pos.groupby("SK_ID_CURR").agg(
            POS_RECORD_COUNT=("MONTHS_BALANCE", "count"),
            POS_MONTHS_BALANCE_MIN=("MONTHS_BALANCE", "min"),
            POS_MONTHS_BALANCE_MAX=("MONTHS_BALANCE", "max"),
            POS_CNT_INSTALMENT_MEAN=("CNT_INSTALMENT", "mean"),
            POS_CNT_INSTALMENT_FUTURE_MEAN=("CNT_INSTALMENT_FUTURE", "mean"),
            POS_SK_DPD_MEAN=("SK_DPD", "mean"),
            POS_SK_DPD_MAX=("SK_DPD", "max"),
            POS_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
            POS_IS_DPD_RATE=("POS_IS_DPD", "mean"),
            POS_IS_DPD_DEF_RATE=("POS_IS_DPD_DEF", "mean"),
        )
    ]

    if "NAME_CONTRACT_STATUS" in pos.columns:
        status_counts = (
            pos.groupby(["SK_ID_CURR", "NAME_CONTRACT_STATUS"]).size().unstack(fill_value=0).add_prefix("POS_STATUS_")
        )
        aggregated_frames.append(status_counts)

    return pd.concat(aggregated_frames, axis=1).reset_index()


def _prepare_credit_card_features(credit_card_df: pd.DataFrame) -> pd.DataFrame:
    credit = credit_card_df.copy()
    denominator = credit["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, pd.NA)
    credit["CC_UTILIZATION"] = credit["AMT_BALANCE"] / denominator
    credit["CC_IS_DPD"] = (credit["SK_DPD"] > 0).astype(int)
    credit["CC_IS_DPD_DEF"] = (credit["SK_DPD_DEF"] > 0).astype(int)

    aggregated_frames: list[pd.DataFrame] = [
        credit.groupby("SK_ID_CURR").agg(
            CC_RECORD_COUNT=("MONTHS_BALANCE", "count"),
            CC_MONTHS_BALANCE_MIN=("MONTHS_BALANCE", "min"),
            CC_MONTHS_BALANCE_MAX=("MONTHS_BALANCE", "max"),
            CC_AMT_BALANCE_MEAN=("AMT_BALANCE", "mean"),
            CC_AMT_BALANCE_MAX=("AMT_BALANCE", "max"),
            CC_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
            CC_DRAWINGS_CURRENT_MEAN=("AMT_DRAWINGS_CURRENT", "mean"),
            CC_DRAWINGS_POS_MEAN=("AMT_DRAWINGS_POS_CURRENT", "mean"),
            CC_PAYMENT_CURRENT_MEAN=("AMT_PAYMENT_CURRENT", "mean"),
            CC_PAYMENT_TOTAL_MEAN=("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
            CC_RECEIVABLE_MEAN=("AMT_TOTAL_RECEIVABLE", "mean"),
            CC_UTILIZATION_MEAN=("CC_UTILIZATION", "mean"),
            CC_UTILIZATION_MAX=("CC_UTILIZATION", "max"),
            CC_DPD_MEAN=("SK_DPD", "mean"),
            CC_DPD_MAX=("SK_DPD", "max"),
            CC_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
            CC_IS_DPD_RATE=("CC_IS_DPD", "mean"),
            CC_IS_DPD_DEF_RATE=("CC_IS_DPD_DEF", "mean"),
        )
    ]

    if "NAME_CONTRACT_STATUS" in credit.columns:
        status_counts = (
            credit.groupby(["SK_ID_CURR", "NAME_CONTRACT_STATUS"]).size().unstack(fill_value=0).add_prefix("CC_STATUS_")
        )
        aggregated_frames.append(status_counts)

    return pd.concat(aggregated_frames, axis=1).reset_index()


def _build_training_dataframe() -> pd.DataFrame:
    application_df = load_home_credit_application_train()
    previous_df = load_home_credit_previous_application()
    bureau_df = load_home_credit_bureau()
    bureau_balance_df = load_home_credit_bureau_balance()
    installments_df = load_home_credit_installments()
    pos_cash_df = load_home_credit_pos_cash()
    credit_card_df = load_home_credit_credit_card()

    previous_features = _prepare_previous_application_features(previous_df)
    bureau_features = _prepare_bureau_features(bureau_df, bureau_balance_df)
    installments_features = _prepare_installments_features(installments_df)
    pos_cash_features = _prepare_pos_cash_features(pos_cash_df)
    credit_card_features = _prepare_credit_card_features(credit_card_df)

    merged_df = application_df.merge(previous_features, on="SK_ID_CURR", how="left")
    merged_df = merged_df.merge(bureau_features, on="SK_ID_CURR", how="left")
    merged_df = merged_df.merge(installments_features, on="SK_ID_CURR", how="left")
    merged_df = merged_df.merge(pos_cash_features, on="SK_ID_CURR", how="left")
    merged_df = merged_df.merge(credit_card_features, on="SK_ID_CURR", how="left")
    return merged_df


def build_home_credit_training_dataframe() -> pd.DataFrame:
    return _build_training_dataframe()


def _get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    target = df["TARGET"]
    drop_columns = ["TARGET", "SK_ID_CURR"]
    features = df.drop(columns=[col for col in drop_columns if col in df.columns])

    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=["number"]).columns.tolist()
    return features, target, numeric_features, categorical_features


def _build_logistic_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced")),
        ]
    )
    return pipeline


def _build_random_forest_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=18,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                    random_state=42,
                ),
            ),
        ]
    )
    return pipeline


def _build_xgboost_pipeline(
    numeric_features: list[str], categorical_features: list[str], scale_pos_weight: float
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="auc",
                    reg_lambda=1.0,
                    min_child_weight=5,
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=1,
                    random_state=42,
                ),
            ),
        ]
    )
    return pipeline


def _evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def _print_metrics(title: str, metrics: dict[str, object]) -> None:
    print(title)
    print(f"Accuracy : {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall   : {metrics['recall']:.3f}")
    print(f"F1 Score : {metrics['f1']:.3f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.3f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print()


def _save_demo_dataset(df: pd.DataFrame) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    demo_df = df.sample(n=min(2500, len(df)), random_state=42).sort_values("SK_ID_CURR")
    demo_df.to_csv(HOME_CREDIT_DEMO_DATA_FILE, index=False, compression="gzip")


def _save_model_report(
    df: pd.DataFrame,
    metrics_list: list[dict[str, object]],
    best_model_path: str,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_map = {metrics["name"]: metrics for metrics in metrics_list}
    best_metrics = max(metrics_list, key=lambda item: item["roc_auc"])
    try:
        model_path_for_report = Path(best_model_path).resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        model_path_for_report = best_model_path

    report = {
        "dataset_shape": [int(df.shape[0]), int(df.shape[1])],
        "default_rate": round(float(df["TARGET"].mean()), 6),
        "models": model_map,
        "best_model_by_roc_auc": {
            "name": best_metrics["name"],
            "path": model_path_for_report,
        },
        "project_strengths": [
            "multi-table feature engineering across application, bureau, POS cash, credit card, and installment histories",
            "class-imbalance aware evaluation using ROC-AUC, precision, recall, and F1",
            "saved production-ready pipeline artifact for reproducible scoring",
        ],
    }
    HOME_CREDIT_MODEL_REPORT_FILE.write_text(json.dumps(report, indent=2))


def _save_portfolio_charts(metrics_list: list[dict[str, object]]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_list)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    metrics_df.plot(x="name", y="roc_auc", kind="bar", legend=False, ax=axes[0], color="#2a9d8f")
    axes[0].set_title("Model ROC-AUC Comparison")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].tick_params(axis="x", rotation=15)

    metrics_df.plot(x="name", y=["precision", "recall", "f1"], kind="bar", ax=axes[1])
    axes[1].set_title("Precision / Recall / F1")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Score")
    axes[1].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(HOME_CREDIT_MODEL_COMPARISON_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_home_credit_baseline() -> Pipeline:
    df = _build_training_dataframe()
    X, y, numeric_features, categorical_features = _get_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logistic_pipeline = _build_logistic_pipeline(numeric_features, categorical_features)
    logistic_pipeline.fit(X_train, y_train)
    logistic_metrics = _evaluate_model("logistic_regression", logistic_pipeline, X_test, y_test)

    random_forest_pipeline = _build_random_forest_pipeline(numeric_features, categorical_features)
    random_forest_pipeline.fit(X_train, y_train)
    random_forest_metrics = _evaluate_model("random_forest", random_forest_pipeline, X_test, y_test)

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    xgboost_pipeline = _build_xgboost_pipeline(numeric_features, categorical_features, scale_pos_weight)
    xgboost_pipeline.fit(X_train, y_train)
    xgboost_metrics = _evaluate_model("xgboost", xgboost_pipeline, X_test, y_test)

    print("Home Credit engineered model comparison")
    print("Dataset shape:", df.shape)
    print("Target distribution:")
    print(y.value_counts(normalize=True).rename("ratio").round(3))
    print()
    _print_metrics("Logistic Regression evaluation", logistic_metrics)
    _print_metrics("Random Forest evaluation", random_forest_metrics)
    _print_metrics("XGBoost evaluation", xgboost_metrics)

    models = [
        ("logistic_regression", logistic_pipeline, logistic_metrics, HOME_CREDIT_MODEL_FILE),
        ("random_forest", random_forest_pipeline, random_forest_metrics, HOME_CREDIT_ADVANCED_MODEL_FILE),
        ("xgboost", xgboost_pipeline, xgboost_metrics, HOME_CREDIT_XGBOOST_MODEL_FILE),
    ]
    best_name, best_pipeline, best_metrics, best_model_path = max(models, key=lambda item: item[2]["roc_auc"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic_pipeline, HOME_CREDIT_MODEL_FILE)
    joblib.dump(random_forest_pipeline, HOME_CREDIT_ADVANCED_MODEL_FILE)
    joblib.dump(xgboost_pipeline, HOME_CREDIT_XGBOOST_MODEL_FILE)
    joblib.dump(best_pipeline, best_model_path)
    _save_demo_dataset(df)
    metrics_list = [logistic_metrics, random_forest_metrics, xgboost_metrics]
    _save_model_report(df, metrics_list, str(best_model_path))
    _save_portfolio_charts(metrics_list)
    print()
    print(f"Saved logistic baseline model to: {HOME_CREDIT_MODEL_FILE}")
    print(f"Saved random forest model to: {HOME_CREDIT_ADVANCED_MODEL_FILE}")
    print(f"Saved xgboost model to: {HOME_CREDIT_XGBOOST_MODEL_FILE}")
    print(f"Saved best model ({best_name}) to: {best_model_path}")
    print(f"Saved demo dataset to: {HOME_CREDIT_DEMO_DATA_FILE}")
    print(f"Saved model report to: {HOME_CREDIT_MODEL_REPORT_FILE}")
    print(f"Saved comparison chart to: {HOME_CREDIT_MODEL_COMPARISON_PNG}")
    return best_pipeline


if __name__ == "__main__":
    train_home_credit_baseline()
