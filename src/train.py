from __future__ import annotations

import joblib
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

from .config import DATA_FILE, MODEL_FILE, MODELS_DIR
from .data_loader import load_dataset
from .generate_sample_data import save_sample_dataset
from .preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN, build_preprocessor


def train_model() -> Pipeline:
    if not DATA_FILE.exists():
        save_sample_dataset(DATA_FILE)

    df = load_dataset(DATA_FILE)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("Dataset shape:", df.shape)
    print("Target distribution:")
    print(y.value_counts(normalize=True).rename("ratio").round(3))
    print()
    print("Model evaluation")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    print()
    print(f"Saved trained pipeline to: {MODEL_FILE}")
    return pipeline


if __name__ == "__main__":
    train_model()
