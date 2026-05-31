from __future__ import annotations

import joblib
import pandas as pd

from .config import MODEL_FILE


def predict_single_applicant() -> None:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_FILE}. Run `python -m src.train` first."
        )

    pipeline = joblib.load(MODEL_FILE)

    applicant = pd.DataFrame(
        [
            {
                "age": 29,
                "income": 42000,
                "loan_amount": 22000,
                "loan_term_months": 48,
                "credit_score": 610,
                "employment_years": 2,
                "existing_loans_count": 2,
                "debt_to_income_ratio": 0.48,
                "loan_purpose": "personal",
                "home_ownership": "rent",
                "marital_status": "single",
            }
        ]
    )

    prediction = int(pipeline.predict(applicant)[0])
    probability = float(pipeline.predict_proba(applicant)[0][1])

    label = "Likely to Default" if prediction == 1 else "Low Default Risk"

    print("Applicant data:")
    print(applicant.to_string(index=False))
    print()
    print("Prediction result:")
    print(f"Predicted class        : {prediction}")
    print(f"Default probability    : {probability:.3f}")
    print(f"Business interpretation: {label}")


if __name__ == "__main__":
    predict_single_applicant()
