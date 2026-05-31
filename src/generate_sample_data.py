from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_FILE, RAW_DATA_DIR


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_sample_dataset(n_rows: int = 1500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 61, n_rows)
    income = rng.normal(65000, 18000, n_rows).clip(18000, 180000)
    loan_amount = rng.normal(18000, 9000, n_rows).clip(2000, 60000)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], n_rows, p=[0.08, 0.17, 0.35, 0.22, 0.18])
    credit_score = rng.normal(680, 75, n_rows).clip(300, 850)
    employment_years = rng.integers(0, 31, n_rows)
    existing_loans_count = rng.integers(0, 6, n_rows)
    debt_to_income_ratio = rng.normal(0.32, 0.14, n_rows).clip(0.03, 0.85)

    loan_purpose = rng.choice(
        ["education", "medical", "personal", "business", "home_improvement"],
        n_rows,
        p=[0.17, 0.14, 0.34, 0.15, 0.20],
    )
    home_ownership = rng.choice(["rent", "mortgage", "own"], n_rows, p=[0.42, 0.38, 0.20])
    marital_status = rng.choice(["single", "married", "divorced"], n_rows, p=[0.43, 0.47, 0.10])

    risk_score = (
        1.6 * (loan_amount / income)
        + 2.4 * debt_to_income_ratio
        + 0.22 * existing_loans_count
        - 0.010 * (credit_score - 650)
        - 0.03 * employment_years
        + 0.003 * (loan_term_months - 24)
        + np.where(home_ownership == "rent", 0.28, 0.0)
        + np.where(loan_purpose == "personal", 0.22, 0.0)
        + np.where(loan_purpose == "business", 0.15, 0.0)
        + np.where(marital_status == "single", 0.10, 0.0)
        - 1.55
    )

    default_probability = _sigmoid(risk_score)
    loan_default = rng.binomial(1, default_probability)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income.round(2),
            "loan_amount": loan_amount.round(2),
            "loan_term_months": loan_term_months,
            "credit_score": credit_score.round(0).astype(int),
            "employment_years": employment_years,
            "existing_loans_count": existing_loans_count,
            "debt_to_income_ratio": debt_to_income_ratio.round(3),
            "loan_purpose": loan_purpose,
            "home_ownership": home_ownership,
            "marital_status": marital_status,
            "loan_default": loan_default,
        }
    )

    # Inject a small amount of missing data to make preprocessing realistic.
    for column in ["income", "credit_score", "employment_years", "home_ownership"]:
        missing_index = rng.choice(df.index, size=max(10, n_rows // 50), replace=False)
        df.loc[missing_index, column] = np.nan

    return df


def save_sample_dataset(output_path: Path = DATA_FILE) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_sample_dataset()
    dataset.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    path = save_sample_dataset()
    print(f"Sample dataset saved to: {path}")
