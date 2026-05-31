from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import HOME_CREDIT_DIR


REQUIRED_HOME_CREDIT_FILES = [
    "application_train.csv",
    "previous_application.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
]


def validate_home_credit_data(base_dir: Path = HOME_CREDIT_DIR) -> list[Path]:
    missing = [base_dir / name for name in REQUIRED_HOME_CREDIT_FILES if not (base_dir / name).exists()]
    return missing


def load_home_credit_application_train(base_dir: Path = HOME_CREDIT_DIR) -> pd.DataFrame:
    missing = validate_home_credit_data(base_dir)
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Home Credit files are missing. Expected these files:\n"
            f"{missing_text}\n"
            "Download at least application_train.csv and previous_application.csv from Kaggle "
            "and extract them into data/raw/home_credit."
        )

    return pd.read_csv(base_dir / "application_train.csv")


def load_home_credit_previous_application(base_dir: Path = HOME_CREDIT_DIR) -> pd.DataFrame:
    missing = validate_home_credit_data(base_dir)
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Home Credit files are missing. Expected these files:\n"
            f"{missing_text}\n"
            "Download at least application_train.csv and previous_application.csv from Kaggle "
            "and extract them into data/raw/home_credit."
        )

    return pd.read_csv(base_dir / "previous_application.csv")


def load_home_credit_bureau(base_dir: Path = HOME_CREDIT_DIR) -> pd.DataFrame:
    missing = validate_home_credit_data(base_dir)
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Home Credit files are missing. Expected these files:\n"
            f"{missing_text}\n"
            "Download the Home Credit competition data and extract it into data/raw/home_credit."
        )

    usecols = [
        "SK_ID_CURR",
        "SK_ID_BUREAU",
        "CREDIT_ACTIVE",
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
        "CREDIT_TYPE",
        "DAYS_CREDIT_UPDATE",
        "AMT_ANNUITY",
    ]
    return pd.read_csv(base_dir / "bureau.csv", usecols=usecols)


def load_home_credit_bureau_balance(base_dir: Path = HOME_CREDIT_DIR) -> pd.DataFrame:
    missing = validate_home_credit_data(base_dir)
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Home Credit files are missing. Expected these files:\n"
            f"{missing_text}\n"
            "Download the Home Credit competition data and extract it into data/raw/home_credit."
        )

    usecols = ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"]
    return pd.read_csv(base_dir / "bureau_balance.csv", usecols=usecols)


def load_home_credit_installments(base_dir: Path = HOME_CREDIT_DIR) -> pd.DataFrame:
    missing = validate_home_credit_data(base_dir)
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Home Credit files are missing. Expected these files:\n"
            f"{missing_text}\n"
            "Download the Home Credit competition data and extract it into data/raw/home_credit."
        )

    usecols = [
        "SK_ID_CURR",
        "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ]
    return pd.read_csv(base_dir / "installments_payments.csv", usecols=usecols)


def load_home_credit_pos_cash(base_dir: Path = HOME_CREDIT_DIR) -> pd.DataFrame:
    missing = validate_home_credit_data(base_dir)
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Home Credit files are missing. Expected these files:\n"
            f"{missing_text}\n"
            "Download the Home Credit competition data and extract it into data/raw/home_credit."
        )

    usecols = [
        "SK_ID_CURR",
        "MONTHS_BALANCE",
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        "NAME_CONTRACT_STATUS",
        "SK_DPD",
        "SK_DPD_DEF",
    ]
    return pd.read_csv(base_dir / "POS_CASH_balance.csv", usecols=usecols)


def load_home_credit_credit_card(base_dir: Path = HOME_CREDIT_DIR) -> pd.DataFrame:
    missing = validate_home_credit_data(base_dir)
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Home Credit files are missing. Expected these files:\n"
            f"{missing_text}\n"
            "Download the Home Credit competition data and extract it into data/raw/home_credit."
        )

    usecols = [
        "SK_ID_CURR",
        "MONTHS_BALANCE",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_ATM_CURRENT",
        "AMT_DRAWINGS_CURRENT",
        "AMT_DRAWINGS_POS_CURRENT",
        "AMT_INST_MIN_REGULARITY",
        "AMT_PAYMENT_CURRENT",
        "AMT_PAYMENT_TOTAL_CURRENT",
        "AMT_RECEIVABLE_PRINCIPAL",
        "AMT_RECIVABLE",
        "AMT_TOTAL_RECEIVABLE",
        "CNT_DRAWINGS_ATM_CURRENT",
        "CNT_DRAWINGS_CURRENT",
        "CNT_DRAWINGS_POS_CURRENT",
        "CNT_INSTALMENT_MATURE_CUM",
        "NAME_CONTRACT_STATUS",
        "SK_DPD",
        "SK_DPD_DEF",
    ]
    return pd.read_csv(base_dir / "credit_card_balance.csv", usecols=usecols)
