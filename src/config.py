from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

DATA_FILE = RAW_DATA_DIR / "loan_data.csv"
MODEL_FILE = MODELS_DIR / "credit_risk_model.joblib"
HOME_CREDIT_DIR = RAW_DATA_DIR / "home_credit"
HOME_CREDIT_MODEL_FILE = MODELS_DIR / "home_credit_baseline.joblib"
HOME_CREDIT_ADVANCED_MODEL_FILE = MODELS_DIR / "home_credit_advanced.joblib"
HOME_CREDIT_XGBOOST_MODEL_FILE = MODELS_DIR / "home_credit_xgboost.joblib"
HOME_CREDIT_DEMO_DATA_FILE = PROCESSED_DATA_DIR / "home_credit_demo_applicants.csv.gz"
HOME_CREDIT_MODEL_REPORT_FILE = REPORTS_DIR / "home_credit_model_comparison.json"
HOME_CREDIT_THRESHOLD_CSV = REPORTS_DIR / "home_credit_thresholds.csv"
HOME_CREDIT_THRESHOLD_SUMMARY_FILE = REPORTS_DIR / "home_credit_threshold_summary.json"
HOME_CREDIT_MODEL_COMPARISON_PNG = REPORTS_DIR / "home_credit_model_comparison.png"
HOME_CREDIT_THRESHOLD_PNG = REPORTS_DIR / "home_credit_threshold_tradeoff.png"
HOME_CREDIT_FEATURE_IMPORTANCE_CSV = REPORTS_DIR / "home_credit_feature_importance.csv"
HOME_CREDIT_FEATURE_IMPORTANCE_PNG = REPORTS_DIR / "home_credit_feature_importance.png"
HOME_CREDIT_FAIRNESS_CSV = REPORTS_DIR / "home_credit_fairness_summary.csv"
HOME_CREDIT_FAIRNESS_JSON = REPORTS_DIR / "home_credit_fairness_summary.json"
