# CreditRisk AI - Credit Risk SaaS

CreditRisk AI is a deployable Streamlit SaaS-style product for borrower default-risk scoring. It lets lenders and analysts upload applicant CSV files, score default probability in batch, tune decision thresholds, download decisions, and inspect model performance.

**Live app:** https://credit-risk-loan.streamlit.app/

**Dataset:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)

## Product Capabilities

- Batch applicant scoring from uploaded CSV files
- Downloadable scored applicant output
- Threshold policy testing for risk teams
- Single-applicant review workflow
- Model comparison, threshold reports, and feature importance
- Deployment-ready Streamlit configuration
- Cloud-safe fallback model for deployment compatibility

## Internship Readiness

This project is strong enough to showcase for data science and machine learning internships because it demonstrates:

- End-to-end ML delivery, from raw multi-table data to a deployed user-facing product
- Feature engineering across application, bureau, installment, POS cash, and credit card histories
- Imbalanced classification evaluation using precision, recall, F1, and ROC-AUC
- Decision-threshold tuning for real lending tradeoffs
- Model explainability through feature-importance reporting
- Practical deployment work on Streamlit Cloud with reproducible dependency pinning

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Streamlit

## SaaS Workflow

1. Upload borrower-level applicant data.
2. The app validates/fills the model schema.
3. The model scores default probability for every applicant.
4. A configurable threshold maps scores into approval/review decisions.
5. Users download scored applicants for business review.
6. Analysts inspect feature importance and threshold tradeoffs.

## Project Structure

- `.streamlit/`: Streamlit deployment configuration
- `data/raw/home_credit/`: Kaggle competition data
- `data/processed/`: processed artifacts for demos
- `models/`: saved model pipelines
- `reports/`: model comparison and threshold analysis outputs
- `MODEL_CARD.md`: model purpose, limitations, metrics, and responsible-use notes
- `DEPLOYMENT.md`: cloud deployment guide
- `notebooks/`: EDA notebook for presentation
- `src/`: training, feature engineering, and evaluation code
- `app.py`: SaaS-style Streamlit application

## Modeling Workflow

1. Load `application_train.csv` as the core borrower table
2. Aggregate behavioral features from:
   - `previous_application.csv`
   - `bureau.csv`
   - `bureau_balance.csv`
   - `installments_payments.csv`
   - `POS_CASH_balance.csv`
   - `credit_card_balance.csv`
3. Merge engineered features back to the main applicant table
4. Split into train and test sets with stratification
5. Train and compare:
   - Logistic Regression
   - Random Forest
   - XGBoost
6. Save the best-performing model
7. Tune thresholds to reflect different lending business priorities
8. Surface everything in a Streamlit product app
9. Generate feature importance artifacts for model explainability

## Current Results

Engineered dataset:

- Rows: `307,511`
- Columns: `324`
- Default rate: about `8.1%`

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------|---------|---------|---------|---------|---------|
| Logistic Regression | 0.747 | 0.183 | 0.615 | 0.282 | 0.755 |
| Random Forest | 0.647 | 0.140 | 0.654 | 0.231 | 0.707 |
| XGBoost | 0.730 | 0.185 | 0.690 | 0.292 | 0.781 |


**Best Saved Model:** `models/home_credit_xgboost.joblib`

### Deployment Status

- Live app: https://credit-risk-loan.streamlit.app/
- Primary trained model: XGBoost pipeline saved in `models/home_credit_xgboost.joblib`
- Cloud fallback: Logistic Regression trained from packaged demo applicants if the hosted Python runtime cannot unpickle the saved XGBoost artifact
- Recommended Streamlit Cloud runtime: Python `3.11`

### Threshold Tuning

- Best F1 Threshold: `0.70`
- Business Threshold: `0.55`

The model achieved its highest F1 score at a threshold of 0.70. A lower threshold of 0.55 improves recall and may be preferable when minimizing missed defaults is more important than precision.

## Product Screenshots

### Model comparison

![Model comparison](reports/home_credit_model_comparison.png)

### Threshold tradeoff

![Threshold tradeoff](reports/home_credit_threshold_tradeoff.png)

### Feature importance

![Feature importance](reports/home_credit_feature_importance.png)

## Key Files

- [src/train_home_credit.py](src/train_home_credit.py): multi-table feature engineering, model training, and comparison
- [src/threshold_tuning.py](src/threshold_tuning.py): threshold tradeoff analysis
- [src/explain_model.py](src/explain_model.py): feature importance report for the best saved model
- [app.py](app.py): SaaS-style Streamlit application
- [MODEL_CARD.md](MODEL_CARD.md): concise model card for responsible ML discussion
- [notebooks/home_credit_eda.ipynb](notebooks/home_credit_eda.ipynb): presentation-ready EDA notebook
- [reports/home_credit_model_comparison.json](reports/home_credit_model_comparison.json): saved model metrics
- [reports/home_credit_model_comparison.png](reports/home_credit_model_comparison.png): portfolio-ready model comparison visual
- [reports/home_credit_feature_importance.csv](reports/home_credit_feature_importance.csv): top model features
- [reports/home_credit_feature_importance.png](reports/home_credit_feature_importance.png): feature importance visual
- [reports/home_credit_threshold_summary.json](reports/home_credit_threshold_summary.json): recommended thresholds
- [reports/home_credit_threshold_tradeoff.png](reports/home_credit_threshold_tradeoff.png): threshold tradeoff visual for GitHub or slides

## How To Run Locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train models and generate artifacts:

```bash
python -m src.train_home_credit
```

Run threshold tuning:

```bash
python -m src.threshold_tuning
```

Generate model explainability artifacts:

```bash
python -m src.explain_model
```

Launch the Streamlit app:

```bash
streamlit run app.py
```

## Deploy

See [DEPLOYMENT.md](DEPLOYMENT.md).

Recommended cloud setup:

- Repository: `hemant2186/credit-risk-loan-default-prediction`
- Branch: `main`
- Main file: `app.py`

## Resume Bullets

- Built and deployed **CreditRisk AI**, a Streamlit SaaS-style app for batch borrower default-risk scoring, threshold policy testing, and downloadable lending decisions.
- Engineered borrower-level features from the Kaggle Home Credit dataset across application, bureau, installment, POS cash, credit card, and previous-application tables.
- Compared Logistic Regression, Random Forest, and XGBoost on an imbalanced credit-risk problem, achieving best ROC-AUC of `0.781` with threshold tuning for business tradeoffs.
- Added model explainability reports, deployment configuration, and cloud-safe fallback behavior for a reliable public demo.

## Resume Links

Live product:

```text
https://credit-risk-loan.streamlit.app/
```

GitHub project:

```text
https://github.com/hemant2186/credit-risk-loan-default-prediction
```

