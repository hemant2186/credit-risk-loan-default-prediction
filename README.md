# Credit Risk Loan Default Prediction

An internship-ready data science project built on Kaggle's **Home Credit Default Risk** competition data. The project predicts borrower default risk using multi-table feature engineering, model comparison, threshold tuning, explainability artifacts, and a Streamlit demo.

## Why This Project Matters

Lenders do not just want a model with high accuracy. They need a system that helps them:

- identify risky borrowers early
- balance false approvals against false rejections
- explain tradeoffs between recall and precision
- operationalize a scoring pipeline on real applicant data

This project is designed around that framing.

## What Makes This Resume-Worthy

- Uses a real, industry-style Kaggle credit-risk dataset
- Engineers features from multiple linked tables, not just one flat CSV
- Handles class imbalance with business-relevant metrics
- Compares multiple models, including boosting, instead of stopping at a single baseline
- Tunes decision thresholds for lending tradeoffs
- Adds feature-importance reporting and a model card
- Includes a recruiter-friendly Streamlit demo
- Includes an EDA notebook for storytelling and insight generation

## Project Structure

- `data/raw/home_credit/`: Kaggle competition data
- `data/processed/`: processed artifacts for demos
- `models/`: saved model pipelines
- `reports/`: model comparison and threshold analysis outputs
- `MODEL_CARD.md`: model purpose, limitations, metrics, and responsible-use notes
- `notebooks/`: EDA notebook for presentation
- `src/`: training, feature engineering, and evaluation code
- `app.py`: Streamlit demo application

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
8. Surface everything in a Streamlit demo
9. Generate feature importance artifacts for model explainability

## Current Results

Engineered dataset:

- Rows: `307,511`
- Columns: `324`
- Default rate: about `8.1%`

Model comparison:

- Logistic Regression
  - Accuracy: `0.747`
  - Precision: `0.183`
  - Recall: `0.615`
  - F1: `0.282`
  - ROC-AUC: `0.755`
- Random Forest
  - Accuracy: `0.647`
  - Precision: `0.140`
  - Recall: `0.654`
  - F1: `0.231`
  - ROC-AUC: `0.707`
- XGBoost
  - Accuracy: `0.730`
  - Precision: `0.185`
  - Recall: `0.690`
  - F1: `0.292`
  - ROC-AUC: `0.781`

Best saved model:

- `models/home_credit_xgboost.joblib`

Threshold tuning:

- Best F1 threshold: `0.70`
- Business-friendly threshold: `0.55`

Interpretation:

- `0.70` is the stronger balanced threshold for the best model if you care more about F1.
- `0.55` is a more recall-friendly business threshold that still keeps precision above `0.20`.

## Key Files

- [src/train_home_credit.py](src/train_home_credit.py): multi-table feature engineering, model training, and comparison
- [src/threshold_tuning.py](src/threshold_tuning.py): threshold tradeoff analysis
- [src/explain_model.py](src/explain_model.py): feature importance report for the best saved model
- [app.py](app.py): Streamlit demo
- [MODEL_CARD.md](MODEL_CARD.md): concise model card for responsible ML discussion
- [notebooks/home_credit_eda.ipynb](notebooks/home_credit_eda.ipynb): presentation-ready EDA notebook
- [reports/home_credit_model_comparison.json](reports/home_credit_model_comparison.json): saved model metrics
- [reports/home_credit_model_comparison.png](reports/home_credit_model_comparison.png): portfolio-ready model comparison visual
- [reports/home_credit_feature_importance.csv](reports/home_credit_feature_importance.csv): top model features
- [reports/home_credit_feature_importance.png](reports/home_credit_feature_importance.png): feature importance visual
- [reports/home_credit_threshold_summary.json](reports/home_credit_threshold_summary.json): recommended thresholds
- [reports/home_credit_threshold_tradeoff.png](reports/home_credit_threshold_tradeoff.png): threshold tradeoff visual for GitHub or slides

## How To Run

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

Launch the Streamlit demo:

```bash
streamlit run app.py
```

## How To Talk About This On Your Resume

Example wording:

- Built a credit-risk default prediction pipeline on Kaggle Home Credit data using multi-table feature engineering across application, bureau, installment, POS cash, and credit card histories.
- Evaluated Logistic Regression, Random Forest, and XGBoost on an imbalanced credit-risk dataset, achieving a best ROC-AUC of `0.781` with threshold tuning for lending tradeoffs.
- Developed a Streamlit-based demo, EDA workflow, model card, feature-importance report, and portfolio-ready visuals to communicate borrower risk patterns and model behavior to non-technical stakeholders.

## Honest Assessment

Yes, this is strong enough to put on your resume for data science internships.

Why:

- it uses a recognized dataset
- it goes beyond beginner one-file modeling
- it shows feature engineering, evaluation maturity, and product thinking

What would make it even stronger later:

- SHAP-based explanations for individual applicants
- fairness analysis across demographic proxy groups
- a short presentation deck or screenshots captured from the running Streamlit app
