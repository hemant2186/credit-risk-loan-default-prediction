# CreditRisk AI Complete Project Defense Report

This report is written from the actual repository implementation as of 2026-06-29. The main production-style path is the Home Credit pipeline in `src/train_home_credit.py`, served by `api.py` and `app.py`. There is also a smaller synthetic-loan baseline path in `src/train.py`, `src/preprocess.py`, `src/generate_sample_data.py`, and `src/predict.py`.

## 1. Executive Summary

**Problem solved.** The project predicts whether a loan applicant is likely to experience repayment difficulty/default. In the Home Credit dataset, the target is `TARGET`, where `1` means repayment difficulty/default risk and `0` means no observed repayment difficulty. The app converts a model probability into a lending decision such as `Manual review / decline` or `Approve / monitor`.

**Banking value.** Banks lose money when they approve borrowers who default, but they also lose revenue and customer trust when they reject good borrowers. This project supports pre-screening, manual-review prioritization, portfolio monitoring, threshold policy testing, and repeatable scoring through a Streamlit UI and FastAPI endpoint.

**Why classification.** The target is discrete: default vs non-default. The model estimates `P(default = 1)` and then applies a threshold. This makes it binary classification, not regression. The probability is continuous, but the business action is categorical.

## 2. Architecture

Implemented flow:

```text
Raw Home Credit tables
  -> table loaders in src/home_credit.py
  -> feature aggregation in src/train_home_credit.py
  -> borrower-level merged dataframe
  -> drop TARGET and SK_ID_CURR from features
  -> stratified train/test split
  -> train Logistic Regression, Random Forest, XGBoost
  -> evaluate accuracy, precision, recall, F1, ROC-AUC
  -> save model artifacts and report artifacts
  -> tune thresholds
  -> generate explainability/fairness reports
  -> serve model through FastAPI and Streamlit
  -> score applicants and produce risk bands/decisions
```

Key files:

- `src/home_credit.py`: validates and loads Kaggle files: `application_train.csv`, `previous_application.csv`, `bureau.csv`, `bureau_balance.csv`, `installments_payments.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`.
- `src/train_home_credit.py`: main feature engineering, training, evaluation, model saving, demo-data saving, and chart/report creation.
- `src/threshold_tuning.py`: threshold sweep from 0.10 to 0.90.
- `src/explain_model.py`: XGBoost feature importance export.
- `src/fairness_analysis.py`: fairness snapshot by demographic/business groups.
- `api.py`: FastAPI scoring API.
- `app.py`: Streamlit SaaS-style application.
- `reports/*.json` and `reports/*.csv`: stored evaluation, threshold, feature importance, and fairness artifacts.
- `models/*.joblib`: trained pipelines.

Important distinction: `reports/home_credit_model_comparison.json` reports `dataset_shape = [307511, 324]`. That includes `TARGET` and `SK_ID_CURR`; the fitted Home Credit model uses 322 scoring input features after those two columns are removed.

## 3. Dataset Analysis

**Primary dataset.** Home Credit Default Risk from Kaggle, documented in `README.md`, `MODEL_CARD.md`, and `data/raw/home_credit/README.md`.

**Target variable.** `TARGET`, loaded from `application_train.csv`. `TARGET = 1` means repayment difficulty. `TARGET = 0` means no repayment difficulty.

**Training size and class imbalance.** `reports/home_credit_model_comparison.json` reports 307,511 rows and a default rate of 0.080729, about 8.1%. This is highly imbalanced: defaults are the minority class.

**Feature count.** The training dataframe has 324 columns, including `SK_ID_CURR` and `TARGET`. The fitted Home Credit model requires 322 input features. The packaged demo set `data/processed/home_credit_demo_applicants.csv.gz` has 2,500 rows and 324 columns.

**Most important features.** `reports/home_credit_feature_importance.csv` lists the top XGBoost features. Top 25 are:

1. `EXT_SOURCE_3`
2. `EXT_SOURCE_2`
3. `NAME_EDUCATION_TYPE_Higher education`
4. `CODE_GENDER_F`
5. `EXT_SOURCE_1`
6. `BUREAU_BUREAU_IS_CLOSED_MEAN`
7. `CODE_GENDER_M`
8. `NAME_INCOME_TYPE_Working`
9. `FLAG_DOCUMENT_3`
10. `INST_DAYS_PAST_DUE_MEAN`
11. `PREV_NAME_CONTRACT_STATUS_Refused`
12. `CC_DRAWINGS_CURRENT_MEAN`
13. `NAME_INCOME_TYPE_Pensioner`
14. `CC_RECEIVABLE_MEAN`
15. `FLAG_EMP_PHONE`
16. `PREV_CNT_PAYMENT_MAX`
17. `INST_AMT_PAYMENT_SUM`
18. `DAYS_BIRTH`
19. `BUREAU_CREDIT_TYPE_Microloan`
20. `NAME_EDUCATION_TYPE_Secondary / secondary special`
21. `OWN_CAR_AGE`
22. `AMT_GOODS_PRICE`
23. `NAME_CONTRACT_TYPE_Revolving loans`
24. `REGION_RATING_CLIENT_W_CITY`
25. `NAME_FAMILY_STATUS_Married`

**Potential leakage risks.**

- `TARGET` is dropped in `src/train_home_credit.py` before modeling, which is correct.
- `SK_ID_CURR` is dropped from model features, which avoids learning applicant IDs.
- Previous application, bureau, installment, POS, and credit-card records must represent information available before the current lending decision. In real production, each joined table would require timestamp cutoffs. The repository does not enforce point-in-time joins, so temporal leakage is the biggest production concern.
- `EXT_SOURCE_*` variables are highly predictive external scores. They are valid if available at application time, but they must be governed because they may encode opaque third-party risk signals.
- `CODE_GENDER`, education, income type, and family status are used by the model. These are fairness/compliance-sensitive and should be reviewed before a real lending deployment.
- The notebooks refer to processed files such as `train_cleaned.csv` and `train_fe.csv`, but the current repository only packages the Home Credit demo CSV. The notebook flow is educational, while the reproducible script flow is in `src/train_home_credit.py`.

## 4. Feature Engineering Deep Dive

### Simple ratio features in `src/feature_engineering.py`

These features are implemented in `create_features(df)`:

| Feature | Formula | Business meaning | Expected impact |
|---|---|---|---|
| `CREDIT_INCOME_RATIO` | `AMT_CREDIT / AMT_INCOME_TOTAL` | Credit exposure relative to annual income. | Higher values usually indicate higher repayment burden. |
| `ANNUITY_INCOME_RATIO` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | Periodic payment relative to income. | Higher values suggest affordability stress. |
| `CREDIT_ANNUITY_RATIO` | `AMT_CREDIT / AMT_ANNUITY` | Approximate repayment length/intensity. | Captures relationship between principal and payment. |
| `AGE_YEARS` | `-DAYS_BIRTH / 365` | Applicant age in interpretable years. | Nonlinear life-stage and stability signal. |
| `EMPLOYMENT_YEARS` | `-DAYS_EMPLOYED / 365` | Tenure/stability in employment. | Longer employment usually lowers risk. |
| `EMPLOYMENT_AGE_RATIO` | `EMPLOYMENT_YEARS / AGE_YEARS` | Share of adult life employed. | Normalizes tenure by age. |
| `INCOME_PER_PERSON` | `AMT_INCOME_TOTAL / CNT_FAM_MEMBERS` | Income available per household member. | Lower values can indicate household financial pressure. |

The notebook `notebooks/02_feature_engineering.ipynb` also creates risk flags:

| Feature | Formula | Business meaning | Expected impact |
|---|---|---|---|
| `HIGH_CREDIT_INCOME` | `CREDIT_INCOME_RATIO > 0.5` | Large loan relative to income. | Flags high leverage. |
| `LOW_EMPLOYMENT` | `EMPLOYMENT_YEARS < 1` | Short employment history. | Flags instability. |
| `HIGH_ANNUITY_BURDEN` | `ANNUITY_INCOME_RATIO > 0.3` | Payment consumes large income share. | Flags affordability stress. |

### Main Home Credit engineered feature families in `src/train_home_credit.py`

The main model does multi-table aggregation. For each borrower (`SK_ID_CURR`), it converts relational history into one row.

**Previous applications: `_prepare_previous_application_features`.**

- Numeric aggregation formula: for each existing numeric column among `AMT_ANNUITY`, `AMT_APPLICATION`, `AMT_CREDIT`, `AMT_DOWN_PAYMENT`, `AMT_GOODS_PRICE`, `HOUR_APPR_PROCESS_START`, `NFLAG_LAST_APPL_IN_DAY`, `RATE_DOWN_PAYMENT`, `DAYS_DECISION`, `CNT_PAYMENT`, compute `mean`, `max`, and `min` by `SK_ID_CURR`.
- Output name pattern: `PREV_<COLUMN>_<STAT>`.
- Categorical count formula: for `NAME_CONTRACT_STATUS`, `NAME_CONTRACT_TYPE`, `WEEKDAY_APPR_PROCESS_START`, `NAME_CASH_LOAN_PURPOSE`, count applications per category.
- Count formula: `PREV_APPLICATION_COUNT = count(previous applications)`.
- Business meaning: prior approval/refusal behavior, requested loan sizes, down payments, repayment plan length, and application frequency.
- Expected impact: rejected previous applications, many applications, or high prior credit requests can signal elevated risk.

**Bureau and bureau balance: `_prepare_bureau_features`.**

- `BUREAU_BALANCE_STATUS_IS_DPD = STATUS in ["1", "2", "3", "4", "5"]`.
- `BUREAU_BALANCE_STATUS_IS_WRITEOFF = STATUS == "5"`.
- Per bureau account: `BUREAU_BALANCE_MONTHS_COUNT`, `MONTHS_MIN`, `MONTHS_MAX`, `DPD_RATE`, `WRITEOFF_RATE`.
- Account flags: `BUREAU_IS_ACTIVE = CREDIT_ACTIVE == "Active"`, `BUREAU_IS_CLOSED = CREDIT_ACTIVE == "Closed"`.
- Borrower aggregations: for bureau numeric fields such as overdue days, credit age, end dates, max overdue, prolong count, credit sums, debt, limits, overdue sums, update recency, annuity, bureau-balance statistics, active/closed flags, compute `mean`, `max`, and `sum`.
- Categorical counts: counts by `CREDIT_ACTIVE` and `CREDIT_TYPE`.
- Count formula: `BUREAU_RECORD_COUNT = count(bureau accounts)`.
- Business meaning: outside-credit history, open debt burden, overdue records, and credit-product mix.
- Expected impact: bureau delinquency and active high-debt accounts should increase risk; clean closed accounts can lower risk.

**Installments: `_prepare_installments_features`.**

- `INST_PAYMENT_GAP = AMT_PAYMENT - AMT_INSTALMENT`.
- `INST_DAYS_PAST_DUE = max(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0)`.
- `INST_DAYS_EARLY = max(DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT, 0)`.
- Aggregations: record count, payment gap mean/min/max, days past due mean/max, days early mean, payment sum, installment sum, max installment number.
- Business meaning: historical payment completeness and timeliness.
- Expected impact: late payments and payment shortfalls increase default risk; early/on-time payments reduce risk.

**POS cash: `_prepare_pos_cash_features`.**

- `POS_IS_DPD = SK_DPD > 0`.
- `POS_IS_DPD_DEF = SK_DPD_DEF > 0`.
- Aggregations: record count, month range, mean current/future installments, mean/max DPD, mean default DPD, DPD rate, default-DPD rate.
- Categorical counts by `NAME_CONTRACT_STATUS`.
- Business meaning: behavior on point-of-sale/cash loans.
- Expected impact: repeated DPD statuses imply repayment stress.

**Credit card: `_prepare_credit_card_features`.**

- `CC_UTILIZATION = AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL`, with zero limits replaced by missing.
- `CC_IS_DPD = SK_DPD > 0`.
- `CC_IS_DPD_DEF = SK_DPD_DEF > 0`.
- Aggregations: record count, month range, balance mean/max, limit mean, drawings means, payment means, receivable mean, utilization mean/max, DPD mean/max, default DPD mean, DPD rate, default-DPD rate.
- Categorical counts by `NAME_CONTRACT_STATUS`.
- Business meaning: revolving-credit utilization, repayment behavior, and delinquency.
- Expected impact: high utilization, high receivables, and DPD events increase risk.

## 5. Model Comparison

The main comparison is in `src/train_home_credit.py`.

| Model | Why used | Advantages | Disadvantages | Why rejected as final |
|---|---|---|---|---|
| Logistic Regression | Baseline with `class_weight="balanced"`. | Interpretable, fast, stable probabilities, common in credit scoring. | Linear decision boundary; misses complex interactions. | ROC-AUC 0.755, lower than XGBoost. |
| Random Forest | Nonlinear tree ensemble with `class_weight="balanced_subsample"`. | Handles interactions, robust to outliers, feature importance available. | Lower ROC-AUC here, larger model, less calibrated, weaker ranking than XGBoost. | ROC-AUC 0.707, worst of the three. |
| XGBoost | Gradient-boosted trees with `scale_pos_weight`. | Strong tabular performance, handles nonlinearities, strong ranking, feature importances. | Less interpretable than logistic regression, artifact/version sensitivity, needs tuning. | Chosen as final because it had best ROC-AUC 0.781. |

## 6. Metrics Analysis

Metrics from `reports/home_credit_model_comparison.json`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.747 | 0.183 | 0.615 | 0.282 | 0.755 |
| Random Forest | 0.647 | 0.140 | 0.654 | 0.231 | 0.707 |
| XGBoost | 0.730 | 0.185 | 0.690 | 0.292 | 0.781 |

**Accuracy** measures total correct predictions, but it can be misleading because 91.9% of applicants are non-defaults. A model can look accurate by mostly predicting non-default.

**Precision** answers: among applicants flagged as risky, how many truly defaulted? High precision reduces unnecessary rejections/manual reviews of good borrowers.

**Recall** answers: among true defaulters, how many did the model catch? High recall reduces false approvals of risky borrowers.

**F1** balances precision and recall. It is useful when both missed defaults and false alarms matter.

**ROC-AUC** measures ranking quality across thresholds. In credit risk, ranking is extremely important because lenders may choose different cutoffs depending on market conditions, risk appetite, capital constraints, or product type. XGBoost is final because it ranks applicants best by ROC-AUC.

## 7. Threshold Tuning

Threshold logic appears in:

- `src/threshold_tuning.py`: sweeps thresholds from 0.10 to 0.90 in 0.05 increments.
- `app.py`: sidebar slider defaults to recommended business threshold and applies decisions.
- `api.py`: request schema default threshold is 0.55, constrained between 0.1 and 0.9.

**Default threshold.**

- Model classifiers normally default to 0.50 when calling `pipeline.predict`.
- The app/API business default is 0.55.

**Selected threshold.**

- Best F1 threshold: 0.70.
- Business threshold: 0.55.

**Threshold results from `reports/home_credit_threshold_summary.json`:**

- At threshold 0.70: accuracy 0.8808, precision 0.3027, recall 0.3660, F1 0.3313.
- At threshold 0.55: accuracy 0.7781, precision 0.2076, recall 0.6205, F1 0.3111.

**Business tradeoff.**

- Lower threshold: catches more defaulters, improves recall, but creates more false positives and more manual reviews.
- Higher threshold: improves precision and approval confidence, but misses more defaulters.
- The project chooses 0.55 as a recall-friendly business threshold because missed defaults are expensive in lending.

## 8. Fairness Analysis

Fairness code is in `src/fairness_analysis.py`; display is in the Monitoring tab in `app.py`.

**Protected or sensitive groups used.**

- `CODE_GENDER`
- `NAME_EDUCATION_TYPE`
- `NAME_INCOME_TYPE`
- `NAME_FAMILY_STATUS`

**Metrics computed.**

- `applicants`: group sample size.
- `avg_default_probability`: average model score by group.
- `high_risk_rate`: share of group with score above threshold.
- `observed_default_rate`: actual default rate in demo sample.

**Important results from `reports/home_credit_fairness_summary.csv`:**

- Gender: M high-risk rate 0.3278 vs F 0.2177.
- Education: Lower secondary high-risk rate 0.3514; Higher education 0.1435.
- Family status: Single/not married high-risk rate 0.3425; Widow 0.1654.
- Income type: Working high-risk rate 0.3186; Pensioner 0.1512.

**Business importance.** Fairness analysis helps detect whether model decisions disproportionately send certain groups to high-risk/manual review. That affects customer access, reputation, and portfolio strategy.

**Regulatory importance.** Credit decisioning is regulated. A real system would need legally reviewed protected-class definitions, adverse-action reason codes, fair-lending testing, monitoring, human oversight, and documentation. The repository correctly labels this as a snapshot, not a regulatory audit.

## 9. FastAPI Analysis

FastAPI app: `api.py`.

**Request model.**

```python
class ApplicantBatch(BaseModel):
    applicants: list[dict[str, Any]] = Field(..., min_length=1)
    threshold: float = Field(0.55, ge=0.1, le=0.9)
```

**Endpoint: `GET /health`.**

- Purpose: health check for deployment and monitoring.
- Request: no body.
- Response: `status`, `model`, `required_features`.
- Why needed: lets orchestration tools verify the service is alive and has loaded a model.

**Endpoint: `GET /schema`.**

- Purpose: expose required scoring fields and one sample applicant.
- Request: no body.
- Response: `required_features`, `sample_applicant`.
- Why needed: helps clients map internal loan-origination data to the model schema.

**Endpoint: `POST /score`.**

- Purpose: batch score applicants.
- Request: JSON with `applicants` and optional `threshold`.
- Response: `threshold`, `count`, and `results` containing available `SK_ID_CURR`, `default_probability`, `risk_band`, and `decision`.
- Decision logic: low if probability is below `min(0.25, threshold/2)`, high if probability is at least threshold, otherwise review.
- Why API is needed: enables integration with backend loan systems, batch workflows, or other apps without depending on Streamlit.

## 10. Streamlit Analysis

Streamlit app: `app.py`.

**App setup.** Uses `st.set_page_config` with wide layout and title `CreditRisk AI`.

**Model loading.** Reads model path from `reports/home_credit_model_comparison.json`. If the XGBoost artifact cannot be loaded in the cloud runtime, it trains a fallback logistic model from packaged demo applicants.

**Feature alignment.** `prepare_scoring_data` removes `TARGET`, fills missing required features with blank values, coerces numeric columns, and preserves categorical columns.

**Scoring.** `score_applicants` calls `predict_proba`, creates `default_probability`, assigns `risk_band`, and maps decisions.

**Dashboard tab.**

- Shows portfolio counts, high-risk rate, average risk score.
- Shows best model, ROC-AUC, training default rate.
- Shows active threshold, best F1 threshold, and business threshold.
- Lists highest-risk applicants.

**Batch Scoring tab.**

- Provides downloadable scoring template.
- Accepts uploaded CSV.
- Validates schema overlap.
- Scores applicants.
- Warns about missing features.
- Exports scored applicants and audit log.

**Applicant Review tab.**

- Lets user select `SK_ID_CURR`.
- Scores one applicant.
- Displays default probability, decision, known outcome, and applicant snapshot.

**Analytics tab.**

- Shows model comparison chart.
- Shows threshold tradeoff chart/table.
- Shows feature importance table and chart.

**Monitoring tab.**

- Shows score distribution.
- Shows decision mix.
- Shows drift watchlist summary.
- Shows fairness snapshot if generated.

**Product tab.**

- Explains use cases: pre-screening, portfolio monitoring, policy testing.
- States production boundaries.

## 11. Monitoring & Drift

Monitoring code is in `app.py`, especially `render_monitoring`.

**What is monitored.**

- Score distribution by probability bins.
- Decision/risk-band mix.
- Portfolio summary: applicant count, high-risk count, high-risk rate, average score.
- Drift watchlist features: `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `EXT_SOURCE_2`, `EXT_SOURCE_3`, `BUREAU_RECORD_COUNT`, `INST_DAYS_PAST_DUE_MEAN`.
- Fairness snapshot from generated CSV.

**How drift is detected.**

The current implementation does not compute statistical drift tests like PSI, KS test, or population stability over time. It computes current batch means and missing rates and labels them as values to track against future production batches. This is a monitoring prototype, not full automated drift detection.

**Why monitored.**

Credit risk distributions change with macroeconomic conditions, product mix, marketing channels, underwriting policies, and borrower behavior. Monitoring prevents silent model degradation.

## 12. Production Readiness

**Strengths.**

- End-to-end pipeline from data loading to UI/API.
- Saved sklearn/XGBoost pipelines include preprocessing and model together.
- Stratified split preserves class imbalance.
- Imbalance-aware model settings: `class_weight` and `scale_pos_weight`.
- Threshold tuning is business-aware.
- Audit-log download includes timestamp, model source, threshold, applicant ID, score, band, decision.
- FastAPI provides machine-consumable serving.
- Streamlit provides analyst workflow.
- Model card and deployment docs exist.

**Security gaps.**

- No authentication or authorization.
- No encryption/storage controls for uploaded applicant data.
- No API rate limiting.
- `joblib.load` is used; model artifacts must be trusted because pickle/joblib loading can execute unsafe objects.
- CORS is disabled in Streamlit config, but API security is not hardened.

**Scalability gaps.**

- Model loads globally in API, which is good.
- Streamlit is fine for demos but not ideal for high-throughput scoring.
- Batch scoring loads uploaded CSV in memory.
- No async scoring queue or database.
- No container/Dockerfile in repo.

**Logging gaps.**

- App supports downloadable audit logs, but no centralized server logging.
- API has no structured request logging.
- No model/version registry.

**Error handling.**

- Streamlit handles missing model artifacts, bad CSV upload, missing schema fields, and scoring exceptions.
- API does not wrap scoring errors with custom HTTP exceptions.
- Home Credit loaders provide explicit missing-file messages.

**Production upgrades before real lending.**

- Point-in-time feature store.
- AuthN/AuthZ.
- PII controls and encrypted storage.
- Model registry and immutable model versioning.
- SHAP/adverse-action explanations for final model.
- Calibration testing.
- Fair-lending/legal review.
- PSI/KS drift monitoring.
- CI tests and data contracts.
- Dockerized deployment.
- Human review workflow and approval audit trail.

## 13. Resume Defense Questions: 100 Interviewer Questions With Ideal Answers

1. **Q:** What problem does this project solve? **A:** It predicts loan default risk and helps lenders prioritize review and tune approval thresholds.
2. **Q:** Why is this classification? **A:** The label is binary: `TARGET = 1` default difficulty, `0` no difficulty.
3. **Q:** What is the positive class? **A:** `TARGET = 1`, repayment difficulty/default risk.
4. **Q:** What dataset is used? **A:** Kaggle Home Credit Default Risk.
5. **Q:** How many rows are in training? **A:** 307,511 in the generated model report.
6. **Q:** What is the default rate? **A:** About 8.1%.
7. **Q:** Why is imbalance important? **A:** Accuracy can be misleading and defaults are the costly minority class.
8. **Q:** What metric is primary? **A:** ROC-AUC for ranking quality, with recall/precision/F1 for threshold behavior.
9. **Q:** Why ROC-AUC in credit risk? **A:** Lending policy can choose different thresholds, so ranking matters.
10. **Q:** What was the best model? **A:** XGBoost by ROC-AUC, 0.781.
11. **Q:** Why use logistic regression? **A:** It is interpretable, fast, and a strong credit-scoring baseline.
12. **Q:** Why use random forest? **A:** To capture nonlinear interactions and compare a bagging tree model.
13. **Q:** Why use XGBoost? **A:** It performs strongly on tabular data and handled nonlinear signals best here.
14. **Q:** Why did random forest underperform? **A:** Its ROC-AUC was 0.707, likely weaker ranking and less effective boosting on sparse/imbalanced data.
15. **Q:** What is `scale_pos_weight`? **A:** XGBoost imbalance weight: negatives divided by positives in training.
16. **Q:** What is class weighting? **A:** It gives more importance to the minority default class during fitting.
17. **Q:** How is train/test split done? **A:** `train_test_split` with `test_size=0.2`, `random_state=42`, and `stratify=y`.
18. **Q:** Why stratify? **A:** To preserve the 8.1% default rate in both train and test sets.
19. **Q:** What files store models? **A:** `models/home_credit_baseline.joblib`, `models/home_credit_advanced.joblib`, `models/home_credit_xgboost.joblib`.
20. **Q:** What is the final artifact? **A:** `models/home_credit_xgboost.joblib`.
21. **Q:** What is in the artifact? **A:** A sklearn `Pipeline` containing preprocessing and classifier.
22. **Q:** Why save preprocessing with the model? **A:** It ensures training and inference transformations match.
23. **Q:** What preprocessing is used? **A:** Numeric median imputation and scaling; categorical most-frequent imputation and one-hot encoding.
24. **Q:** Why one-hot encode? **A:** Linear and tree sklearn pipelines need numeric inputs.
25. **Q:** Why median imputation? **A:** It is robust to outliers in financial data.
26. **Q:** Why scale numeric features? **A:** Logistic regression benefits from scaled inputs; it is included consistently in pipelines.
27. **Q:** Does XGBoost need scaling? **A:** Not usually, but scaling is harmless and keeps a common pipeline pattern.
28. **Q:** What is the threshold default in the API? **A:** 0.55.
29. **Q:** What is the classifier default threshold? **A:** 0.50 when using `.predict`.
30. **Q:** What threshold maximizes F1? **A:** 0.70.
31. **Q:** What business threshold is selected? **A:** 0.55.
32. **Q:** Why choose 0.55 instead of 0.70? **A:** It preserves more recall, catching more risky borrowers.
33. **Q:** What happens when threshold decreases? **A:** Recall rises, precision usually falls.
34. **Q:** What happens when threshold increases? **A:** Precision rises, recall falls.
35. **Q:** At 0.55, what are precision and recall? **A:** Precision 0.2076 and recall 0.6205.
36. **Q:** At 0.70, what are precision and recall? **A:** Precision 0.3027 and recall 0.3660.
37. **Q:** What is a false negative in this project? **A:** A true defaulter predicted as safe/approved.
38. **Q:** Why are false negatives expensive? **A:** The bank may approve a borrower who defaults.
39. **Q:** What is a false positive? **A:** A good borrower flagged as risky.
40. **Q:** Why are false positives costly? **A:** They can reduce approvals, revenue, and customer satisfaction.
41. **Q:** What are the strongest features? **A:** `EXT_SOURCE_3`, `EXT_SOURCE_2`, education type, gender, `EXT_SOURCE_1`.
42. **Q:** Why are external scores important? **A:** They summarize third-party risk signals and rank borrowers well.
43. **Q:** What is a leakage risk with historical tables? **A:** Joining records created after application time.
44. **Q:** How would you fix temporal leakage? **A:** Use point-in-time joins with application timestamps and feature cutoffs.
45. **Q:** Is `SK_ID_CURR` used as a feature? **A:** No, it is dropped.
46. **Q:** Is `TARGET` used as a feature? **A:** No, it is separated as the label.
47. **Q:** What previous-application features are created? **A:** Aggregated amounts, down payments, decision days, payment counts, categorical status counts, and application count.
48. **Q:** What bureau features are created? **A:** Credit overdue, active/closed flags, debt/limit sums, bureau-balance DPD/writeoff rates, credit-type counts.
49. **Q:** What installment features are created? **A:** Payment gap, days past due, days early, payment sums, and count features.
50. **Q:** What POS cash features are created? **A:** DPD flags/rates, installment statistics, status counts.
51. **Q:** What credit-card features are created? **A:** Utilization, balances, limits, drawings, payments, receivables, DPD statistics.
52. **Q:** What does `CC_UTILIZATION` mean? **A:** Balance divided by credit limit.
53. **Q:** Why is utilization predictive? **A:** High utilization can indicate financial stress.
54. **Q:** What does `INST_DAYS_PAST_DUE_MEAN` mean? **A:** Average number of late-payment days across installments.
55. **Q:** Why use aggregation? **A:** The model needs one row per applicant, while raw data is relational.
56. **Q:** What is the Streamlit app for? **A:** Analyst-facing scoring, threshold tuning, monitoring, and downloads.
57. **Q:** What is the FastAPI app for? **A:** Programmatic scoring integration.
58. **Q:** What does `/health` return? **A:** Status, model name, and required feature count.
59. **Q:** What does `/schema` return? **A:** Required features and a sample applicant.
60. **Q:** What does `/score` return? **A:** Probability, risk band, and decision for each applicant.
61. **Q:** What is the request schema for `/score`? **A:** `applicants: list[dict]` plus `threshold` between 0.1 and 0.9.
62. **Q:** What are risk bands? **A:** Low, Review, High based on probability relative to threshold.
63. **Q:** How is Low assigned? **A:** Probability below `min(0.25, threshold/2)`.
64. **Q:** How is High assigned? **A:** Probability greater than or equal to threshold.
65. **Q:** What is the Batch Scoring page? **A:** CSV upload, score generation, table preview, scored CSV download, audit log download.
66. **Q:** What is the Applicant Review page? **A:** Single-applicant probability and decision review.
67. **Q:** What is the Analytics page? **A:** Model comparison, threshold tradeoff, feature importance.
68. **Q:** What is the Monitoring page? **A:** Score distribution, decision mix, drift watchlist, fairness snapshot.
69. **Q:** What is monitored for drift? **A:** Means and missing rates of selected financial/risk features.
70. **Q:** Is drift detection fully automated? **A:** No; it is a lightweight watchlist, not PSI/KS-based drift detection.
71. **Q:** What fairness groups are analyzed? **A:** Gender, education type, income type, family status.
72. **Q:** What fairness metrics are computed? **A:** Average score, high-risk rate, observed default rate, applicant count.
73. **Q:** Is this a regulatory fairness audit? **A:** No, the code explicitly calls it a portfolio snapshot.
74. **Q:** What fairness issue appears in the snapshot? **A:** Male applicants have higher high-risk rate than female applicants in the demo sample.
75. **Q:** Why is gender sensitive? **A:** It can be protected or prohibited in real credit decisions depending on jurisdiction.
76. **Q:** Would you deploy with gender as a feature? **A:** Not without legal/compliance review; likely exclude or constrain it.
77. **Q:** What is model explainability here? **A:** XGBoost feature importance in scripts and SHAP notebook for the older logistic path.
78. **Q:** What is missing for production explainability? **A:** Per-applicant reason codes for the final XGBoost model.
79. **Q:** What is the model card for? **A:** It documents intended use, data, metrics, limitations, and talking points.
80. **Q:** What are the biggest production gaps? **A:** Auth, PII security, point-in-time joins, automated drift, legal fairness review, centralized logging.
81. **Q:** What is the cloud fallback model? **A:** A logistic regression trained from packaged demo applicants if XGBoost unpickling fails.
82. **Q:** Why might XGBoost fail to load? **A:** Python/package version mismatch in cloud runtime.
83. **Q:** What does `requirements.txt` pin? **A:** pandas, numpy, sklearn, joblib, Streamlit, matplotlib, seaborn, xgboost, FastAPI, uvicorn.
84. **Q:** What runtime is recommended? **A:** Python 3.11.
85. **Q:** What does the audit log contain? **A:** Timestamp, model source, threshold, applicant ID, probability, risk band, decision.
86. **Q:** Why is audit logging important? **A:** Governance, traceability, review, and compliance.
87. **Q:** How does the app handle missing upload features? **A:** Adds missing model features as blank values and warns the user.
88. **Q:** What is risky about filling missing upload fields? **A:** Scores can be unreliable if too many engineered features are absent.
89. **Q:** How would you improve schema validation? **A:** Add strict data contracts, required fields, type checks, and rejection rules.
90. **Q:** How would you improve calibration? **A:** Use calibration curves, Brier score, Platt/isotonic calibration, and out-of-time validation.
91. **Q:** How would you improve evaluation? **A:** Add PR-AUC, cost curves, calibration, reject-rate analysis, and segment metrics.
92. **Q:** How would you improve deployment? **A:** Dockerize, add CI/CD, health checks, logging, monitoring, and model registry.
93. **Q:** How would you retrain? **A:** Schedule data refresh, rebuild features point-in-time, validate metrics, register model, deploy after approval.
94. **Q:** How would you monitor fairness over time? **A:** Track group-level approval/high-risk rates, error rates, adverse impact, and drift.
95. **Q:** Why not rely only on accuracy? **A:** With 8.1% positives, a trivial non-default model can look highly accurate.
96. **Q:** What is F1 useful for? **A:** It balances precision and recall at a chosen threshold.
97. **Q:** Why is precision low? **A:** Defaults are rare; many high-risk flags are false positives in imbalanced data.
98. **Q:** Does low precision make the model useless? **A:** No; it can still rank risk well and support manual review if recall and ROC-AUC are useful.
99. **Q:** What would you say this project proves? **A:** End-to-end ML delivery: data engineering, modeling, threshold policy, explainability, fairness, UI, API.
100. **Q:** What would you not claim? **A:** That it is ready for real automated lending decisions without compliance, security, fairness, and point-in-time validation.

## 14. Viva Mode: 25 Most Likely Questions From This Repository

1. **Explain the full project in two minutes.** It is an end-to-end Home Credit default-risk system: load raw relational credit data, aggregate borrower-level features, train logistic/random-forest/XGBoost classifiers, select XGBoost by ROC-AUC, tune thresholds, serve scoring through Streamlit and FastAPI, and provide monitoring/fairness/reporting artifacts.
2. **Why is the default rate important?** Only 8.1% default, so the model must handle imbalance and accuracy alone is not enough.
3. **Why did you choose XGBoost as final?** It achieved the best ROC-AUC: 0.781 vs 0.755 logistic and 0.707 random forest.
4. **Why is ROC-AUC more important than accuracy?** ROC-AUC measures ranking across thresholds, which matters when lending policy changes.
5. **Why is recall important?** Recall catches actual defaulters; missed defaulters are financially expensive.
6. **Why is precision still important?** Low precision creates unnecessary manual reviews or rejected good customers.
7. **What threshold did you choose and why?** Business threshold 0.55 because it keeps recall at 0.6205, while best-F1 threshold 0.70 has much lower recall at 0.3660.
8. **What are the top predictive features?** `EXT_SOURCE_3`, `EXT_SOURCE_2`, education, gender, `EXT_SOURCE_1`, bureau closure signals, and installment delinquency.
9. **What is the biggest leakage risk?** Historical tables may include records after the current application unless point-in-time cutoffs are enforced.
10. **How did you aggregate relational data?** Grouped each child table by `SK_ID_CURR` and computed means, maxes, mins, sums, rates, and category counts.
11. **What does the API do?** It exposes `/health`, `/schema`, and `/score` for programmatic scoring.
12. **What does the Streamlit app do?** It provides dashboard, batch scoring, applicant review, analytics, monitoring, and product notes.
13. **What is the fairness analysis?** A group-level snapshot of average score, high-risk rate, and observed default rate by gender, education, income type, and family status.
14. **Is the fairness analysis enough for production?** No; it is a portfolio snapshot, not a legally sufficient fair-lending audit.
15. **What monitoring is implemented?** Score distribution, decision mix, selected feature means/missing rates, and fairness table display.
16. **Is true drift detection implemented?** Not yet; it needs baseline comparison with PSI, KS, or other drift tests.
17. **How do you handle missing values?** Numeric median imputation; categorical most-frequent imputation in the main pipelines.
18. **How do you handle categorical variables?** One-hot encoding with `handle_unknown="ignore"`.
19. **How do you handle uploaded CSVs with missing fields?** The app fills missing required model features with blank values and warns the user.
20. **What is the difference between the simple and Home Credit pipelines?** The simple pipeline uses synthetic `loan_data.csv`; the Home Credit pipeline is the main multi-table production-style implementation.
21. **Why include FastAPI if Streamlit exists?** Streamlit is for analysts; FastAPI is for backend integration and automated systems.
22. **What are the main production gaps?** Authentication, PII controls, point-in-time feature engineering, logging, automated drift/fairness monitoring, CI/CD, and legal review.
23. **How would you make it production-ready?** Add feature store, model registry, Docker, CI, tests, monitoring, auth, encryption, reason codes, and compliance workflow.
24. **What is your strongest technical contribution?** Multi-table feature engineering plus business threshold tuning and deployable UI/API.
25. **What limitation would you openly disclose?** It is a portfolio decision-support prototype and should not be used for real lending decisions without governance and validation.

