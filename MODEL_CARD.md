# Model Card: Home Credit Default Risk

## Intended Use

This model estimates the probability that a loan applicant will default. It is built as a portfolio project for data science and machine learning internships, not as a production lending decision system.

## Dataset

- Source: Kaggle Home Credit Default Risk competition data
- Training rows: 307,511
- Engineered columns: 324
- Positive class: `TARGET = 1`, applicant had repayment difficulty
- Class balance: about 8.1% positive class

## Modeling Approach

The project builds borrower-level features from multiple related tables:

- application data
- previous applications
- bureau records
- bureau monthly history
- installment payments
- POS cash balance
- credit card balance

The final comparison includes Logistic Regression, Random Forest, and XGBoost. XGBoost is currently the strongest model by ROC-AUC.

## Current Performance

| Model | ROC-AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.755 | 0.183 | 0.615 | 0.282 |
| Random Forest | 0.707 | 0.140 | 0.654 | 0.231 |
| XGBoost | 0.781 | 0.185 | 0.690 | 0.292 |

Threshold tuning improves how the model is used for business decisions:

- Best F1 threshold: 0.70
- Recall-friendly business threshold: 0.55

## Explainability

Run this command to generate a feature importance table and chart:

```bash
python -m src.explain_model
```

Outputs:

- `reports/home_credit_feature_importance.csv`
- `reports/home_credit_feature_importance.png`

## Limitations

- The model is trained on historical competition data and should not be used for real lending decisions.
- Credit models can encode socioeconomic bias. A real deployment would need fairness testing, regulatory review, monitoring, and human oversight.
- ROC-AUC is useful for ranking risk, but lending policy depends on costs, approval rates, risk appetite, and compliance constraints.
- Feature importance is directional for portfolio explainability; it is not a causal explanation.

## Internship Talking Points

- Built a multi-table ML pipeline instead of a single flat CSV classifier.
- Evaluated models with imbalance-aware metrics.
- Tuned decision thresholds for realistic lending tradeoffs.
- Created a Streamlit app and reporting artifacts for stakeholder communication.
