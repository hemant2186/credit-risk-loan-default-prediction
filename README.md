# Credit Risk / Loan Default Prediction

## Overview
This project implements an end-to-end machine learning system to predict
loan default risk using real-world banking data.

The system is designed to support financial institutions in making
data-driven loan approval decisions while minimizing business risk.

---

## Business Problem
Loan defaults cause significant financial losses.
The challenge is to:
- Identify high-risk applicants early
- Reduce false approvals
- Balance revenue growth with risk management

---

## Dataset
**Home Credit Default Risk (Kaggle)**

- 300,000+ loan applicants
- Financial, demographic, and credit history features
- Highly imbalanced target variable

Target:
- `1` → Loan Default  
- `0` → Loan Repaid  

---

## Approach
1. Data understanding and cleaning  
2. Business-driven feature engineering  
3. Cost-sensitive model training  
4. Threshold optimization based on financial loss  
5. Model explainability using SHAP  
6. Deployment using Streamlit  

---

## Models Used
- Logistic Regression (interpretable baseline)
- Random Forest (non-linear benchmark)

**Primary Metric:** ROC-AUC  
**Decision Metric:** Expected Financial Loss  

---

## Explainability
- Global feature importance using SHAP
- Individual prediction explanations
- Decision-level risk factors for transparency

---

## Deployment
A Streamlit web application enables:
- Applicant data input
- Default probability prediction
- Risk category recommendation:
  - Low Risk → Approve
  - Medium Risk → Manual Review
  - High Risk → Reject

---

## Key Learnings
- Feature engineering has a greater impact than complex models
- Business-cost optimization is critical in real-world ML systems
- Explainability and fairness are essential in financial applications

---

## Limitations
- Model is trained on historical data
- Performance may degrade due to data drift
- Designed to assist, not replace, human decision-making

---

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, SHAP, Streamlit

---

## Author
**Hemant Kumar**  
B.Tech Student | Aspiring Data Analyst  
📧 hk6227084@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/hemant-kumar-171472210
