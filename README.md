# 💳 Credit Risk / Loan Default Prediction

## 📌 Project Overview
This project presents an **end-to-end machine learning solution** for predicting
loan default risk using real-world banking data.

The system is designed to **assist financial institutions** in making
**data-driven loan approval decisions** by balancing **risk control** and
**business growth**, while maintaining **transparency and explainability**.

---

## 🏦 Business Problem
Loan default is one of the primary risks faced by banks and fintech companies.
Incorrect credit decisions can result in:

- Financial losses due to loan defaults  
- Missed revenue by rejecting low-risk applicants  
- Reduced trust due to opaque decision-making  

### Objective
Build a predictive system that:
- Identifies high-risk applicants early  
- Minimizes costly false approvals  
- Supports fair and explainable decision-making  

---

## 📊 Dataset
**Home Credit Default Risk (Kaggle)**

- 300,000+ loan applications  
- Financial, demographic, and credit history features  
- Highly imbalanced target variable (real-world scenario)  

### Target Variable
- `1` → Loan Default  
- `0` → Loan Repaid  

---

## 🛠️ Methodology & Workflow
The project follows a **production-aware ML workflow**:

1. Data understanding and cleaning  
2. Business-driven feature engineering  
3. Handling class imbalance  
4. Cost-sensitive model training  
5. Threshold optimization based on financial loss  
6. Model explainability using SHAP  
7. Interactive deployment using Streamlit  

---

## 🤖 Models Implemented
- **Logistic Regression** — interpretable baseline model  
- **Random Forest** — captures non-linear relationships  

### Evaluation Strategy
- **Primary Metric:** ROC-AUC (robust to class imbalance)  
- **Decision Metric:** Expected Financial Loss (business-focused)  

---

## 🔍 Model Explainability
Explainability is a core focus of this project:

- Global feature importance using SHAP  
- Individual-level prediction explanations  
- Decision-level risk factors for rejected applications  

This ensures transparency and aligns with real-world regulatory requirements.

---

## 🚀 Deployment
A **Streamlit web application** enables real-time predictions by allowing users to:

- Enter applicant information  
- View loan default probability  
- Receive risk-based recommendations:
  - 🟢 Low Risk → Approve  
  - 🟡 Medium Risk → Manual Review  
  - 🔴 High Risk → Reject  

---

## 📈 Key Learnings
- Feature engineering often has a larger impact than model complexity  
- Business-aware metrics outperform generic accuracy-based evaluation  
- Explainability and fairness are essential in financial ML systems  
- ML models should support—not replace—human decision-making  

---

## ⚠️ Limitations & Future Scope
- Model is trained on historical data and may face data drift  
- Performance requires periodic monitoring and retraining  
- Future improvements may include:
  - Drift detection mechanisms  
  - Advanced cost-sensitive learning  
  - Fairness audits across demographic groups  

---

## 🧰 Tech Stack
- **Programming:** Python  
- **Data Processing:** Pandas, NumPy  
- **Modeling:** Scikit-learn  
- **Explainability:** SHAP  
- **Deployment:** Streamlit  

---

## 🔗 Live Demo
https://credit-risk-loan-default-prediction.onrender.com

---

## 📂 Project Structure
- notebooks/ → EDA & experimentation
- src/ → reusable ML logic
- models/ → trained artifacts
- app.py → Streamlit deployment

---

## 👤 Author
**Hemant Kumar**  
B.Tech Student | Aspiring Data Analyst  
📧 hk6227084@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/hemant-kumar-171472210
