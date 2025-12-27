# 💳 Credit Risk / Loan Default Prediction

## 📌 Project Overview
This project is an **end-to-end machine learning system** that predicts the
**probability of loan default** using real-world banking data.

The solution is designed to **support financial institutions** in making
**data-driven credit decisions** by balancing **risk management**, 
**business growth**, and **regulatory transparency**.

The project covers the **entire ML lifecycle** — from data preprocessing and
feature engineering to **model explainability and real-time deployment** using
a Streamlit web application.

---

## 🏦 Business Problem
Loan default is one of the most critical risks faced by banks and fintech
companies. Poor credit decisions can lead to:

- Significant financial losses due to defaults  
- Missed revenue by rejecting low-risk applicants  
- Regulatory and trust issues caused by opaque models  

### 🎯 Objective
Build a predictive system that:
- Identifies **high-risk loan applicants** early  
- Reduces **costly false approvals**  
- Supports **fair, transparent, and explainable** decision-making  

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
The project follows a **production-aware machine learning workflow**:

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

### 📐 Evaluation Strategy
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
A **Streamlit web application** enables real-time predictions by allowing users
to:

- Enter applicant information  
- View loan default probability  
- Receive risk-based recommendations:
  - 🟢 Low Risk → Approve  
  - 🟡 Medium Risk → Manual Review  
  - 🔴 High Risk → Reject  

🔗 **Live Demo:**  
https://credit-risk-loan-default-prediction.onrender.com

---

## 📦 Model & Data Handling
- Trained model artifacts are managed using **Git LFS**
- Datasets are excluded from the repository due to GitHub size limits  

### To reproduce results:
1. Download the dataset from Kaggle  
2. Place raw files inside `data/raw/`  
3. Run preprocessing and training scripts  

---

## 📈 Key Learnings
- Feature engineering often has a larger impact than model complexity  
- Business-aware metrics outperform generic accuracy-based evaluation  
- Explainability and fairness are essential in financial ML systems  
- ML models should **support**, not replace, human decision-making  

---

## ⚠️ Limitations & Future Scope
- Model is trained on historical data and may face data drift  
- Requires periodic monitoring and retraining  
- Future improvements may include:
  - Drift detection mechanisms  
  - Advanced cost-sensitive learning  
  - Fairness audits across demographic groups  

---

## 🧰 Tech Stack
- **Programming:** Python  
- **Data Processing:** Pandas, NumPy  
- **Modeling:** Scikit-learn (Logistic Regression, Random Forest)  
- **Explainability:** SHAP  
- **Deployment:** Streamlit  

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
