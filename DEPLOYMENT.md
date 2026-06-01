# Deployment Guide

This project is packaged as a Streamlit SaaS-style product called **CreditRisk AI**.

## Recommended: Streamlit Community Cloud

Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy

1. Push the repository to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app from this repository:
   - Repository: `hemant2186/credit-risk-loan-default-prediction`
   - Branch: `main`
   - Main file path: `app.py`
4. In Advanced settings, select Python `3.11` when available.
5. Deploy.

If the app runs on Python `3.14`, the saved XGBoost artifact may not unpickle cleanly. The app includes a cloud-safe fallback model so users can still score applicants, but Python `3.11` is recommended for the trained XGBoost pipeline.

The app uses:

- `requirements.txt` for Python dependencies
- `runtime.txt` for Python version
- `.streamlit/config.toml` for Streamlit settings

## Alternative: Render

Use the included `Procfile`.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

## Product Workflow

Users can:

- upload borrower/application CSV files
- score default probability in batch
- tune decision thresholds
- download scored applicants
- inspect model performance and feature importance

## Production Notes

This is a decision-support product prototype. Before real lending use, add:

- user authentication
- encrypted storage
- audit logs
- fairness and bias testing
- model monitoring
- compliance review
