from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

def train_model(data_path):
    df = pd.read_csv(data_path)

    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, "../models/credit_risk_model.pkl")
    joblib.dump(scaler, "../models/scaler.pkl")
    joblib.dump(X_train.columns, "../models/feature_columns.pkl")

    return model
