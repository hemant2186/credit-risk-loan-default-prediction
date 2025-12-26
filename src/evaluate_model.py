from sklearn.metrics import roc_auc_score

def evaluate(model, X_val, y_val):
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    return auc
