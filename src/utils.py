def calculate_business_loss(y_true, y_pred, fn_cost=50000, fp_cost=5000):
    loss = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 0:
            loss += fn_cost
        elif true == 0 and pred == 1:
            loss += fp_cost
    return loss
