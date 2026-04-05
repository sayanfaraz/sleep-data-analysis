import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)  # for AUC

    mlflow.log_metrics({
        'accuracy': accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "auc_weighted": roc_auc_score(y_test, y_prob, multi_class='ovo', average='weighted'),
        "auc_macro": roc_auc_score(y_test, y_prob, multi_class='ovo', average='macro')
    }) # type: ignore