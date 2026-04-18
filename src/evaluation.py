import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a classification model and return metrics.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "log_loss": log_loss(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return results


def print_evaluation(results, model_name, threshold):
    """
    Print evaluation results neatly.
    """
    print(f"\n===== {model_name} (threshold={threshold}) =====")

    print("Confusion Matrix:")
    print(results["confusion_matrix"])

    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")
    print(f"ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"Log Loss:  {results['log_loss']:.4f}")


def plot_conf_matrix(results, title):
    """
    Plot confusion matrix.
    """
    cm = results["confusion_matrix"]

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()