import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from train_model import train_model2
import train_model

def evaluate_model(
    model_path: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Load a saved CatBoost model and compute classification metrics.

    Parameters
    ----------
    model_path : str
        Path to the saved .pkl model file.
    X_test : pd.DataFrame
        Held-out feature matrix.
    y_test : pd.Series
        True binary labels.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc
    """
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    # Get probability scores if available, fall back to decision_function
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }

    print(f"\n{type(model).__name__} Evaluation")
    print("-" * 30)
    for name, value in metrics.items():
        display = f"{value:.4f}" if value is not None else "N/A"
        print(f"{name.capitalize():<12} {display}")

    return metrics



import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay)
import matplotlib.pyplot as plt


def evaluate_cervical_cancer_model(model_path, X_test_path, y_test_path):
    """
    Évalue le modèle sur les métriques exigées par le projet :
    ROC-AUC, accuracy, precision, recall, et F1-score.
    
    Args:
        model_path (str): Chemin vers le modèle sauvegardé (.pkl).
        X_test (pd.DataFrame): Features de test.
        y_test (pd.Series): Cibles de test.
    """
    
    # Charge les fichiers en utilisant les chemins passés en arguments
    model = joblib.load(model_path)
    X_test = joblib.load(X_test_path)
    y_test = joblib.load(y_test_path)
    
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Nécessaire pour le ROC-AUC
    
    # Calcul des métriques exigées 
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }
    
    print("--- RÉSULTATS DE L'ÉVALUATION ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n--- RAPPORT DÉTAILLÉ ---")
    print(classification_report(y_test, y_pred))

    predictions = model.predict(X_test)
    # On crée la matrice de confusion
    mc = confusion_matrix(y_test, predictions)

    # On l'affiche joliment
    disp = ConfusionMatrixDisplay(confusion_matrix=mc, display_labels=['Sain', 'Risque'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice de Confusion - Cancer du Col")
    plt.show()
    
    return metrics

if __name__ == "__main__":
    evaluate_cervical_cancer_model('model_rf.pkl', 'data/X_test.pkl', 'data/y_test.pkl')


