<<<<<<< Updated upstream
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from data_processing import load_and_clean_data, remove_outliers_iqr, optimize_memory, preprocess_data, supprimer_colonnes_zero
=======
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
>>>>>>> Stashed changes

def evaluate():
    # 1. Préparation des données de test
    df = load_and_clean_data('data/risk_factors_cervical_cancer.csv')
    _, X_test_scaled, _, y_test, _, _, _ = preprocess_data(df)    
    # 2. Chargement avec PICKLE
    try:
        with open('trained_svm_assets.pkl', 'rb') as f:
            assets = pickle.load(f)
        model = assets['model']
    except FileNotFoundError:
        print("❌ Erreur : Fichier pkl non trouvé. Lance d'abord train_model.py")
        return
    
    # 3. Prédictions et Scores
    predictions = model.predict(X_test_scaled)
    print("\n📊 MATRICE DE CONFUSION :")
    print(confusion_matrix(y_test, predictions))
    print("\n📋 RAPPORT DE CLASSIFICATION :")
    print(classification_report(y_test, predictions))

<<<<<<< Updated upstream
=======
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

>>>>>>> Stashed changes
