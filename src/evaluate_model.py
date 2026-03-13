
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import src.train_model as train_model

def evaluate_model(
    model_path: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """

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

from src.train_model import (
    train_model_svm, 
    train_model_XGBoost, 
    train_model_catboost, 
    train_model_Randomforest
)

def run_all_evaluations():
    """
    Lance l'entraînement et l'évaluation pour chaque modèle 
    en utilisant la fonction evaluate_model fournie.
    """
    
    # 1. Liste des modèles à tester
    model_trainers = {
        "SVM": train_model_svm,
        "XGBoost": train_model_XGBoost,
        "CatBoost": train_model_catboost,
        "Random Forest": train_model_Randomforest
    }

    results = []

    print("--- DÉBUT DE L'ÉVALUATION GÉNÉRALE ---")

    for name, train_func in model_trainers.items():
        try:
            # L'entraînement retourne : model_path, X_test (DataFrame), y_test
            path, X_test, y_test = train_func()
            
            # APPEL DE TA FONCTION evaluate_model
            metrics = evaluate_model(path, X_test, y_test)
            
            # Stockage pour comparaison
            metrics['Model'] = name
            results.append(metrics)
            
        except Exception as e:
            print(f"❌ Erreur lors de l'évaluation de {name}: {e}")

    # 2. Affichage d'un tableau comparatif final
    if results:
        df_results = pd.DataFrame(results)
        # On remet 'Model' en première colonne
        cols = ['Model'] + [c for c in df_results.columns if c != 'Model']
        print("\n" + "="*50)
        print("📊 RÉSUMÉ DES PERFORMANCES")
        print("="*50)
        print(df_results[cols].to_string(index=False))

if __name__ == "__main__":
    run_all_evaluations()