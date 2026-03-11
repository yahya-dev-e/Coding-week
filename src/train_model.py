import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
import joblib
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier


def preprocess_data(filepath):
    """Charge, nettoie et prépare les données."""

    # Chargement — les '?' sont des valeurs manquantes
    df = pd.read_csv(filepath, na_values='?')

    # Supprimer colonnes avec > 80% de valeurs manquantes
    df = df.loc[:, df.isnull().mean() < 0.8]

    # Convertir tout en numérique
    df = df.apply(pd.to_numeric, errors='coerce')

    TARGET = 'Biopsy'
    cols   = [c for c in df.columns if c != TARGET]

    X = df[cols]
    y = df[TARGET]

    # Split stratifié (important car dataset déséquilibré)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Imputation par la médiane
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=cols)
    X_test  = pd.DataFrame(imputer.transform(X_test),      columns=cols)

    # Scaling
    scaler  = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols)
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=cols)

    return X_train, X_test, y_train, y_test, imputer, scaler, cols



def train_model_catboost(
    data_path: str = "data/risk_factors_cervical_cancer.csv",
    model_output_path: str = "models/catboost_model.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
    target: str = "Biopsy",
    catboost_params: dict = None,
) -> dict:
    """
    Load data, train a CatBoost classifier, and save the model.

    Parameters
    ----------
    data_path : str
        Path to the raw CSV dataset.
    model_output_path : str
        Where to save the trained model (.pkl).
    test_size : float
        Fraction of data reserved for the held-out test set.
    random_state : int
        Seed for reproducibility.
    target : str
        Name of the binary target column.
    catboost_params : dict, optional
        Override default CatBoost hyperparameters.

    Returns
    -------
    dict with keys:
        "model"      – fitted CatBoostClassifier
        "X_test"     – held-out features (pd.DataFrame)
        "y_test"     – held-out labels  (pd.Series)
        "model_path" – absolute path of the saved model file
    """
    # ── 1. Load ──────────────────────────────────────────────────────────────
    df = pd.read_csv(data_path, na_values="?")

    X = df.drop(columns=[target])
    y = df[target]

    # Coerce to numeric (columns were forced to str by '?' values)
    X = X.apply(pd.to_numeric, errors="coerce")

    # ── 2. Impute ─────────────────────────────────────────────────────────────
    X = X.fillna(X.median())

    # ── 3. Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ── 4. SMOTE ──────────────────────────────────────────────────────────────
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # ── 5. Train ──────────────────────────────────────────────────────────────
    default_params = dict(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        verbose=100,
        random_seed=random_state,
    )
    if catboost_params:
        default_params.update(catboost_params)

    model = CatBoostClassifier(**default_params)
    model.fit(X_train, y_train)

    # ── 6. Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved → {os.path.abspath(model_output_path)}")

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "model_path": os.path.abspath(model_output_path),
    }


if __name__ == "__main__":
    train_model_catboost()

def train_model():
    print("--- Début du pipeline d'entraînement XGBoost ---")

    # ── 1. Prétraitement ──────────────────────────────────────────
    filepath = 'data/risk_factors_cervical_cancer.csv'
    X_train, X_test, y_train, y_test, imputer, scaler, cols = preprocess_data(filepath)
    print(f"Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")

    # ── 2. Calcul du scale_pos_weight (gestion du déséquilibre) ──
    neg   = (y_train == 0).sum()
    pos   = (y_train == 1).sum()
    spw   = neg / pos
    print(f"scale_pos_weight = {spw:.2f}")

    # ── 3. Entraînement ───────────────────────────────────────────
    print("Entraînement de XGBClassifier...")
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,   # ← corrige le déséquilibre
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # ── 4. Évaluation rapide ──────────────────────────────────────
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test)
    score  = model.score(X_test, y_test)
    print(f"Accuracy sur le jeu de test : {score:.4f}")
    print(classification_report(y_test, y_pred))

    # ── 5. Sauvegarde du package complet ─────────────────────────
    data_to_save = {
        'model':   model,
        'imputer': imputer,
        'scaler':  scaler,
        'columns': list(cols)
    }
    with open('trained_xgboost_assets.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)

    print("✅ Modèle et preprocessing sauvegardés dans 'trained_xgboost_assets.pkl'")

from sklearn.svm import SVC
from data_processing import load_and_clean_data, preprocess_data

def train_model_svm():
    # 1. Pipeline de données
    df = load_and_clean_data('data/risk_factors_cervical_cancer.csv')
    X_train, X_test, y_train, y_test, imputer, scaler, cols = preprocess_data(df)    
    
    # 2. Entraînement SVM
    model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. Sauvegarde avec JOBLIB (pour matcher avec evaluate_model)
    model_path = 'svm_model.joblib'
    joblib.dump(model, model_path)
    
    # On sauvegarde les autres outils séparément pour ton application Streamlit
    assets = {'imputer': imputer, 'scaler': scaler, 'columns': list(cols)}
    joblib.dump(assets, 'svm_assets.joblib')
    
    print(f"✅ Modèle sauvegardé dans '{model_path}'.")
    
    # 4. evaluate_model attend un DataFrame Pandas pour X_test
    X_test_df = pd.DataFrame(X_test, columns=cols)
    
    # On retourne les 3 arguments exacts requis par evaluate_model
    return model_path, X_test_df, y_test
