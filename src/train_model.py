import pickle
import pandas as pd
<<<<<<< HEAD
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


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

=======
from sklearn.svm import SVC
from data_processing import load_and_clean_data, remove_outliers_iqr, optimize_memory, preprocess_data, supprimer_colonnes_zero

def train_model():
    # 1. Pipeline de données
    df = load_and_clean_data('data/risk_factors_cervical_cancer.csv')
    X_train, X_test, y_train, y_test, imputer, scaler, cols = preprocess_data(df)    
    # 2. Entraînement SVM
    model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. Sauvegarde avec PICKLE (Standard Python)
    # On enregistre tout dans un seul dictionnaire pour faire propre
    data_to_save = {
        'model': model,
        'imputer': imputer,
        'scaler': scaler,
        'columns': list(cols)
    }
    
    with open('trained_svm_assets.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
        
    print("✅ Modèle et outils sauvegardés dans 'trained_svm_assets.pkl' (à la racine).")
>>>>>>> 29fc9756240f669fbe46879f415b9ff56b90c3d1

