<<<<<<< Updated upstream

<<<<<<< HEAD
=======
import pickle
import pandas as pd
<<<<<<< HEAD
import numpy as np
=======
import pandas as pd
import numpy as np
import joblib
>>>>>>> origin/main
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
<<<<<<< HEAD
import os
import joblib
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

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
=======
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from data_processing import load_and_clean_data, remove_outliers_iqr, preprocess_data
import os
from catboost import CatBoostClassifier





def train_model_catboost():
    # 1. Pipeline de données centralisé
    df = load_and_clean_data('data/risk_factors_cervical_cancer.csv')
    X_train, X_test, y_train, y_test, imputer, scaler, cols = preprocess_data(df)
    
    # 2. Entraînement CatBoost
    model = CatBoostClassifier(
>>>>>>> origin/main
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
<<<<<<< HEAD
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


def train_model2():
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
=======
        verbose=100, # Affiche la progression tous les 100 arbres
        random_seed=42
    )
    model.fit(X_train, y_train)
    
    # 3. Sauvegarde avec JOBLIB (pour matcher avec evaluate_model)
    model_path = 'catboost_model.joblib'
    joblib.dump(model, model_path)
    
    # Sauvegarde des outils de preprocessing pour l'interface Streamlit
    assets = {'imputer': imputer, 'scaler': scaler, 'columns': list(cols)}
    joblib.dump(assets, 'catboost_assets.joblib')
    
    print(f"✅ Modèle sauvegardé dans '{model_path}'.")
    
    # 4. Conversion de X_test en DataFrame Pandas pour evaluate_model
    X_test_df = pd.DataFrame(X_test, columns=cols)
    
    # Retour des 3 arguments exacts requis
    return model_path, X_test_df, y_test







def train_model_XGBoost():
    # 1. Pipeline de données
    df = load_and_clean_data('data/risk_factors_cervical_cancer.csv')
    X_train, X_test, y_train, y_test, imputer, scaler, cols = preprocess_data(df)  # ✅ 7 valeurs

    # 2. Entraînement XGBoost
>>>>>>> origin/main
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
<<<<<<< HEAD
        scale_pos_weight=spw,   # ← corrige le déséquilibre
=======
>>>>>>> origin/main
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

<<<<<<< HEAD
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

def train_model3():
    # 1. Pipeline de données
    df = load_and_clean_data('data/risk_factors_cervical_cancer.csv')
    X_train, X_test, y_train, y_test, imputer, scaler, cols = preprocess_data(df)    
=======
    # 3. Sauvegarde avec JOBLIB
    model_path = 'xgboost_model.joblib'
    joblib.dump(model, model_path)

    assets = {'imputer': imputer, 'scaler': scaler, 'columns': list(cols)}
    joblib.dump(assets, 'xgboost_assets.joblib')

    print(f"✅ Modèle sauvegardé dans '{model_path}'.")



    # 4. X_test en DataFrame pour evaluate_model
    X_test_df = pd.DataFrame(X_test, columns=cols)

    return model_path, X_test_df, y_test

#SVM model:


def train_model_svm():
    # 1. Pipeline de données
    df = load_and_clean_data('data/risk_factors_cervical_cancer.csv')
    X_train, X_test, y_train, y_test, imputer, scaler, cols = preprocess_data(df)    
    
>>>>>>> origin/main
    # 2. Entraînement SVM
    model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
<<<<<<< HEAD
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

if __name__ == "__main__":
    train_model()
>>>>>>> Stashed changes
=======
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



###Random-forest-model:

def train_model_Randomforest():
    # 1. Pipeline de données
    df = load_and_clean_data('data/risk_factors_cervical_cancer.csv')
    X_train, X_test, y_train, y_test, imputer, scaler, cols = preprocess_data(df)

    # 2. Entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 3. Sauvegarde locale du modèle
    model_path = 'Random_forest_model.joblib'
    joblib.dump(model, model_path)

    assets = {'imputer': imputer, 'scaler': scaler, 'columns': list(cols)}
    joblib.dump(assets, 'Random_forest_assets.joblib')

    print(f"✅ Modèle sauvegardé dans '{model_path}'.")

    # 4. evaluate_model attend un DataFrame Pandas pour X_test
    X_test_df = pd.DataFrame(X_test, columns=cols)
    
    # On retourne les 3 arguments exacts requis par evaluate_model
    return model_path, X_test_df, y_test

>>>>>>> origin/main
