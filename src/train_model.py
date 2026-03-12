
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from data_processing import load_and_clean_data, remove_outliers_iqr, preprocess_data
def train_model_XGBoost():
    # 1. Pipeline de données
    df = load_and_clean_data('data/risk_factors_cervical_cancer.csv')
    X_train, X_test, y_train, y_test, imputer, scaler, cols = preprocess_data(df)  # ✅ 7 valeurs

    # 2. Entraînement XGBoost
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # 3. Sauvegarde avec JOBLIB
    model_path = 'xgboost_model.joblib'
    joblib.dump(model, model_path)

    assets = {'imputer': imputer, 'scaler': scaler, 'columns': list(cols)}
    joblib.dump(assets, 'xgboost_assets.joblib')

    print(f"✅ Modèle sauvegardé dans '{model_path}'.")



    # 4. X_test en DataFrame pour evaluate_model
    X_test_df = pd.DataFrame(X_test, columns=cols)

    return model_path, X_test_df, y_test



###Random-forest-model:

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_cervical_cancer_model_Random_forest(data_path, model_save_path='model_rf.pkl'):
    """
    Entraîne un RandomForestClassifier sur les données fournies.
    
    Args:
        data_path (str): Chemin vers le fichier CSV.
        model_save_path (str): Chemin pour sauvegarder le modèle entraîné.
        
    Returns:
        tuple: (model_entraine, model_save_path, X_test, y_test)
    """
    # 1. Chargement et nettoyage
    df = pd.read_csv(data_path)
    df = df.replace('?', np.nan)
    df = df.apply(pd.to_numeric)
    
    # Imputation par la moyenne
    df = df.fillna(df.mean())

    # 2. Séparation X et y
    # On retire les cibles potentielles pour isoler les features
    features_to_drop = ['Biopsy', 'Hinselmann', 'Schiller', 'Citology']
    X = df.drop(features_to_drop, axis=1)
    y = df['Biopsy']

    # 3. Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Sauvegarde locale du modèle
    joblib.dump(model, model_save_path)

    # --- AJOUT : Sauvegarde des données de test ---
    joblib.dump(X_test, 'data/X_test.pkl')
    joblib.dump(y_test, 'data/y_test.pkl')
    
    print(f"Modèle et données de test sauvegardés.")
    return model, model_save_path, X_test, y_test

