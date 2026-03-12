
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
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

#SVM model:


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



###Random-forest-model:

def train_model_Randomforest(data_path='data/risk_factors_cervical_cancer.csv', model_save_path='model_rf.pkl'):
    # 1. Chargement et nettoyage
    df = pd.read_csv(data_path)
    df = df.replace('?', np.nan)
    df = df.apply(pd.to_numeric)
    
    # Imputation par la moyenne
    df = df.fillna(df.mean())

    # 2. Séparation X et y
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
    print(f"✅ Modèle Random Forest sauvegardé dans '{model_save_path}'.")

    # On retourne UNIQUEMENT les 3 arguments attendus par evaluate_model
    return model_save_path, X_test, y_test

