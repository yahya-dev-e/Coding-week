import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from src.data_processing import load_and_clean_data, preprocess_data

def train_model_catboost():
    df = load_and_clean_data('data/risk_factors_cervical_cancer.csv')
    X_train, X_test, y_train, y_test, imputer, scaler, cols = preprocess_data(df)
    
    model = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6, 
        loss_function="Logloss", verbose=100, random_seed=42
    )
    model.fit(X_train, y_train)
    
    # Création du dossier s'il n'existe pas
    os.makedirs('models', exist_ok=True)
    
    # Sauvegarde dans le dossier models/
    model_path = 'models/catboost_model.joblib'
    joblib.dump(model, model_path)
    joblib.dump({'imputer': imputer, 'scaler': scaler, 'columns': list(cols)}, 'models/catboost_assets.joblib')
    
    print(f"✅ Modèle CatBoost sauvegardé dans '{model_path}'.")
    return model_path, pd.DataFrame(X_test, columns=cols), y_test

def train_model_XGBoost():
    df = load_and_clean_data('data/risk_factors_cervical_cancer.csv')
    X_train, X_test, y_train, y_test, imputer, scaler, cols = preprocess_data(df)

    model = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, 
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42
    )
    model.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/xgboost_model.joblib'
    joblib.dump(model, model_path)
    joblib.dump({'imputer': imputer, 'scaler': scaler, 'columns': list(cols)}, 'models/xgboost_assets.joblib')

    print(f"✅ Modèle XGBoost sauvegardé dans '{model_path}'.")
    return model_path, pd.DataFrame(X_test, columns=cols), y_test

def train_model_svm():
    df = load_and_clean_data('data/risk_factors_cervical_cancer.csv')
    X_train, X_test, y_train, y_test, imputer, scaler, cols = preprocess_data(df)    
    
    model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/svm_model.joblib'
    joblib.dump(model, model_path)
    joblib.dump({'imputer': imputer, 'scaler': scaler, 'columns': list(cols)}, 'models/svm_assets.joblib')
    
    print(f"✅ Modèle SVM sauvegardé dans '{model_path}'.")
    return model_path, pd.DataFrame(X_test, columns=cols), y_test

def train_model_Randomforest():
    df = load_and_clean_data('data/risk_factors_cervical_cancer.csv')
    X_train, X_test, y_train, y_test, imputer, scaler, cols = preprocess_data(df)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/Random_forest_model.joblib'
    joblib.dump(model, model_path)
    joblib.dump({'imputer': imputer, 'scaler': scaler, 'columns': list(cols)}, 'models/Random_forest_assets.joblib')

    print(f"✅ Modèle Random Forest sauvegardé dans '{model_path}'.")
    return model_path, pd.DataFrame(X_test, columns=cols), y_test