
import os
import pandas as pd
import numpy as np
import joblib
from src.data_processing import load_and_clean_data, optimize_memory, preprocess_data
from src.train_model import train_model_Randomforest

def test_dataset_loading():
    df = load_and_clean_data("data/risk_factors_cervical_cancer.csv")
    assert df.shape[0] > 0
    assert '?' not in df.values # Vérifie que le nettoyage a fonctionné

def test_optimize_memory():
    df = pd.DataFrame({
        "a": np.array([1, 2, 3], dtype='int64'),
        "b": np.array([1.0, 2.0, 3.0], dtype='float64')
    })
    before = df.memory_usage().sum()
    df_opt = optimize_memory(df)
    after = df_opt.memory_usage().sum()
    assert after < before

def test_preprocess_data():
    # Création d'un mini-dataset équilibré pour éviter le crash du stratify
    df = pd.DataFrame({
        "Age": [20, 30, 40, 50],
        "Number of sexual partners": [1, 2, 3, 4],
        "Biopsy": [0, 1, 0, 1] 
    })
    
    # Récupération des 7 valeurs
    X_train, X_test, y_train, y_test, imputer, scaler, cols = preprocess_data(df)
    
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

def test_training_and_prediction():
    # Teste le pipeline entier pour s'assurer que tout s'enchaîne
    model_path, X_test_df, y_test = train_model_Randomforest()
    
    assert os.path.exists(model_path)
    
    model = joblib.load(model_path)
    sample = X_test_df.iloc[[0]] # Récupère une ligne avec le bon nombre de colonnes
    pred = model.predict(sample)
    
    assert pred[0] in [0, 1]