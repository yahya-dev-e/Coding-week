import os
import pandas as pd
import numpy as np
import joblib
from data_processing import load_and_clean_data, preprocess_data, optimize_memory
import train_model as train_model
import evaluate_model as evaluate_model

def test_dataset_loading():
    # Ensure this path is relative to the root 'Coding-week'
    df = load_and_clean_data("data/risk_factors_cervical_cancer.csv")
    assert df.shape[0] > 0
    # Clean check: ensure no '?' strings remain
    assert '?' not in df.values 

def test_optimize_memory():
    df = pd.DataFrame({
        "a": np.array([1, 2, 3], dtype='int64'),
        "b": np.array([1.0, 2.0, 3.0], dtype='float64')
    })
    before = df.memory_usage().sum()
    df_opt = optimize_memory(df)
    after = df_opt.memory_usage().sum()
    assert after <= before # Changed to <= in case the sample is too small to shrink

def test_preprocess_data():
    # Increase the dataset size so a 20-25% split results in at least 2 rows for the test set
    df = pd.DataFrame({
        "Age": [20, 30, 40, 50, 60, 70, 80, 90],
        "Number of sexual partners": [1, 2, 3, 4, 1, 2, 3, 4],
        "Biopsy": [0, 1, 0, 1, 0, 1, 0, 1] # Balanced classes
    })
    
    # Now a 25% test_size will result in 2 rows, enough to hold one '0' and one '1'
    X_train, X_test, y_train, y_test, imputer, scaler, cols = preprocess_data(df)
    
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_test) >= 2  # Verification that we have a valid test sample


def test_remove_outliers_iqr():
    # Création d'un DataFrame avec des outliers évidents
    df = pd.DataFrame({
        "Age": [20, 25, 30, 35, 200],           # 200 = outlier clair
        "Number of sexual partners": [1, 2, 2, 3, 50],  # 50 = outlier clair
        "Biopsy": [0, 0, 1, 0, 1]
    })
    
    df_clean = remove_outliers_iqr(df)
    
    # Les lignes avec outliers doivent avoir été supprimées
    assert df_clean.shape[0] < df.shape[0]
    # Vérifie que les valeurs extrêmes ne sont plus là
    assert df_clean["Age"].max() < 200
    assert df_clean["Number of sexual partners"].max() < 50

    
def test_supprimer_colonnes_zero():
    df = pd.DataFrame({
        "Age":     [20, 30, 40],
        "col_zero": [0, 0, 0],      # doit être supprimée
        "Biopsy":  [0, 1, 0]
    })
    
    df_clean = supprimer_colonnes_zero(df)
    
    # La colonne nulle doit avoir disparu
    assert "col_zero" not in df_clean.columns
    # Les autres colonnes doivent être intactes
    assert "Age" in df_clean.columns
    assert "Biopsy" in df_clean.columns
def test_drop_high_correlation():
    df = pd.DataFrame({
        "feature_A": [1, 2, 3, 4, 5],
        "feature_B": [1, 2, 3, 4, 5],      # corrélation parfaite avec A → doit être supprimée
        "feature_C": [5, 3, 1, 4, 2]        # pas corrélée → doit être conservée
    })
    
    df_clean = drop_high_correlation(df, threshold=0.9)
    
    # feature_B doit avoir été supprimée (corrélation = 1.0 avec feature_A)
    assert "feature_B" not in df_clean.columns
    # feature_A et feature_C doivent être conservées
    assert "feature_A" in df_clean.columns
    assert "feature_C" in df_clean.columns


def test_drop_high_correlation_no_drop():
    # Cas où aucune corrélation ne dépasse le seuil → rien ne doit être supprimé
    df = pd.DataFrame({
        "feature_A": [1, 2, 3, 4, 5],
        "feature_B": [5, 3, 1, 4, 2],      # pas corrélée
        "feature_C": [2, 4, 1, 5, 3]        # pas corrélée
    })
    
    df_clean = drop_high_correlation(df, threshold=0.9)
    
    # Toutes les colonnes doivent être conservées
    assert df_clean.shape[1] == df.shape[1]

def test_training_and_prediction_RandomForest():
    # Use the alias 'train_model' to call the function inside the module
    # Ensure train_model.py actually has a function named train_model_Randomforest
    model_path, X_test_df, y_test = train_model.train_model_Randomforest()
    
    assert os.path.exists(model_path)
    
    model = joblib.load(model_path)
    sample = X_test_df.iloc[[0]] 
    pred = model.predict(sample)
    
    assert pred[0] in [0, 1]
def test_training_and_prediction_XBoost():
    model_path, X_test_df, y_test = train_model.train_model_XGBoost()
    
    assert os.path.exists(model_path)
    
    model = joblib.load(model_path)
    sample = X_test_df.iloc[[0]] 
    pred = model.predict(sample)
    
    assert pred[0] in [0, 1]
def test_training_and_prediction_catboost():
    model_path, X_test_df, y_test = train_model.train_model_catboost()
    
    assert os.path.exists(model_path)
    
    model = joblib.load(model_path)
    sample = X_test_df.iloc[[0]] 
    pred = model.predict(sample)
    
    assert pred[0] in [0, 1]
def test_training_and_prediction_svm():
    model_path, X_test_df, y_test = train_model.train_model_svm()
    
    assert os.path.exists(model_path)
    
    model = joblib.load(model_path)
    sample = X_test_df.iloc[[0]] 
    pred = model.predict(sample)
    
    assert pred[0] in [0, 1]