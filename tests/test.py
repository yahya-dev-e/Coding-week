
import os
import pandas as pd
import numpy as np
import joblib
from src.data_processing import load_and_clean_data, optimize_memory, preprocess_data, remove_outliers_iqr,supprimer_colonnes_zero, drop_high_correlation
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