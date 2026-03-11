import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from data_processing import load_and_clean_data, remove_outliers_iqr, optimize_memory, preprocess_data, supprimer_colonnes_zero

def evaluate():
    # 1. Préparation des données de test
    df = load_and_clean_data('data/risk_factors_cervical_cancer.csv')
    _, X_test_scaled, _, y_test, _, _, _ = preprocess_data(df)    
    # 2. Chargement avec PICKLE
    try:
        with open('trained_svm_assets.pkl', 'rb') as f:
            assets = pickle.load(f)
        model = assets['model']
    except FileNotFoundError:
        print("❌ Erreur : Fichier pkl non trouvé. Lance d'abord train_model.py")
        return
    
    # 3. Prédictions et Scores
    predictions = model.predict(X_test_scaled)
    print("\n📊 MATRICE DE CONFUSION :")
    print(confusion_matrix(y_test, predictions))
    print("\n📋 RAPPORT DE CLASSIFICATION :")
    print(classification_report(y_test, predictions))

