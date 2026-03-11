import pickle
import pandas as pd
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

if __name__ == "__main__":
    train_model()