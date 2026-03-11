import joblib
import pandas as pd
from sklearn.svm import SVC
from data_processing import load_and_clean_data, preprocess_data

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
