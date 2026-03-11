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
train_model_XGBoost()