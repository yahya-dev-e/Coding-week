import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from data_processing import load_and_clean_data, remove_outliers_iqr, preprocess_data
import os
from catboost import CatBoostClassifier





def train_model_catboost(
    data_path: str = "data/risk_factors_cervical_cancer.csv",
    model_output_path: str = "models/catboost_model.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
    target: str = "Biopsy",
    catboost_params: dict = None,
) -> dict:
    """
    Load data, train a CatBoost classifier, and save the model.

    Parameters
    ----------
    data_path : str
        Path to the raw CSV dataset.
    model_output_path : str
        Where to save the trained model (.pkl).
    test_size : float
        Fraction of data reserved for the held-out test set.
    random_state : int
        Seed for reproducibility.
    target : str
        Name of the binary target column.
    catboost_params : dict, optional
        Override default CatBoost hyperparameters.

    Returns
    -------
    dict with keys:
        "model"      – fitted CatBoostClassifier
        "X_test"     – held-out features (pd.DataFrame)
        "y_test"     – held-out labels  (pd.Series)
        "model_path" – absolute path of the saved model file
    """
    # ── 1. Load ──────────────────────────────────────────────────────────────
    df = pd.read_csv(data_path, na_values="?")

    X = df.drop(columns=[target])
    y = df[target]

    # Coerce to numeric (columns were forced to str by '?' values)
    X = X.apply(pd.to_numeric, errors="coerce")

    # ── 2. Impute ─────────────────────────────────────────────────────────────
    X = X.fillna(X.median())

    # ── 3. Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ── 4. SMOTE ──────────────────────────────────────────────────────────────
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # ── 5. Train ──────────────────────────────────────────────────────────────
    default_params = dict(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        verbose=100,
        random_seed=random_state,
    )
    if catboost_params:
        default_params.update(catboost_params)

    model = CatBoostClassifier(**default_params)
    model.fit(X_train, y_train)

    # ── 6. Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved → {os.path.abspath(model_output_path)}")

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "model_path": os.path.abspath(model_output_path),
    }






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
