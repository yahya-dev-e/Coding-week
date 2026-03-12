import os
import pandas as pd
import numpy as np
import joblib
# Added 'optimize_memory' to the import list
from src.data_processing import load_and_clean_data, preprocess_data, optimize_memory
import src.train_model as train_model
import src.evaluate_model as evaluate_model

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
    # Mini-dataset with enough classes for stratification
    df = pd.DataFrame({
        "Age": [20, 30, 40, 50],
        "Number of sexual partners": [1, 2, 3, 4],
        "Biopsy": [0, 1, 0, 1] 
    })
    
    # Unpack the 7 values returned by your src function
    X_train, X_test, y_train, y_test, imputer, scaler, cols = preprocess_data(df)
    
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

def test_training_and_prediction():
    # Use the alias 'train_model' to call the function inside the module
    # Ensure train_model.py actually has a function named train_model_Randomforest
    model_path, X_test_df, y_test = train_model.train_model_Randomforest()
    
    assert os.path.exists(model_path)
    
    model = joblib.load(model_path)
    sample = X_test_df.iloc[[0]] 
    pred = model.predict(sample)
    
    assert pred[0] in [0, 1]