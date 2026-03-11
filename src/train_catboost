import pandas as pd
import numpy as np
import data_processing as dp

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
import joblib


# Load dataset and treat '?' as missing values
df = pd.read_csv("data/risk_factors_cervical_cancer.csv", na_values='?')


# Target column
target = "Biopsy"

X = df.drop(columns=[target])
y = df[target]


# Convert all columns to numeric (important because '?' made them strings)
X = X.apply(pd.to_numeric, errors='coerce')


# Replace missing values with column medians
X = X.fillna(X.median())


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Handle class imbalance
smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)


# Train CatBoost model
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    verbose=100
)

model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)


print("CatBoost Performance")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc)


# Save model
joblib.dump(model, "models/catboost_model.pkl")

print("Model saved successfully.")
