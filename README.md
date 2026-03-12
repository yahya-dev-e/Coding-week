# Cervical Cancer Risk Classification

This repository contains an end-to-end Machine Learning project aimed at analyzing risk factors and predicting the likelihood of cervical cancer. The project encompasses exploratory data analysis (EDA), data processing, model training, evaluation, and a deployable web application.

## 📂 Project Structure

* **`app/`**: Contains the code for the web application (`app.py`).
* **`data/`**: Stores the dataset (`risk_factors_cervical_cancer.csv`).
* **`models/`**: Holds the serialized trained models and their assets (Random Forest, CatBoost, SVM, XGBoost).
* **`notebook/`**: Contains Jupyter notebooks for Exploratory Data Analysis (`eda (5).ipynb`).
* **`src/`**: The core source code for the machine learning pipeline:
    * `data_processing.py`: Scripts for cleaning and preparing the data.
    * `train_model.py`: Scripts for training the various machine learning models.
    * `evaluate_model.py`: Scripts for evaluating model performance.
* **`tests/`**: Unit tests to ensure the reliability of the models (`test_models.py`).
* **`Dockerfile`**: Configuration for containerizing the application.
* **`requirements.txt`**: List of Python dependencies required to run the project.

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.x installed. It is recommended to use a virtual environment.

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yahya-dev-e/Coding-week.git](https://github.com/yahya-dev-e/Coding-week.git)
   cd Coding-week
2. Install the required dependancies:
   ```bash
   pip install -r requirements.txt
## 🧠 Machine Learninig models

This project evaluates several algorithms to find the best performing model for the dataset. The trained models are saved in the models/ directory and include:

  * Random Forest Classifier.
  
  * SVM.

  * CatBoost Classifier.
  
  * XGBoost.