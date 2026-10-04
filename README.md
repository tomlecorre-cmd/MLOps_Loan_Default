# MLOps Credit Risk Predictor

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-172434?style=for-the-badge)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)

This repository contains an end-to-end classification pipeline predicting loan defaults. It implements model tracking with MLflow (using a SQLite database) and deploys the best-performing model via a containerized Streamlit web application.

## 1. Data & Preprocessing
* **Dataset:** Built on `Loan_Data.csv` containing 10,000 customer profiles.
* **Leakage Prevention:** Exploratory Data Analysis (`exploration.ipynb`) revealed an abnormally high correlation between the target (`default`) and two features: `credit_lines_outstanding` (0.86) and `total_debt_outstanding` (0.75). These were dropped in `preprocessing.py` to prevent data leakage.
* **Scaling:** Applied `StandardScaler`, saved as `scaler.pkl` for inference.

## 2. Modeling & Experiment Tracking
Experiments are tracked locally using **MLflow** with a SQLite backend (`mlflow.db`). We trained and logged three model families:
1. **Logistic Regression:** Grid search on penalties (`l1`, `l2`, `None`) using the `saga` solver.
2. **Random Forest:** Benchmarked different depths and estimators (50/100 trees).
3. **XGBoost:** Tested `max_depth` variations with logloss evaluation.

A custom script (`select_best.py`) queries `mlflow.db` to automatically extract the model with the highest F1-Score (XGBoost).

## 3. Web Application
The front-end is built with **Streamlit** (`app.py`). It features two pages:
* **Leaderboard:** Directly queries the `mlflow.db` SQLite database to display the ranking and metrics (Accuracy, Precision, Recall, F1-Score) of all trained models.
* **Inference:** Uses the saved `xgb_model.pkl` to compute live default probabilities based on user inputs (Loan Amount, Income, Years Employed, FICO Score).

## 4. Run Locally (Docker)
The entire pipeline and web app are containerized.

```bash
# Build the image
docker build -t mlops_loan_default .

# Run the container
docker run -p 8501:8501 mlops_loan_default
