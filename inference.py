"""Inference module for Loan Approval Prediction Model."""

import joblib
import pandas as pd
import numpy as np

# Loading model and the metadata
bundle = joblib.load("loan_logreg_model.pkl")
model = bundle["model"]
threshold = bundle["threshold"]
feature_names = bundle["feature_names"]
scaler = bundle["scaler"]   # MinMaxScaler loaded

# columns scaled during training
scaled_cols = ["LoanAmount_log", "TotalIncome_log", "Balance_Income"]


def predict_from_features(feature_row: dict):
    """
    feature_row contain these 14 final features:

    LoanAmount_log, TotalIncome_log, EMI, Balance_Income,
    Gender_Male, Married_Yes, Dependents_0, Dependents_1,
    Dependents_2, Education_Not Graduate, Self_Employed_Yes,
    Credit_History_1.0, Property_Area_Semiurban, Property_Area_Urban
    """

    # Convert to DataFrame
    df = pd.DataFrame([feature_row])

    # Keep correct order
    df = df[feature_names]

    # Apply MinMax scaling to the same 3 columns
    df_scaled = df.copy()
    df_scaled[scaled_cols] = scaler.transform(df[scaled_cols])

    X = df_scaled.values

    # Predict
    proba = model.predict_proba(X)[0, 1]
    pred = int(proba >= threshold)

    return {
        "probability": float(proba),
        "prediction": pred,
        "label": "Approved" if pred == 1 else "Rejected"
    }
