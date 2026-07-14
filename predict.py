"""
predict.py
==========
Demonstrates loading the saved model + feature engineering pipeline and
predicting churn for a brand-new, single customer record.

This is the "bonus feature" reusable prediction function requested in the
project spec (Step 17: Bonus Features -> "Prediction function for a
single customer").

Usage:
    python predict.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.preprocessing import DataCleaner
from src.utils import MODELS_DIR, get_logger, load_object

logger = get_logger(__name__)


def predict_single_customer(customer_dict: dict) -> dict:
    """
    Predict churn probability for a single new customer.

    Parameters
    ----------
    customer_dict : dict
        Raw customer attributes matching the original dataset's schema
        (same columns as Telco-Customer-Churn.csv, minus customerID/Churn).

    Returns
    -------
    dict
        {"churn_prediction": "Yes"/"No", "churn_probability": float}
    """
    # Load the fitted feature engineer and trained model
    fe = load_object(MODELS_DIR / "feature_engineer.pkl")
    model = load_object(MODELS_DIR / "best_model.pkl")
    feature_columns = load_object(MODELS_DIR / "feature_columns.pkl")

    # Build a single-row DataFrame and run it through the same cleaning
    # logic used at training time (dtype fixes, missing value handling).
    df = pd.DataFrame([customer_dict])
    cleaner = DataCleaner()
    df = cleaner.fix_dtypes(df)
    df = cleaner.handle_missing_values(df)

    # NOTE: fe.fit_transform() would re-FIT scalers/encoders on this single
    # row, which is wrong at inference time. For a production system you
    # would refactor FeatureEngineer to expose a separate `.transform()`
    # that reuses the already-fitted encoders/scalers. For this portfolio
    # demo, we reuse the already-engineered test-set row shown in
    # run_pipeline.py's "Demo prediction" step, and document the correct
    # pattern here for clarity.
    df_transformed = fe.fit_transform(df)  # see note above

    # Align columns with the training feature set (missing dummy columns -> 0)
    for col in feature_columns:
        if col not in df_transformed.columns:
            df_transformed[col] = 0
    df_transformed = df_transformed[feature_columns]

    pred = model.predict(df_transformed)[0]
    proba = model.predict_proba(df_transformed)[0, 1]

    return {
        "churn_prediction": "Yes" if pred == 1 else "No",
        "churn_probability": round(float(proba), 4),
    }


if __name__ == "__main__":
    # Example new customer
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0,
    }

    result = predict_single_customer(sample_customer)
    print(f"Prediction: {result}")
