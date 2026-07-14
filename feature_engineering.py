"""
feature_engineering.py
=======================
Advanced feature engineering for the Telco Customer Churn dataset.

Responsibilities:
    - Binary label encoding.
    - One-hot encoding of categorical columns.
    - Feature scaling (StandardScaler / MinMaxScaler).
    - Skewness check + log transformation.
    - Creation of new, business-driven features.

All transformations are wrapped in a single `FeatureEngineer` class so the
exact same fitted transformers can be reused at inference time (critical
for avoiding train/serve skew in production).
"""

from typing import List

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from src.utils import get_logger

logger = get_logger(__name__)

BINARY_COLUMNS: List[str] = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
    "MultipleLines",
    "gender",
]

ONE_HOT_COLUMNS: List[str] = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "DeviceProtection",
    "StreamingTV",
    "StreamingMovies",
    "OnlineBackup",
    "OnlineSecurity",
    "TechSupport",
]

SERVICE_COLUMNS: List[str] = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


class FeatureEngineer:
    """
    Encapsulates all feature engineering logic as a fit/transform object,
    mirroring the scikit-learn Transformer API. This makes it trivial to
    slot into an sklearn Pipeline later if desired.
    """

    def __init__(self) -> None:
        self.label_encoders_: dict = {}
        self.standard_scaler_: StandardScaler = StandardScaler()
        self.minmax_scaler_: MinMaxScaler = MinMaxScaler()
        self.monthly_charges_mean_: float = 0.0
        self.fitted_: bool = False

    # ----------------------------------------------------------------
    def _encode_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode Yes/No (and SeniorCitizen, already 0/1) columns."""
        df = df.copy()

        # SeniorCitizen is already numeric (0/1) in the raw data, but we
        # ensure dtype consistency.
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

        for col in BINARY_COLUMNS:
            if col not in df.columns:
                continue
            if col not in self.label_encoders_:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders_[col] = le
            else:
                le = self.label_encoders_[col]
                df[col] = le.transform(df[col].astype(str))
        logger.info(f"Label-encoded binary columns: {[c for c in BINARY_COLUMNS if c in df.columns]}")
        return df

    # ----------------------------------------------------------------
    def _one_hot_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode nominal categorical columns with >2 categories."""
        cols_present = [c for c in ONE_HOT_COLUMNS if c in df.columns]
        df = pd.get_dummies(df, columns=cols_present, drop_first=False)
        # Convert resulting boolean dummy columns to int (0/1) for model compatibility
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)
        logger.info(f"One-hot encoded columns: {cols_present}")
        return df

    # ----------------------------------------------------------------
    def _create_new_features(self, df: pd.DataFrame, service_flags: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer new, business-meaningful features.

        service_flags: the ORIGINAL (pre one-hot) categorical service
        columns, needed to count "Yes" services before they get expanded
        into dummy columns.
        """
        df = df.copy()

        # 1. Average Monthly Spend = TotalCharges / tenure (avoid div-by-zero)
        df["AvgMonthlySpend"] = np.where(
            df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
        )

        # 2. Number of Services subscribed to (count columns == 'Yes')
        def count_services(row: pd.Series) -> int:
            count = 0
            for col in SERVICE_COLUMNS:
                if col in row.index and str(row[col]) == "Yes":
                    count += 1
                elif col == "PhoneService" and str(row.get(col, "")) == "Yes":
                    count += 1
            return count

        df["NumServices"] = service_flags.apply(count_services, axis=1)

        # 3. Customer Lifetime Category based on tenure (months)
        def lifetime_category(t: float) -> str:
            if t <= 12:
                return "New"
            elif t <= 36:
                return "Medium"
            else:
                return "Loyal"

        df["CustomerLifetimeCategory"] = df["tenure"].apply(lifetime_category)

        # 4. Auto Payment Flag
        if "PaymentMethod" in service_flags.columns:
            df["AutoPaymentFlag"] = (
                service_flags["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)
            )
        else:
            df["AutoPaymentFlag"] = 0

        # 5. Monthly Charge Category (Low / Medium / High) via terciles
        low_thresh, high_thresh = df["MonthlyCharges"].quantile([0.33, 0.66])

        def charge_category(x: float) -> str:
            if x <= low_thresh:
                return "Low"
            elif x <= high_thresh:
                return "Medium"
            else:
                return "High"

        df["MonthlyChargeCategory"] = df["MonthlyCharges"].apply(charge_category)

        # 6. Tenure Groups (4 buckets across the 0-72 month range)
        def tenure_group(t: float) -> str:
            if t <= 12:
                return "New Customer"
            elif t <= 24:
                return "Growing Customer"
            elif t <= 48:
                return "Long-term Customer"
            else:
                return "Premium Customer"

        df["TenureGroup"] = df["tenure"].apply(tenure_group)

        # 7. Premium Customer Flag: MonthlyCharges above the dataset average
        self.monthly_charges_mean_ = df["MonthlyCharges"].mean()
        df["PremiumCustomerFlag"] = (df["MonthlyCharges"] > self.monthly_charges_mean_).astype(int)

        logger.info(
            "Created new features: AvgMonthlySpend, NumServices, CustomerLifetimeCategory, "
            "AutoPaymentFlag, MonthlyChargeCategory, TenureGroup, PremiumCustomerFlag"
        )
        return df

    # ----------------------------------------------------------------
    def _check_skewness_and_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check skewness of TotalCharges and apply a log1p transform if the
        distribution is significantly right-skewed (|skew| > 0.75 is a
        common rule-of-thumb threshold).
        """
        df = df.copy()
        col_skew = skew(df["TotalCharges"].dropna())
        logger.info(f"Skewness of TotalCharges before transform: {col_skew:.3f}")

        if abs(col_skew) > 0.75:
            # log1p handles zero values gracefully (log(1+x)), unlike log(x)
            df["TotalCharges_log"] = np.log1p(df["TotalCharges"])
            new_skew = skew(df["TotalCharges_log"])
            logger.info(
                f"TotalCharges is significantly skewed ({col_skew:.3f}). "
                f"Applied log1p transform -> new skewness: {new_skew:.3f}. "
                "Log transformation compresses the long right tail (high-spend "
                "customers), making the distribution closer to normal, which "
                "benefits distance-based and linear models (KNN, Logistic "
                "Regression, SVM)."
            )
        else:
            logger.info("TotalCharges skewness within acceptable range; no transform applied.")
        return df

    # ----------------------------------------------------------------
    def _scale_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Scale numeric features.
            - StandardScaler on MonthlyCharges (roughly normal, used by
              distance/gradient based models that assume zero-mean/unit-var).
            - MinMaxScaler on tenure & TotalCharges (bounded, interpretable
              0-1 range; good for features with a hard natural minimum of 0).
        """
        df = df.copy()

        if fit:
            df[["MonthlyCharges"]] = self.standard_scaler_.fit_transform(df[["MonthlyCharges"]])
            df[["tenure", "TotalCharges"]] = self.minmax_scaler_.fit_transform(df[["tenure", "TotalCharges"]])
        else:
            df[["MonthlyCharges"]] = self.standard_scaler_.transform(df[["MonthlyCharges"]])
            df[["tenure", "TotalCharges"]] = self.minmax_scaler_.transform(df[["tenure", "TotalCharges"]])

        logger.info("Scaled MonthlyCharges (StandardScaler) and tenure/TotalCharges (MinMaxScaler).")
        return df

    # ----------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit all transformers on the training data and return the fully
        engineered feature set.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned data (output of DataCleaner.clean()).

        Returns
        -------
        pd.DataFrame
            Fully engineered, model-ready DataFrame.
        """
        service_flags = df[[c for c in SERVICE_COLUMNS + ["PaymentMethod"] if c in df.columns]].copy()

        df = self._create_new_features(df, service_flags)
        df = self._check_skewness_and_transform(df)
        df = self._encode_binary(df)

        # Also one-hot encode the newly created categorical features
        extra_categoricals = ["CustomerLifetimeCategory", "MonthlyChargeCategory", "TenureGroup"]
        df = pd.get_dummies(df, columns=[c for c in extra_categoricals if c in df.columns], drop_first=False)
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

        df = self._one_hot_encode(df)
        df = self._scale_features(df, fit=True)

        self.fitted_ = True
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df
