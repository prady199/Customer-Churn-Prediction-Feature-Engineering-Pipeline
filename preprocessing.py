"""
preprocessing.py
=================
Handles data loading and cleaning for the Telco Customer Churn dataset.

Responsibilities:
    - Load the raw CSV file.
    - Fix data types (e.g., TotalCharges should be numeric, not object).
    - Handle missing values.
    - Remove duplicate records.

Keeping cleaning logic separate from feature engineering keeps each module
focused on a single responsibility (Single Responsibility Principle).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """
    Encapsulates all data-cleaning steps for the Telco Churn dataset.

    Using a class (rather than loose functions) lets us store intermediate
    state -- e.g., how many rows were dropped -- which is useful for
    reporting/auditing in a production pipeline.
    """

    def __init__(self) -> None:
        self.n_duplicates_removed_: int = 0
        self.n_missing_filled_: int = 0

    # ----------------------------------------------------------------
    def load_data(self, filepath: Path) -> pd.DataFrame:
        """
        Load the raw CSV file into a pandas DataFrame.

        Parameters
        ----------
        filepath : Path
            Path to the Telco-Customer-Churn.csv file.

        Returns
        -------
        pd.DataFrame
            Raw, unmodified data.

        Raises
        ------
        FileNotFoundError
            If the CSV file does not exist at the given path.
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Dataset not found at {filepath}")

        df = pd.read_csv(filepath)
        logger.info(f"Loaded dataset with shape {df.shape}")
        return df

    # ----------------------------------------------------------------
    def fix_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix known dtype issues in the raw dataset.

        The most notable issue in the Telco dataset is that
        'TotalCharges' is stored as an object/string column because a
        handful of rows contain blank strings (customers with 0 tenure
        who haven't been billed yet). We coerce it to numeric, turning
        those blanks into NaN so they can be handled explicitly.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with corrected dtypes.
        """
        df = df.copy()
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            logger.info("Converted TotalCharges to numeric (errors coerced to NaN).")
        return df

    # ----------------------------------------------------------------
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Strategy
        --------
        For 'TotalCharges', missing values occur exactly when tenure == 0
        (brand-new customers who have not been charged yet). The most
        sensible business-informed imputation is 0, since they genuinely
        have not accumulated any charges -- NOT the column mean, which
        would fabricate a charge history that never happened.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values handled.
        """
        df = df.copy()
        n_missing_before = df.isnull().sum().sum()

        if "TotalCharges" in df.columns:
            mask = df["TotalCharges"].isnull()
            df.loc[mask, "TotalCharges"] = 0.0
            logger.info(f"Filled {mask.sum()} missing TotalCharges values with 0 (tenure=0 customers).")

        n_missing_after = df.isnull().sum().sum()
        self.n_missing_filled_ = int(n_missing_before - n_missing_after)
        return df

    # ----------------------------------------------------------------
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[str] = "customerID") -> pd.DataFrame:
        """
        Remove duplicate rows, keyed on customerID when available.

        Parameters
        ----------
        df : pd.DataFrame
        subset : Optional[str]
            Column(s) to consider when identifying duplicates.

        Returns
        -------
        pd.DataFrame
            De-duplicated DataFrame.
        """
        df = df.copy()
        n_before = len(df)
        if subset and subset in df.columns:
            df = df.drop_duplicates(subset=subset)
        else:
            df = df.drop_duplicates()
        self.n_duplicates_removed_ = n_before - len(df)
        logger.info(f"Removed {self.n_duplicates_removed_} duplicate rows.")
        return df.reset_index(drop=True)

    # ----------------------------------------------------------------
    def clean(self, filepath: Path) -> pd.DataFrame:
        """
        Run the full cleaning pipeline: load -> fix dtypes -> handle
        missing values -> remove duplicates.

        Parameters
        ----------
        filepath : Path
            Path to the raw CSV.

        Returns
        -------
        pd.DataFrame
            Fully cleaned DataFrame, ready for EDA / feature engineering.
        """
        df = self.load_data(filepath)
        df = self.fix_dtypes(df)
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)

        # Drop customerID - it's a unique identifier with no predictive value.
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        logger.info(f"Cleaning complete. Final shape: {df.shape}")
        return df
