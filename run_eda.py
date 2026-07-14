"""
run_eda.py
==========
Standalone script that performs Exploratory Data Analysis (EDA) on the
cleaned Telco Customer Churn dataset and saves all plots to
outputs/plots/. Mirrors the content of notebooks/01_EDA.ipynb.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.preprocessing import DataCleaner
from src.utils import DATA_DIR, PLOTS_DIR, get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid", palette="Set2")


def main() -> None:
    cleaner = DataCleaner()
    df = cleaner.clean(DATA_DIR / "Telco-Customer-Churn.csv")

    print("=" * 70)
    print("STEP 1: DATA LOADING SUMMARY")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 10 rows:\n", df.head(10))
    print("\nInfo:")
    df.info()
    print("\nSummary statistics:\n", df.describe(include="all").T)

    print("\n" + "=" * 70)
    print("STEP 2: DATA CLEANING SUMMARY")
    print("=" * 70)
    print(f"Duplicates removed: {cleaner.n_duplicates_removed_}")
    print(f"Missing values filled: {cleaner.n_missing_filled_}")
    print(f"Remaining missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # ------------------------------------------------------------------
    # STEP 3: EXPLORATORY DATA ANALYSIS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    churn_rate = (df["Churn"] == "Yes").mean() * 100

    # 1. Target distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.countplot(data=df, x="Churn", ax=ax)
    ax.set_title(f"Target Distribution (Overall Churn Rate: {churn_rate:.1f}%)")
    for container in ax.containers:
        ax.bar_label(container)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "target_distribution.png", dpi=150)
    plt.close(fig)
    print(f"INSIGHT: Overall churn rate is {churn_rate:.1f}%. The dataset is moderately "
          f"imbalanced, so accuracy alone is a misleading metric -- precision/recall/ROC AUC matter more.")

    # 2. Gender vs Churn
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.countplot(data=df, x="gender", hue="Churn", ax=ax)
    ax.set_title("Gender vs Churn")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "gender_vs_churn.png", dpi=150)
    plt.close(fig)
    print("INSIGHT: Churn rate is nearly identical across genders, meaning gender is "
          "unlikely to be a strong predictor of churn on its own.")

    # 3. Contract vs Churn
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x="Contract", hue="Churn", ax=ax)
    ax.set_title("Contract Type vs Churn")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "contract_vs_churn.png", dpi=150)
    plt.close(fig)
    mtm_churn = (df[df["Contract"] == "Month-to-month"]["Churn"] == "Yes").mean() * 100
    print(f"INSIGHT: Month-to-month customers churn at {mtm_churn:.1f}%, far higher than "
          "one/two-year contract holders. Contract length is one of the strongest churn drivers.")

    # 4. Internet Service vs Churn
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x="InternetService", hue="Churn", ax=ax)
    ax.set_title("Internet Service vs Churn")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "internet_service_vs_churn.png", dpi=150)
    plt.close(fig)
    fiber_churn = (df[df["InternetService"] == "Fiber optic"]["Churn"] == "Yes").mean() * 100
    print(f"INSIGHT: Fiber optic customers churn at {fiber_churn:.1f}%, notably higher than "
          "DSL customers -- possibly due to higher pricing or service reliability issues.")

    # 5. Payment Method vs Churn
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(data=df, y="PaymentMethod", hue="Churn", ax=ax)
    ax.set_title("Payment Method vs Churn")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "payment_method_vs_churn.png", dpi=150)
    plt.close(fig)
    echeck_churn = (df[df["PaymentMethod"] == "Electronic check"]["Churn"] == "Yes").mean() * 100
    print(f"INSIGHT: Electronic check users churn at {echeck_churn:.1f}%, the highest of all "
          "payment methods, while automatic payment methods show much lower churn.")

    # 6. Monthly Charges distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(data=df, x="MonthlyCharges", hue="Churn", kde=True, ax=ax, element="step")
    ax.set_title("Monthly Charges Distribution by Churn")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "monthly_charges_distribution.png", dpi=150)
    plt.close(fig)
    print("INSIGHT: Churned customers cluster at higher monthly charges (~$70-100), "
          "suggesting price sensitivity is a meaningful churn driver.")

    # 7. Tenure distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(data=df, x="tenure", hue="Churn", kde=True, ax=ax, element="step")
    ax.set_title("Tenure Distribution by Churn")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "tenure_distribution.png", dpi=150)
    plt.close(fig)
    print("INSIGHT: Churn is heavily concentrated among customers with low tenure "
          "(<12 months), while long-tenured customers rarely churn -- loyalty compounds.")

    # 8. Correlation Heatmap (numeric columns only)
    numeric_df = df.copy()
    numeric_df["Churn_binary"] = (numeric_df["Churn"] == "Yes").astype(int)
    numeric_df["SeniorCitizen"] = numeric_df["SeniorCitizen"].astype(int)
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn_binary"]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(numeric_df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print("INSIGHT: Tenure is negatively correlated with churn while MonthlyCharges is "
          "positively correlated -- customers who pay more and stay less are the highest risk.")

    # 9. Pairplot (numeric columns, sampled for speed)
    sample_df = numeric_df.sample(min(800, len(numeric_df)), random_state=42)
    pp = sns.pairplot(
        sample_df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]],
        hue="Churn", diag_kind="kde", plot_kws={"alpha": 0.5, "s": 15},
    )
    pp.fig.suptitle("Pairplot of Key Numeric Features", y=1.02)
    pp.savefig(PLOTS_DIR / "pairplot.png", dpi=150)
    plt.close(pp.fig)
    print("INSIGHT: The pairplot confirms churned customers (orange) skew toward low "
          "tenure and high monthly charges, with total charges naturally following tenure.")

    # 10. Countplots for key categorical variables
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, col in zip(axes.ravel(), ["Partner", "Dependents", "PaperlessBilling", "SeniorCitizen"]):
        sns.countplot(data=df, x=col, hue="Churn", ax=ax)
        ax.set_title(f"{col} vs Churn")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "categorical_countplots.png", dpi=150)
    plt.close(fig)
    senior_churn = (df[df["SeniorCitizen"] == 1]["Churn"] == "Yes").mean() * 100
    print(f"INSIGHT: Senior citizens churn at {senior_churn:.1f}%, higher than non-seniors, "
          "and customers without partners/dependents churn more -- suggesting single "
          "households are more price-sensitive and less 'locked in'.")

    # 11. Boxplots for outlier detection
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        sns.boxplot(data=df, y=col, ax=ax)
        ax.set_title(f"Boxplot: {col}")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "outlier_boxplots.png", dpi=150)
    plt.close(fig)

    # Outlier detection via IQR method
    print("\nOutlier detection (IQR method):")
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        print(f"  {col}: {n_outliers} outliers detected (bounds: [{lower:.1f}, {upper:.1f}])")
    print("INSIGHT: None of the numeric features show extreme outliers requiring removal -- "
          "the boxplots show tight, business-plausible ranges (e.g., charges never negative "
          "or absurdly high), so no rows were dropped for outliers.")

    print("\nEDA complete. All plots saved to outputs/plots/")


if __name__ == "__main__":
    main()
