"""
evaluation.py
==============
Generates evaluation artifacts for a trained classification model:
    - Confusion Matrix
    - Classification Report
    - ROC Curve
    - Precision-Recall Curve
    - Learning Curve
    - Cross-Validation scores
    - Feature Importance bar chart

All plots are saved to the outputs/plots directory as PNG files so they
can be embedded in the README or a portfolio write-up.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, learning_curve

from src.utils import PLOTS_DIR, get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")


def plot_confusion_matrix(y_true, y_pred, filename: str = "confusion_matrix.png") -> Path:
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix - Best Model")
    fig.tight_layout()
    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix to {out_path}")
    return out_path


def get_classification_report(y_true, y_pred) -> str:
    """Return the sklearn classification report as a formatted string."""
    return classification_report(y_true, y_pred, target_names=["No Churn", "Churn"])


def plot_roc_curve(model, X_test, y_test, filename: str = "roc_curve.png") -> Path:
    """Plot and save the ROC curve for a fitted model."""
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    ax.set_title("ROC Curve - Best Model")
    ax.legend()
    fig.tight_layout()
    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved ROC curve to {out_path}")
    return out_path


def plot_precision_recall_curve(model, X_test, y_test, filename: str = "precision_recall_curve.png") -> Path:
    """Plot and save the Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("Precision-Recall Curve - Best Model")
    fig.tight_layout()
    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved precision-recall curve to {out_path}")
    return out_path


def plot_learning_curve(model, X, y, filename: str = "learning_curve.png", cv: int = 5) -> Path:
    """Plot training vs. validation score as a function of training set size."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1,
    )
    train_mean, val_mean = train_scores.mean(axis=1), val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(train_sizes, train_mean, "o-", label="Training score")
    ax.plot(train_sizes, val_mean, "o-", label="Cross-validation score")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Learning Curve - Best Model")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved learning curve to {out_path}")
    return out_path


def get_cross_val_scores(model, X, y, cv: int = 5, scoring: str = "roc_auc") -> np.ndarray:
    """Return cross-validation scores for a given model."""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    logger.info(f"Cross-val {scoring} scores: {scores.round(4).tolist()} | Mean: {scores.mean():.4f}")
    return scores


def plot_feature_importance(
    model, feature_names: List[str], top_n: int = 20, filename: str = "feature_importance.png"
) -> Optional[pd.DataFrame]:
    """
    Plot the top-N most important features for tree-based models.

    Parameters
    ----------
    model : fitted estimator
        Must expose `feature_importances_` (tree-based models).
    feature_names : List[str]
        Names corresponding to the model's input columns.
    top_n : int
        Number of top features to display.

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame of feature, importance -- or None if the model has no
        `feature_importances_` attribute.
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning(f"{model.__class__.__name__} has no feature_importances_ attribute.")
        return None

    importances = model.feature_importances_
    imp_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax, palette="viridis")
    ax.set_title(f"Top {top_n} Feature Importances - Best Model")
    fig.tight_layout()
    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved feature importance chart to {out_path}")
    return imp_df
