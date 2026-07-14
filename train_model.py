"""
train_model.py
===============
Trains multiple classification algorithms on the engineered Telco Churn
dataset, performs hyperparameter tuning via GridSearchCV, and persists
the best-performing model to disk.

Design notes
------------
- `ModelTrainer` trains a dictionary of baseline models and returns a
  comparison table (accuracy, precision, recall, F1, ROC AUC).
- `tune_model` runs GridSearchCV for a given estimator + param grid,
  reusable for RandomForest, DecisionTree, GradientBoosting, etc.
"""

from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.utils import get_logger

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False


def get_baseline_models() -> Dict[str, object]:
    """
    Return a dictionary of baseline (default hyperparameter) models to
    benchmark against one another.

    Returns
    -------
    Dict[str, object]
        Mapping of model name -> unfitted scikit-learn estimator.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "SVM": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=42
        )
    return models


class ModelTrainer:
    """
    Trains and evaluates multiple classification models, producing a
    standardized comparison table.
    """

    def __init__(self) -> None:
        self.results_: Dict[str, Dict[str, float]] = {}
        self.fitted_models_: Dict[str, object] = {}

    def train_and_evaluate(
        self,
        models: Dict[str, object],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """
        Fit each model on the training set and compute evaluation metrics
        on the held-out test set.

        Parameters
        ----------
        models : Dict[str, object]
            Mapping of model name -> unfitted estimator.
        X_train, y_train, X_test, y_test : pd.DataFrame / pd.Series
            Train/test splits.

        Returns
        -------
        pd.DataFrame
            Comparison table sorted by ROC AUC (descending).
        """
        for name, model in models.items():
            logger.info(f"Training {name} ...")
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = (
                    model.predict_proba(X_test)[:, 1]
                    if hasattr(model, "predict_proba")
                    else y_pred
                )

                metrics = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                    "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                    "ROC AUC": roc_auc_score(y_test, y_proba),
                }
                self.results_[name] = metrics
                self.fitted_models_[name] = model
                logger.info(f"{name} -> ROC AUC: {metrics['ROC AUC']:.4f}")
            except Exception as exc:
                logger.error(f"Failed to train {name}: {exc}")

        comparison_df = pd.DataFrame(self.results_).T.sort_values("ROC AUC", ascending=False)
        return comparison_df


def tune_model(
    estimator: object,
    param_grid: Dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> Tuple[object, Dict]:
    """
    Perform hyperparameter tuning using GridSearchCV.

    Parameters
    ----------
    estimator : object
        Unfitted scikit-learn compatible estimator.
    param_grid : Dict[str, list]
        Grid of hyperparameters to search.
    X_train, y_train : pd.DataFrame / pd.Series
        Training data.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric used to select the best parameters.

    Returns
    -------
    Tuple[object, Dict]
        (best fitted estimator, best parameter dictionary)
    """
    logger.info(f"Starting GridSearchCV for {estimator.__class__.__name__} ...")
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    logger.info(
        f"Best params for {estimator.__class__.__name__}: {grid_search.best_params_} "
        f"(CV {scoring}: {grid_search.best_score_:.4f})"
    )
    return grid_search.best_estimator_, grid_search.best_params_
