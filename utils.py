"""
utils.py
========
Common utility functions used across the Customer Churn Prediction project.

This module centralizes:
    - Logging configuration
    - File path helpers
    - Generic helper functions (e.g., saving/loading objects)

Following the DRY (Don't Repeat Yourself) principle, any functionality that
is needed in more than one script lives here.
"""

import logging
import os
from pathlib import Path
from typing import Any

import joblib

# --------------------------------------------------------------------------
# Project directory constants
# --------------------------------------------------------------------------
# Using pathlib makes path handling OS-independent (Windows/Mac/Linux safe).
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"

# Make sure the directories exist (idempotent - safe to call every run)
for directory in (DATA_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Create and configure a logger.

    Using a logger instead of print() is an industry best practice because
    it gives us timestamps, severity levels, and the ability to redirect
    output to files in production.

    Parameters
    ----------
    name : str
        Usually pass __name__ from the calling module.

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on re-import
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_object(obj: Any, filepath: os.PathLike) -> None:
    """
    Persist any Python object (model, encoder, scaler, etc.) to disk using
    joblib, which is more efficient than pickle for objects containing
    large numpy arrays (e.g., scikit-learn models).

    Parameters
    ----------
    obj : Any
        The object to serialize.
    filepath : os.PathLike
        Destination path (should end in .pkl).
    """
    try:
        joblib.dump(obj, filepath)
    except Exception as exc:  # broad except is fine here - we re-raise with context
        raise IOError(f"Failed to save object to {filepath}: {exc}") from exc


def load_object(filepath: os.PathLike) -> Any:
    """
    Load a previously saved object from disk.

    Parameters
    ----------
    filepath : os.PathLike
        Path to the .pkl file.

    Returns
    -------
    Any
        The deserialized Python object.
    """
    try:
        return joblib.load(filepath)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"No saved object found at {filepath}") from exc
