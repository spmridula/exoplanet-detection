# src/models/baseline.py
"""
baseline.py
───────────
Week 2, Day 8: Random Forest and XGBoost classifiers.

These are the baseline models — deliberately simple.
Their job is to set a performance floor that the CNN/LSTM must beat.

Usage
-----
    from src.models.baseline import RandomForestModel, XGBoostModel

    rf = RandomForestModel()
    rf.fit(X_train, y_train)
    metrics = rf.evaluate(X_val, y_val)
    print(metrics)
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


def _make_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute the full evaluation metric suite. Called by both models."""
    return {
        "precision":         float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":            float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":                float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":           float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }


class RandomForestModel:
    """
    Random Forest classifier wrapped for this project's interface.

    Why Random Forest first?
    - Handles mixed feature scales without normalization
    - Built-in feature importance — tells us which of our 20 features matter
    - Robust to the class imbalance via class_weight='balanced'
    - Fast to train — good for rapid iteration
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 15,
        min_samples_leaf: int = 5,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),   # handles NaN folded features
            ("scaler",  StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight="balanced",
                n_jobs=n_jobs,
                random_state=random_state,
            )),
        ])
        self.feature_names_: Optional[list] = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RandomForestModel":
        """Train on a feature DataFrame."""
        self.feature_names_ = list(X.columns)
        self.pipeline.fit(X, y)
        logger.info(f"RandomForest trained on {len(X)} samples, {X.shape[1]} features")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """Return full metric dict. Primary metric is F1 (not accuracy)."""
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        metrics = _make_metrics(y, y_pred, y_prob)
        logger.info(f"RF  →  F1={metrics['f1']:.3f}  AUC={metrics['roc_auc']:.3f}  "
                    f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}")
        print(classification_report(y, y_pred, target_names=["No planet", "Planet"]))
        return metrics

    def feature_importances(self) -> pd.Series:
        """
        Return feature importances sorted descending.
        Only valid after fit(). Used in Day 12 analysis.
        """
        clf = self.pipeline.named_steps["clf"]
        return pd.Series(
            clf.feature_importances_,
            index=self.feature_names_,
        ).sort_values(ascending=False)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)
        logger.info(f"Model saved: {path}")

    @classmethod
    def load(cls, path: str) -> "RandomForestModel":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.pipeline = pickle.load(f)
        return obj


class XGBoostModel:
    """
    XGBoost classifier — typically outperforms Random Forest on tabular data
    by learning residuals iteratively (gradient boosting).

    Key difference from RF:
    - scale_pos_weight handles imbalance directly in the loss function
    - eval_metric='aucpr' optimizes the precision-recall curve (better for imbalance)
    - More hyperparameters to tune, but higher ceiling performance
    """

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 8.3,   # approximate neg/pos ratio in Kepler DR25
        random_state: int = 42,
    ):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("Run: pip install xgboost")

        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                use_label_encoder=False,
                random_state=random_state,
                verbosity=0,
            )),
        ])
        self.feature_names_: Optional[list] = None

    def fit(self, X: pd.DataFrame, y: np.ndarray,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[np.ndarray] = None) -> "XGBoostModel":
        """
        Train XGBoost. Optionally pass validation set for early stopping.
        """
        self.feature_names_ = list(X.columns)

        if X_val is not None and y_val is not None:
            self.pipeline.named_steps["clf"].set_params(early_stopping_rounds=30)
            # XGBoost needs raw arrays for eval_set, not piped
            X_imp = self.pipeline.named_steps["imputer"].fit_transform(X)
            X_val_imp = self.pipeline.named_steps["imputer"].transform(X_val)
            self.pipeline.named_steps["clf"].fit(
                X_imp, y,
                eval_set=[(X_val_imp, y_val)],
                verbose=False,
            )
        else:
            self.pipeline.fit(X, y)

        logger.info(f"XGBoost trained on {len(X)} samples, {X.shape[1]} features")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        metrics = _make_metrics(y, y_pred, y_prob)
        logger.info(f"XGB →  F1={metrics['f1']:.3f}  AUC={metrics['roc_auc']:.3f}  "
                    f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}")
        print(classification_report(y, y_pred, target_names=["No planet", "Planet"]))
        return metrics

    def feature_importances(self) -> pd.Series:
        clf = self.pipeline.named_steps["clf"]
        return pd.Series(
            clf.feature_importances_,
            index=self.feature_names_,
        ).sort_values(ascending=False)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.pipeline = pickle.load(f)
        return obj