"""
Classical Machine Learning Baseline Models
"""

import time
from typing import Any, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def train_classical_svc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    gamma: str = "scale",
    probability: bool = True,
    seed: int = 42,
) -> Tuple[SVC, float]:
    t0 = time.perf_counter()
    model = SVC(C=C, kernel="rbf", gamma=gamma, probability=probability, random_state=seed)
    model.fit(X_train, y_train)
    t1 = time.perf_counter()
    return model, (t1 - t0) * 1000.0


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 50, seed: int = 42):
    t0 = time.perf_counter()
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=5, random_state=seed)
    model.fit(X_train, y_train)
    t1 = time.perf_counter()
    return model, (t1 - t0) * 1000.0


def predict_classical(model: Any, X_test: np.ndarray):
    t0 = time.perf_counter()
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    t1 = time.perf_counter()
    return preds, probs, (t1 - t0) * 1000.0
