"""
Evaluation and benchmarking engine.
"""

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, train_time_ms: float = 0.0, inference_time_ms: float = 0.0) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "confusion_matrix": cm.tolist(),
        "train_time_ms": round(float(train_time_ms), 2),
        "inference_time_ms": round(float(inference_time_ms), 2),
    }


def run_low_data_benchmark(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, fractions: List[float] | None = None, seed: int = 42) -> List[Dict[str, Any]]:
    fractions = fractions or [0.25, 0.50, 0.75, 1.0]
    results = []

    from q_resilience_ai.src.classical_model import train_classical_svc, predict_classical
    from q_resilience_ai.src.quantum_model import QISKIT_AVAILABLE, predict_quantum, train_quantum_qsvc

    for frac in fractions:
        n_sub = max(6, int(len(X_train) * frac))
        sub_X = X_train[:n_sub]
        sub_y = y_train[:n_sub]

        c_model, c_train_t = train_classical_svc(sub_X, sub_y, seed=seed)
        c_preds, _, c_inf_t = predict_classical(c_model, X_test)
        c_eval = evaluate_predictions(y_test, c_preds, c_train_t, c_inf_t)

        q_eval = None
        if QISKIT_AVAILABLE:
            try:
                q_model, q_train_t, _ = train_quantum_qsvc(sub_X, sub_y, num_qubits=sub_X.shape[1], reps=2)
                q_preds, q_inf_t = predict_quantum(q_model, X_test)
                q_eval = evaluate_predictions(y_test, q_preds, q_train_t, q_inf_t)
            except Exception as exc:  # pragma: no cover
                q_eval = {"error": str(exc)}

        results.append({
            "fraction": frac,
            "sample_count": n_sub,
            "classical": c_eval,
            "quantum": q_eval,
        })

    return results
