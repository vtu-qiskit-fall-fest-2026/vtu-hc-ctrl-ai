"""
Unit tests for evaluation metrics and confusion matrix.
"""

import numpy as np

from q_resilience_ai.src.evaluation import evaluate_predictions


def test_metric_calculation():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    res = evaluate_predictions(y_true, y_pred, train_time_ms=10.5, inference_time_ms=2.1)

    assert res["true_positives"] == 2
    assert res["false_positives"] == 1
    assert res["true_negatives"] == 2
    assert res["false_negatives"] == 1
    assert abs(res["accuracy"] - (4 / 6)) < 1e-3
