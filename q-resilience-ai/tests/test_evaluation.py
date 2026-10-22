import numpy as np

from src.evaluation import evaluate_predictions


def test_evaluate_predictions_basic():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert "confusion_matrix" in metrics
