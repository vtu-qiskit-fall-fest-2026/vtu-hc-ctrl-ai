"""
Unit tests for preprocessing and data-leakage prevention.
"""

import numpy as np
import pytest

from q_resilience_ai.src.data_generator import generate_microgrid_dataset
from q_resilience_ai.src.preprocessing import MicrogridPreprocessor, prepare_dataset_splits


def test_scaler_avoids_test_fitting():
    df = generate_microgrid_dataset(n_samples=60, seed=42)
    X_train, X_test, y_train, y_test, preprocessor = prepare_dataset_splits(df, test_size=0.25, seed=42)

    assert np.all(X_train >= -1e-6)
    assert np.all(X_train <= 2.0 * np.pi + 1e-6)
    assert X_train.shape[1] == 4
    assert X_test.shape[1] == 4
    assert len(X_train) + len(X_test) == len(df)


def test_preprocessor_requires_fit_before_transform():
    pre = MicrogridPreprocessor()
    df = generate_microgrid_dataset(n_samples=10, seed=1)
    with pytest.raises(RuntimeError):
        pre.transform(df)
