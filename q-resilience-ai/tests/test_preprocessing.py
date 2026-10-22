import numpy as np

from src.preprocessing import prepare_dataset_splits
from src.data_generator import generate_microgrid_dataset


def test_prepare_dataset_splits_shapes():
    df = generate_microgrid_dataset(n_samples=80, seed=7)
    X_train, X_test, y_train, y_test, preprocessor = prepare_dataset_splits(df, seed=7)
    assert X_train.shape[0] == 60
    assert X_test.shape[0] == 20
    assert y_train.shape[0] == 60
    assert y_test.shape[0] == 20
    assert preprocessor.is_fitted is True
