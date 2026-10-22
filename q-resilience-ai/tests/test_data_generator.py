import pandas as pd

from src.data_generator import generate_microgrid_dataset, validate_microgrid_dataframe


def test_generate_microgrid_dataset():
    df = generate_microgrid_dataset(n_samples=120, seed=42)
    assert len(df) == 120
    assert set(df["is_abnormal"].unique()).issubset({0, 1})
    assert "demand_kw" in df.columns
    assert "scenario_type" in df.columns


def test_validate_microgrid_dataframe():
    df = generate_microgrid_dataset(n_samples=120, seed=42)
    valid, msg = validate_microgrid_dataframe(df)
    assert valid is True
    assert "valid" in msg.lower()
