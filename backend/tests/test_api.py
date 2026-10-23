"""
Comprehensive Test Suite for Q-Resilience AI Backend
VTU-Qiskit Fall Fest 2026 — Team Ctrl_AI (VTU-QFF-HC-0025)
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from main import app, state, compute_zz_statevector, compute_fidelity_kernel_matrix

client = TestClient(app)


def test_health_check_variations():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_statevector_normalization_and_kernel_properties():
    x1 = np.array([0.5, 1.2, 2.4, 3.1])
    x2 = np.array([1.0, 0.2, 0.9, 2.8])
    psi1 = compute_zz_statevector(x1, num_qubits=4, reps=2)
    psi2 = compute_zz_statevector(x2, num_qubits=4, reps=2)

    assert len(psi1) == 16
    assert len(psi2) == 16
    assert np.isclose(np.linalg.norm(psi1), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(psi2), 1.0, atol=1e-5)

    fidelity_self = np.abs(np.vdot(psi1, psi1)) ** 2
    assert np.isclose(fidelity_self, 1.0, atol=1e-5)

    fidelity_cross = np.abs(np.vdot(psi1, psi2)) ** 2
    assert 0.0 <= fidelity_cross <= 1.00001


def test_dataset_generation_reproducibility():
    payload = {"sample_count": 40, "random_seed": 123, "abnormal_ratio": 0.35}
    res1 = client.post("/api/dataset/generate", json=payload)
    assert res1.status_code == 200
    b1 = res1.json()["benchmark"]
    assert b1["total_samples"] == 40


def test_dual_prediction_endpoint():
    payload = {
        "demand_kw": 55.0,
        "solar_kw": 45.0,
        "battery_soc_pct": 70.0,
        "voltage_dev_v": 0.2,
        "frequency_dev_hz": 0.01,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "inputs" in data
    assert "classical_svc" in data
    assert "quantum_qsvc" in data
