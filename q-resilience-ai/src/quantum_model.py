"""
Quantum Machine Learning Engine
"""

import time
from typing import Any, Dict, Tuple

import numpy as np

try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    QISKIT_AVAILABLE = True
    QISKIT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    QISKIT_AVAILABLE = False
    QISKIT_IMPORT_ERROR = str(exc)


def build_zz_feature_map(num_qubits: int = 4, reps: int = 2, entanglement: str = "linear"):
    if not QISKIT_AVAILABLE:
        raise ImportError(f"Qiskit Machine Learning is required. Error: {QISKIT_IMPORT_ERROR}")
    return ZZFeatureMap(feature_dimension=num_qubits, reps=reps, entanglement=entanglement, insert_barriers=True)


def get_circuit_metadata(circuit) -> Dict[str, Any]:
    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "total_gates": sum(circuit.count_ops().values()),
        "gate_counts": dict(circuit.count_ops()),
        "ascii_diagram": str(circuit.draw(output="text")),
    }


def train_quantum_qsvc(X_train: np.ndarray, y_train: np.ndarray, num_qubits: int = 4, reps: int = 2, C: float = 1.0) -> Tuple[Any, float, Dict[str, Any]]:
    if not QISKIT_AVAILABLE:
        raise RuntimeError(f"Cannot train quantum classifier: Qiskit unavailable: {QISKIT_IMPORT_ERROR}")

    feature_map = build_zz_feature_map(num_qubits=num_qubits, reps=reps)
    circuit_meta = get_circuit_metadata(feature_map)
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    qsvc = QSVC(quantum_kernel=kernel, C=C)

    t0 = time.perf_counter()
    qsvc.fit(X_train, y_train)
    t1 = time.perf_counter()
    return qsvc, (t1 - t0) * 1000.0, circuit_meta


def predict_quantum(qsvc: Any, X_test: np.ndarray):
    t0 = time.perf_counter()
    preds = qsvc.predict(X_test)
    t1 = time.perf_counter()
    return preds, (t1 - t0) * 1000.0
