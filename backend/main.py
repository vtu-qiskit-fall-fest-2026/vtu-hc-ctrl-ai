"""
FastAPI backend for Q-Resilience AI.
Provides health check, benchmark, dataset generation, and prediction endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import qiskit
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from q_resilience_ai.src.data_generator import generate_microgrid_dataset
from q_resilience_ai.src.preprocessing import prepare_dataset_splits
from q_resilience_ai.src.classical_model import train_classical_svc, predict_classical
from q_resilience_ai.src.quantum_model import train_quantum_qsvc, predict_quantum, QISKIT_AVAILABLE

app = FastAPI(title="Q-RESILIENCE AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SensorInput(BaseModel):
    demand_kw: float = Field(..., ge=0)
    solar_kw: float = Field(..., ge=0)
    battery_soc_pct: float = Field(..., ge=0, le=100)
    voltage_dev_v: float = Field(...)
    frequency_dev_hz: float = Field(...)


class DatasetGenerateRequest(BaseModel):
    n_samples: int = 120
    seed: int = 42
    abnormal_ratio: float = 0.35


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "Q-RESILIENCE AI Backend",
        "qiskit_available": QISKIT_AVAILABLE,
        "qiskit_version": qiskit.__version__ if QISKIT_AVAILABLE else None,
        "models_ready": True,
        "dataset_samples": 120,
    }


@app.get("/api/dataset")
def get_dataset(limit: int = 25, seed: int = 42) -> Dict[str, Any]:
    df = generate_microgrid_dataset(n_samples=max(10, min(limit, 200)), seed=seed, abnormal_ratio=0.35)
    return {"samples": df.head(limit).to_dict(orient="records")}


@app.post("/api/predict")
def predict_telemetry(inputs: SensorInput) -> Dict[str, Any]:
    features = np.array([
        [
            inputs.demand_kw,
            inputs.solar_kw,
            inputs.battery_soc_pct,
            inputs.voltage_dev_v,
            inputs.frequency_dev_hz,
        ]
    ], dtype=float)

    df = generate_microgrid_dataset(n_samples=100, seed=42)
    X_train, X_test, y_train, y_test, preprocessor = prepare_dataset_splits(df, test_size=0.25, seed=42)

    model, _ = train_classical_svc(X_train, y_train)
    preds, probs, inf_time = predict_classical(model, features)

    classical_label = int(preds[0])
    classical_probability = float(probs[0]) if probs is not None else 0.0

    quantum_result = None
    if QISKIT_AVAILABLE:
        try:
            q_model, _, _ = train_quantum_qsvc(X_train, y_train)
            q_preds, q_time = predict_quantum(q_model, features)
            quantum_result = {
                "model_name": "Quantum QSVC",
                "prediction": "ABNORMAL" if int(q_preds[0]) == 1 else "NORMAL",
                "label": int(q_preds[0]),
                "inference_time_ms": round(float(q_time), 3),
            }
        except Exception:
            quantum_result = None

    heuristic = {
        "is_physically_abnormal": abs(inputs.voltage_dev_v) > 6.0 or abs(inputs.frequency_dev_hz) > 0.2 or (inputs.demand_kw - (inputs.solar_kw + max(0, (inputs.battery_soc_pct - 20) * 1.5))) > 30.0,
        "voltage_status": "critical" if abs(inputs.voltage_dev_v) > 6.0 else "warning" if abs(inputs.voltage_dev_v) > 3.0 else "nominal",
        "frequency_status": "critical" if abs(inputs.frequency_dev_hz) > 0.2 else "warning" if abs(inputs.frequency_dev_hz) > 0.05 else "nominal",
    }

    return {
        "inputs": {
            "demand_kw": inputs.demand_kw,
            "solar_kw": inputs.solar_kw,
            "battery_soc_pct": inputs.battery_soc_pct,
            "voltage_dev_v": inputs.voltage_dev_v,
            "frequency_dev_hz": inputs.frequency_dev_hz,
        },
        "classical_svc": {
            "model_name": "Classical SVC (RBF)",
            "prediction": "ABNORMAL" if classical_label == 1 else "NORMAL",
            "label": classical_label,
            "probability_abnormal": round(float(classical_probability), 4),
            "inference_time_ms": round(float(inf_time), 3),
        },
        "quantum_qsvc": quantum_result,
        "heuristic_diagnosis": heuristic,
    }


@app.get("/api/benchmark")
def benchmark() -> Dict[str, Any]:
    df = generate_microgrid_dataset(n_samples=120, seed=42, abnormal_ratio=0.35)
    X_train, X_test, y_train, y_test, _ = prepare_dataset_splits(df, test_size=0.25, seed=42)

    c_model, c_train_t = train_classical_svc(X_train, y_train)
    c_preds, _, c_inf_t = predict_classical(c_model, X_test)
    c_accuracy = accuracy_score(y_test, c_preds)
    c_precision = precision_score(y_test, c_preds, zero_division=0)
    c_recall = recall_score(y_test, c_preds, zero_division=0)
    c_f1 = f1_score(y_test, c_preds, zero_division=0)
    c_cm = confusion_matrix(y_test, c_preds, labels=[0, 1]).tolist()

    q_result = None
    if QISKIT_AVAILABLE:
        try:
            q_model, q_train_t, _ = train_quantum_qsvc(X_train, y_train)
            q_preds, q_inf_t = predict_quantum(q_model, X_test)
            q_result = {
                "accuracy": round(float(accuracy_score(y_test, q_preds)), 4),
                "precision": round(float(precision_score(y_test, q_preds, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, q_preds, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, q_preds, zero_division=0)), 4),
                "confusion_matrix": confusion_matrix(y_test, q_preds, labels=[0, 1]).tolist(),
                "train_time_ms": round(float(q_train_t), 3),
                "inference_time_ms": round(float(q_inf_t), 3),
            }
        except Exception as exc:
            q_result = {"error": str(exc)}

    return {
        "classical_svc": {
            "model_name": "Classical SVC (RBF)",
            "accuracy": round(float(c_accuracy), 4),
            "precision": round(float(c_precision), 4),
            "recall": round(float(c_recall), 4),
            "f1_score": round(float(c_f1), 4),
            "confusion_matrix": c_cm,
            "train_time_ms": round(float(c_train_t), 3),
            "inference_time_ms": round(float(c_inf_t), 3),
        },
        "quantum_qsvc": q_result,
    }
