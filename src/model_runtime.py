"""Shared, strictly validated model runtime for CLI, API, and Streamlit."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from .transformer_predictor import TransformerPredictor, load_model
from .xgboost_risk_model import XGBoostRiskModel, load_risk_model


@dataclass
class ModelRuntime:
    transformer: TransformerPredictor
    risk_model: XGBoostRiskModel

    @property
    def ready(self) -> bool:
        return True

    def status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "transformer": {
                "status": self.transformer.model_status,
                "model_id": self.transformer.model_name_or_path,
                "revision": self.transformer.revision,
                "device": self.transformer.device,
            },
            "xgboost": {
                "status": self.risk_model.model_status,
                "feature_schema_version": self.risk_model.metadata["feature_schema_version"],
                "metrics": self.risk_model.metadata.get("metrics", {}),
            },
        }


_runtime: ModelRuntime | None = None
_runtime_error: Exception | None = None
_lock = Lock()


def load_runtime(*, allow_download: bool = False) -> ModelRuntime:
    # On macOS/Python 3.13, loading XGBoost's native runtime after PyTorch can
    # crash the process because of conflicting native thread runtimes.
    risk_model = load_risk_model()
    transformer = load_model(local_files_only=not allow_download)
    return ModelRuntime(transformer=transformer, risk_model=risk_model)


def get_runtime() -> ModelRuntime:
    global _runtime, _runtime_error
    if _runtime is not None:
        return _runtime
    with _lock:
        if _runtime is not None:
            return _runtime
        try:
            _runtime = load_runtime(allow_download=False)
            _runtime_error = None
        except Exception as error:
            _runtime_error = error
            raise
    return _runtime


def runtime_error() -> Exception | None:
    return _runtime_error


def set_runtime_for_testing(runtime: ModelRuntime | None) -> None:
    global _runtime, _runtime_error
    _runtime = runtime
    _runtime_error = None
