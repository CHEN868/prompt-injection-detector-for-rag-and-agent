"""Strict XGBoost risk-fusion model and artifact validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .context_feature_builder import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from .transformer_predictor import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION


DEFAULT_MODEL_PATH = Path("models/xgboost_risk_model.json")
DEFAULT_METADATA_PATH = Path("models/xgboost_risk_model.meta.json")


class RiskModelLoadError(RuntimeError):
    """Raised when XGBoost artifacts do not satisfy the feature contract."""


class XGBoostRiskModel:
    model_status = "xgboost_loaded"

    def __init__(self, model: Any, metadata: dict[str, Any]) -> None:
        self.model = model
        self.metadata = metadata
        self.feature_names = list(metadata["feature_names"])

    def predict_proba(self, features: dict[str, Any]) -> float:
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"Missing XGBoost features: {missing}")
        row = [[float(features[name]) for name in self.feature_names]]
        return float(self.model.predict_proba(row)[0][1])


def _validate_metadata(metadata: dict[str, Any], model: Any) -> None:
    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise RiskModelLoadError("XGBoost feature schema version does not match runtime.")
    if metadata.get("feature_names") != FEATURE_NAMES:
        raise RiskModelLoadError("XGBoost feature names/order do not match runtime contract.")
    if int(getattr(model, "n_features_in_", -1)) != len(FEATURE_NAMES):
        raise RiskModelLoadError(
            f"XGBoost model expects {getattr(model, 'n_features_in_', '?')} features; "
            f"runtime provides {len(FEATURE_NAMES)}."
        )
    if metadata.get("transformer_model_id") != DEFAULT_MODEL_ID:
        raise RiskModelLoadError("XGBoost artifact was trained with a different Transformer model.")
    if metadata.get("transformer_model_revision") != DEFAULT_MODEL_REVISION:
        raise RiskModelLoadError("XGBoost artifact Transformer revision does not match runtime.")


def load_risk_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> XGBoostRiskModel:
    model_file = Path(model_path)
    metadata_file = Path(metadata_path)
    if not model_file.exists() or not metadata_file.exists():
        raise RiskModelLoadError(
            "XGBoost model or metadata is missing. Run: python -m src.prepare_models"
        )
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RiskModelLoadError("Install xgboost before loading the risk model.") from error

    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        model = XGBClassifier()
        model.load_model(str(model_file))
        _validate_metadata(metadata, model)
    except RiskModelLoadError:
        raise
    except Exception as error:
        raise RiskModelLoadError(f"Cannot load XGBoost artifacts: {error}") from error
    return XGBoostRiskModel(model=model, metadata=metadata)


def train_xgboost_model(
    rows: list[dict[str, Any]],
    labels: list[int],
    feature_names: list[str] | None = None,
) -> Any:
    from xgboost import XGBClassifier

    names = feature_names or FEATURE_NAMES
    matrix = [[float(row[name]) for name in names] for row in rows]
    positive = max(1, sum(labels))
    negative = max(1, len(labels) - sum(labels))
    model = XGBClassifier(
        n_estimators=160,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=negative / positive,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )
    model.fit(matrix, labels)
    return model
