"""XGBoost risk fusion model with an explicit rules fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any


MODEL_STATUS_FALLBACK = "fallback_rules"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class FallbackRiskModel:
    """Deterministic fallback used when no trained XGBoost model is available."""

    model_status = MODEL_STATUS_FALLBACK

    def predict_proba(self, features: dict[str, Any]) -> float:
        probability = float(features.get("rule_score", 0)) / 100.0
        probability = max(probability, float(features.get("transformer_prob", 0.0)))

        if features.get("rule_block"):
            probability = max(probability, 0.85)

        has_rule_risk = int(features.get("rule_score", 0)) > 0
        if has_rule_risk and features.get("context_role") == "retrieved_doc":
            probability += 0.15
            if features.get("has_instruction_override") or features.get("has_system_prompt_leakage"):
                probability += 0.10

        if has_rule_risk and features.get("context_role") == "tool_output":
            probability += 0.20
            if features.get("has_tool_hijack"):
                probability += 0.15

        if has_rule_risk and features.get("context_role") == "tool_args":
            probability += 0.20
            if features.get("has_tool_hijack"):
                probability += 0.15

        if has_rule_risk and features.get("context_role") == "chat_history":
            probability += 0.05

        if has_rule_risk and float(features.get("source_trust", 0.5)) < 0.3:
            probability += 0.15
        elif has_rule_risk and float(features.get("source_trust", 0.5)) < 0.6:
            probability += 0.08

        if has_rule_risk:
            probability += min(int(features.get("history_risk_count", 0)), 3) * 0.03

        return _clamp(probability)


class XGBoostRiskModel:
    """Loaded XGBoost model wrapper."""

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        self.model = model
        self.feature_names = feature_names
        self.model_status = "xgboost_loaded"

    def predict_proba(self, features: dict[str, Any]) -> float:
        row = [[float(features.get(name, 0.0)) for name in self.feature_names]]
        probabilities = self.model.predict_proba(row)
        return float(probabilities[0][1])


def load_risk_model(
    model_path: str | Path = "models/xgboost_risk_model.json",
    feature_names: list[str] | None = None,
) -> XGBoostRiskModel | FallbackRiskModel:
    path = Path(model_path)
    if not path.exists():
        return FallbackRiskModel()

    try:
        from xgboost import XGBClassifier
    except ImportError:
        return FallbackRiskModel()

    model = XGBClassifier()
    model.load_model(str(path))
    return XGBoostRiskModel(model=model, feature_names=feature_names or [])


def train_xgboost_model(
    rows: list[dict[str, Any]],
    labels: list[int],
    feature_names: list[str],
) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RuntimeError(
            "xgboost is not installed. Install with: python -m pip install xgboost"
        ) from error

    matrix = [[float(row.get(name, 0.0)) for name in feature_names] for row in rows]
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss",
    )
    model.fit(matrix, labels)
    return model
