"""Build the minimal V3 feature contract used by XGBoost."""

from __future__ import annotations

from typing import Any

from .context_risk_aggregator import clamp_score, encode_permission


FEATURE_SCHEMA_VERSION = "context-risk-v3"
FEATURE_NAMES = [
    "transformer_prob",
    "rule_score",
    "context_risk_score",
    "source_trust_encoded",
    "permission_level_encoded",
]

MAX_FEATURE_COUNT = 6
FORBIDDEN_FEATURE_NAMES = {
    "rule_block",
    "matched_rule_count",
    "has_instruction_override",
    "has_system_prompt_leakage",
    "has_tool_hijack",
    "is_external_source",
    "is_tool_output",
    "history_risk_count",
    "context_role",
    "scenario",
}


def validate_feature_contract(feature_names: list[str] | None = None) -> None:
    names = feature_names or FEATURE_NAMES
    if len(names) > MAX_FEATURE_COUNT:
        raise ValueError(f"XGBoost feature count must be <= {MAX_FEATURE_COUNT}; got {len(names)}.")
    leaked = sorted(set(names) & FORBIDDEN_FEATURE_NAMES)
    if leaked:
        raise ValueError(f"Forbidden rule/context features leaked into XGBoost: {leaked}")
    if names != FEATURE_NAMES:
        raise ValueError(f"Unexpected XGBoost feature contract: {names}")


def build_chunk_features(
    *,
    transformer_prob: float,
    rule_score: float,
    context_risk_score: float,
    source_trust: float,
    permission_level: str | None,
) -> dict[str, float]:
    """Assemble five already-compressed signals; do not derive policy here."""
    validate_feature_contract()
    features = {
        "transformer_prob": clamp_score(transformer_prob),
        "rule_score": clamp_score(rule_score),
        "context_risk_score": clamp_score(context_risk_score),
        "source_trust_encoded": clamp_score(source_trust),
        "permission_level_encoded": encode_permission(permission_level),
    }
    return {name: features[name] for name in FEATURE_NAMES}


def features_for_csv(features: dict[str, Any]) -> dict[str, float]:
    validate_feature_contract(list(features))
    return {name: float(features[name]) for name in FEATURE_NAMES}
