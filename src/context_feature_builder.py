"""Build the versioned numeric feature contract used by XGBoost."""

from __future__ import annotations

from typing import Any


FEATURE_SCHEMA_VERSION = "context-risk-v1"
EXTERNAL_SOURCES = {"external_web", "web", "unknown", "tool", "untrusted"}
PERMISSION_VALUES = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
CONTEXT_ROLES = ("user_input", "retrieved_doc", "tool_output", "tool_args", "chat_history")
SCENARIOS = ("chat", "rag", "agent", "tool_call", "multi_turn")

BASE_FEATURE_NAMES = (
    "transformer_prob",
    "rule_block",
    "rule_score",
    "matched_rule_count",
    "has_instruction_override",
    "has_system_prompt_leakage",
    "has_tool_hijack",
    "source_trust",
    "permission_level_value",
    "history_risk_count",
    "is_external_source",
    "is_tool_output",
)
FEATURE_NAMES = [
    *BASE_FEATURE_NAMES,
    *(f"context_role__{role}" for role in CONTEXT_ROLES),
    *(f"scenario__{scenario}" for scenario in SCENARIOS),
]


def is_external_source(source: str, source_trust: float) -> bool:
    return source in EXTERNAL_SOURCES or source_trust < 0.5


def build_chunk_features(
    chunk: Any,
    rule_result: dict[str, Any],
    transformer_prob: float,
    attack_types: list[str] | None = None,
) -> dict[str, float]:
    """Convert a chunk and detector signals into the exact model feature schema."""
    attack_type_set = set(attack_types or rule_result.get("attack_types", []))
    permission_level = chunk.permission_level or "none"
    features: dict[str, float] = {
        "transformer_prob": float(transformer_prob),
        "rule_block": float(bool(rule_result.get("rule_block", False))),
        "rule_score": float(rule_result.get("rule_score", 0)) / 100.0,
        "matched_rule_count": float(rule_result.get("matched_rule_count", 0)),
        "has_instruction_override": float("instruction_override" in attack_type_set),
        "has_system_prompt_leakage": float("system_prompt_leakage" in attack_type_set),
        "has_tool_hijack": float("tool_call_hijacking" in attack_type_set),
        "source_trust": float(chunk.source_trust),
        "permission_level_value": float(PERMISSION_VALUES.get(permission_level, 0)),
        "history_risk_count": float(chunk.history_risk_count),
        "is_external_source": float(is_external_source(chunk.source, chunk.source_trust)),
        "is_tool_output": float(chunk.context_role == "tool_output"),
    }
    features.update(
        {f"context_role__{role}": float(chunk.context_role == role) for role in CONTEXT_ROLES}
    )
    features.update(
        {f"scenario__{scenario}": float(chunk.scenario == scenario) for scenario in SCENARIOS}
    )
    return {name: features[name] for name in FEATURE_NAMES}


def features_for_csv(features: dict[str, Any]) -> dict[str, float]:
    return {name: float(features[name]) for name in FEATURE_NAMES}
