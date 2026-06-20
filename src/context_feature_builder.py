"""Build model features from context chunks and detector outputs."""

from __future__ import annotations

from typing import Any


EXTERNAL_SOURCES = {"external_web", "web", "unknown", "tool", "untrusted"}
PERMISSION_VALUES = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def is_external_source(source: str, source_trust: float) -> bool:
    return source in EXTERNAL_SOURCES or source_trust < 0.5


def build_chunk_features(
    chunk: Any,
    rule_result: dict[str, Any],
    transformer_prob: float,
    attack_types: list[str] | None = None,
) -> dict[str, Any]:
    """Convert chunk context and detector outputs into fusion features."""
    attack_type_set = set(attack_types or rule_result.get("attack_types", []))
    permission_level = chunk.permission_level or "none"
    return {
        "transformer_prob": float(transformer_prob),
        "rule_block": bool(rule_result.get("rule_block", False)),
        "rule_score": int(rule_result.get("rule_score", 0)),
        "matched_rule_count": int(rule_result.get("matched_rule_count", 0)),
        "has_instruction_override": "instruction_override" in attack_type_set,
        "has_system_prompt_leakage": "system_prompt_leakage" in attack_type_set,
        "has_tool_hijack": "tool_call_hijacking" in attack_type_set,
        "context_role": chunk.context_role,
        "scenario": chunk.scenario,
        "source_trust": float(chunk.source_trust),
        "permission_level": permission_level,
        "permission_level_value": PERMISSION_VALUES.get(permission_level, 0),
        "history_risk_count": int(chunk.history_risk_count),
        "is_external_source": is_external_source(chunk.source, chunk.source_trust),
        "is_tool_output": chunk.context_role == "tool_output",
    }


def features_for_csv(features: dict[str, Any]) -> dict[str, Any]:
    """Return a flat feature row suitable for CSV export and training."""
    row = dict(features)
    for key, value in list(row.items()):
        if isinstance(value, bool):
            row[key] = int(value)
    return row
