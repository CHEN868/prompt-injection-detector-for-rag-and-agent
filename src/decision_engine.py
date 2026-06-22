"""Decision engine for context-aware Prompt Injection risk results."""

from __future__ import annotations


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def decision_from_probability(probability: float, rule_block: bool = False) -> str:
    if rule_block:
        return "BLOCK"
    probability = clamp_probability(probability)
    if probability >= 0.75:
        return "BLOCK"
    if probability >= 0.45:
        return "WARN"
    return "ALLOW"


def decide_risk(probability: float, rule_block: bool = False) -> dict[str, str | float]:
    probability = clamp_probability(probability)
    return {
        "decision": decision_from_probability(probability, rule_block=rule_block),
        "risk_probability": probability,
    }
