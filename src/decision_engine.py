"""Decision engine for context-aware Prompt Injection risk results."""

from __future__ import annotations


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def probability_to_score(probability: float) -> int:
    return int(round(clamp_probability(probability) * 100))


def risk_level_from_probability(probability: float) -> str:
    score = probability_to_score(probability)
    if score >= 90:
        return "CRITICAL"
    if score >= 70:
        return "HIGH"
    if score >= 45:
        return "MEDIUM"
    if score >= 20:
        return "LOW"
    return "SAFE"


def decision_from_probability(probability: float, rule_block: bool = False) -> str:
    if rule_block:
        return "BLOCK"
    probability = clamp_probability(probability)
    if probability >= 0.75:
        return "BLOCK"
    if probability >= 0.45:
        return "REVIEW"
    return "ALLOW"


def decide_risk(probability: float, rule_block: bool = False) -> dict[str, str | int | float]:
    probability = clamp_probability(probability)
    return {
        "decision": decision_from_probability(probability, rule_block=rule_block),
        "risk_level": risk_level_from_probability(probability),
        "risk_score": probability_to_score(probability),
        "risk_probability": probability,
    }
