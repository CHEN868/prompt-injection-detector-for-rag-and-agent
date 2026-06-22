"""Compress raw context metadata into one bounded risk-prior score."""

from __future__ import annotations

from dataclasses import dataclass


EXTERNAL_SOURCES = {"external_web", "web", "unknown", "tool", "untrusted"}
PERMISSION_VALUES = {
    "none": 0.0,
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "critical": 1.0,
}


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def encode_permission(permission_level: str | None) -> float:
    return PERMISSION_VALUES.get(permission_level or "none", 0.0)


def is_external_source(source: str, source_trust: float) -> bool:
    return source in EXTERNAL_SOURCES or source_trust < 0.5


def history_count_to_score(history_risk_count: int) -> float:
    return clamp_score(history_risk_count / 3.0)


@dataclass(frozen=True)
class ContextRiskInput:
    source_trust: float
    permission_level: str | None
    is_external_source: bool
    is_tool_output: bool
    history_risk_score: float


class ContextRiskAggregator:
    """Produce a context prior without inspecting rules or text semantics."""

    TRUST_WEIGHT = 0.35
    PERMISSION_WEIGHT = 0.25
    EXTERNAL_WEIGHT = 0.15
    TOOL_OUTPUT_WEIGHT = 0.15
    HISTORY_WEIGHT = 0.10

    def aggregate(self, signals: ContextRiskInput) -> float:
        untrusted_source = 1.0 - clamp_score(signals.source_trust)
        score = (
            untrusted_source * self.TRUST_WEIGHT
            + encode_permission(signals.permission_level) * self.PERMISSION_WEIGHT
            + float(signals.is_external_source) * self.EXTERNAL_WEIGHT
            + float(signals.is_tool_output) * self.TOOL_OUTPUT_WEIGHT
            + clamp_score(signals.history_risk_score) * self.HISTORY_WEIGHT
        )
        return clamp_score(score)

    def explain(self, signals: ContextRiskInput, score: float) -> str:
        return (
            f"context_risk_score={score:.3f}; source_trust={signals.source_trust:.2f}, "
            f"permission={signals.permission_level or 'none'}, "
            f"external={signals.is_external_source}, tool_output={signals.is_tool_output}, "
            f"history_risk={signals.history_risk_score:.2f}。"
        )


def context_input_from_chunk(chunk) -> ContextRiskInput:
    return ContextRiskInput(
        source_trust=clamp_score(chunk.source_trust),
        permission_level=chunk.permission_level,
        is_external_source=is_external_source(chunk.source, chunk.source_trust),
        is_tool_output=chunk.context_role == "tool_output",
        history_risk_score=history_count_to_score(chunk.history_risk_count),
    )
