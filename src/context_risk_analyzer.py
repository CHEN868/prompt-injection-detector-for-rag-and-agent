"""V3 orchestration: independent rule, semantic, context, and fusion layers."""

from __future__ import annotations

from typing import Any

from .context_feature_builder import build_chunk_features
from .context_models import ChunkRiskResult, ContextChunk, ContextRiskResult, SignalExplanation
from .context_risk_aggregator import ContextRiskAggregator, context_input_from_chunk
from .context_rule_detector import detect_rule_signal
from .decision_engine import decide_risk
from .model_runtime import get_runtime


CONTEXT_AGGREGATOR = ContextRiskAggregator()


def _rule_explanation(rule_block: bool, rule_score: float) -> str:
    if rule_block:
        return f"高置信硬规则命中，rule_score={rule_score:.3f}，规则层直接要求 BLOCK。"
    return f"未触发规则硬阻断，rule_score={rule_score:.3f}。"


def _semantic_explanation(transformer_prob: float) -> str:
    if transformer_prob >= 0.75:
        level = "高"
    elif transformer_prob >= 0.45:
        level = "中"
    else:
        level = "低"
    return f"Transformer 语义注入概率为 {transformer_prob:.3f}，语义风险为{level}。"


def analyze_chunk(chunk: ContextChunk, runtime: Any | None = None) -> ChunkRiskResult:
    """Run four isolated layers and return the uniform V3 signal envelope."""
    active_runtime = runtime or get_runtime()

    rule_signal = detect_rule_signal(chunk.content)
    rule_block = bool(rule_signal["rule_block"])
    rule_score = float(rule_signal["rule_score"])

    transformer_prediction = active_runtime.transformer.predict_chunk(chunk)
    transformer_prob = float(transformer_prediction.transformer_prob)

    context_input = context_input_from_chunk(chunk)
    context_risk_score = CONTEXT_AGGREGATOR.aggregate(context_input)

    features = build_chunk_features(
        transformer_prob=transformer_prob,
        rule_score=rule_score,
        context_risk_score=context_risk_score,
        source_trust=chunk.source_trust,
        permission_level=chunk.permission_level,
    )
    final_probability = active_runtime.risk_model.predict_proba(features)
    state = decide_risk(final_probability, rule_block=rule_block)

    return ChunkRiskResult(
        chunk_id=chunk.chunk_id,
        context_role=chunk.context_role,
        source=chunk.source,
        rule_block=rule_block,
        rule_score=rule_score,
        transformer_prob=transformer_prob,
        context_risk_score=context_risk_score,
        final_risk_probability=float(state["risk_probability"]),
        decision=str(state["decision"]),
        explanation=SignalExplanation(
            rule_signal=_rule_explanation(rule_block, rule_score),
            semantic_signal=_semantic_explanation(transformer_prob),
            context_signal=CONTEXT_AGGREGATOR.explain(context_input, context_risk_score),
        ),
    )


def aggregate_context_risk(
    chunk_results: list[ChunkRiskResult],
    request_id: str | None = None,
) -> ContextRiskResult:
    """Aggregate chunks without creating new model features or policy bonuses."""
    if not chunk_results:
        explanation = SignalExplanation(
            rule_signal="没有可扫描片段，规则信号为 0。",
            semantic_signal="没有可扫描片段，语义信号为 0。",
            context_signal="没有可扫描片段，上下文信号为 0。",
        )
        return ContextRiskResult(
            rule_block=False,
            rule_score=0.0,
            transformer_prob=0.0,
            context_risk_score=0.0,
            final_risk_probability=0.0,
            decision="ALLOW",
            explanation=explanation,
            summary="没有可扫描的上下文片段。",
            request_id=request_id,
        )

    rule_blockers = [item for item in chunk_results if item.rule_block]
    if rule_blockers:
        primary = max(rule_blockers, key=lambda item: item.rule_score)
    else:
        primary = max(chunk_results, key=lambda item: item.final_risk_probability)

    rule_block = bool(rule_blockers)
    max_rule_score = max(item.rule_score for item in chunk_results)
    max_transformer_prob = max(item.transformer_prob for item in chunk_results)
    final_probability = max(item.final_risk_probability for item in chunk_results)
    state = decide_risk(final_probability, rule_block=rule_block)
    risky_chunks = [item for item in chunk_results if item.decision != "ALLOW"]

    summary = (
        "上下文中未发现需要告警的 Prompt Injection 风险。"
        if not risky_chunks
        else (
            f"检测到 {len(risky_chunks)} 个风险片段，主要风险来自 "
            f"{primary.context_role} / {primary.chunk_id}。"
        )
    )
    return ContextRiskResult(
        rule_block=rule_block,
        rule_score=max_rule_score,
        transformer_prob=max_transformer_prob,
        context_risk_score=primary.context_risk_score,
        final_risk_probability=float(state["risk_probability"]),
        decision=str(state["decision"]),
        explanation=SignalExplanation(
            rule_signal=(
                f"{len(rule_blockers)} 个片段触发硬阻断；最大 rule_score={max_rule_score:.3f}。"
                if rule_blockers
                else f"没有片段触发硬阻断；最大 rule_score={max_rule_score:.3f}。"
            ),
            semantic_signal=f"所有片段最大 transformer_prob={max_transformer_prob:.3f}。",
            context_signal=(
                f"主要风险片段 context_risk_score={primary.context_risk_score:.3f}，"
                f"来源角色为 {primary.context_role}。"
            ),
        ),
        summary=summary,
        primary_risk_chunk_id=primary.chunk_id,
        primary_context_role=primary.context_role,
        primary_risk_source=primary.context_role,
        risky_chunk_count=len(risky_chunks),
        chunk_results=chunk_results,
        request_id=request_id,
    )
