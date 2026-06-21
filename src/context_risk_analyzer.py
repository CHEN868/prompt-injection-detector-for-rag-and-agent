"""Context-aware chunk analysis and request-level risk aggregation."""

from __future__ import annotations

from typing import Any

from .context_feature_builder import build_chunk_features
from .context_models import ChunkRiskResult, ContextChunk, ContextRiskResult, MatchedRule
from .context_rule_detector import detect_context_rules
from .decision_engine import decide_risk
from .model_runtime import get_runtime


HARMFUL_ATTACK_TYPES = {
    "instruction_override",
    "system_prompt_leakage",
    "jailbreak_roleplay",
    "policy_bypass",
    "forced_compliance",
    "sensitive_info_request",
    "obfuscated_instruction",
    "tool_call_hijacking",
    "indirect_prompt_injection",
    "semantic_prompt_injection",
}

ROLE_NAMES = {
    "user_input": "用户输入",
    "retrieved_doc": "检索文档",
    "tool_output": "工具返回内容",
    "tool_args": "工具调用参数",
    "chat_history": "历史对话",
}

ATTACK_NAMES = {
    "instruction_override": "指令覆盖",
    "system_prompt_leakage": "系统提示词泄露",
    "jailbreak_roleplay": "角色扮演越狱",
    "policy_bypass": "安全策略绕过",
    "forced_compliance": "强制服从",
    "sensitive_info_request": "敏感信息请求",
    "obfuscated_instruction": "混淆或编码指令",
    "tool_call_hijacking": "工具调用劫持",
    "indirect_prompt_injection": "间接 Prompt Injection",
    "semantic_prompt_injection": "语义 Prompt Injection",
}


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _rules_to_models(rule_result: dict[str, Any]) -> list[MatchedRule]:
    return [
        MatchedRule(
            attack_type=str(rule.get("attack_type", "")),
            name=str(rule.get("name", "")),
            score=int(rule.get("score", 0)),
            severity=str(rule.get("severity", "medium")),
            evidence=[str(item) for item in rule.get("evidence", [])],
        )
        for rule in rule_result.get("matched_rules", [])
    ]


def _should_mark_indirect(chunk: ContextChunk, attack_types: list[str]) -> bool:
    if chunk.context_role not in {"retrieved_doc", "tool_output"}:
        return False
    return bool(
        {
            "instruction_override",
            "system_prompt_leakage",
            "policy_bypass",
            "forced_compliance",
            "tool_call_hijacking",
        }
        & set(attack_types)
    )


def _role_bonus(chunk: ContextChunk, attack_types: list[str]) -> int:
    if not HARMFUL_ATTACK_TYPES & set(attack_types):
        return 0
    if chunk.context_role == "retrieved_doc":
        return 25 if {"instruction_override", "system_prompt_leakage"} & set(attack_types) else 15
    if chunk.context_role in {"tool_output", "tool_args"}:
        return 35 if "tool_call_hijacking" in attack_types else 20
    if chunk.context_role == "chat_history":
        return 5
    return 0


def _trust_penalty(chunk: ContextChunk, has_risk: bool) -> int:
    if not has_risk:
        return 0
    if chunk.source_trust < 0.3:
        return 15
    if chunk.source_trust < 0.6:
        return 8
    return 0


def _permission_bonus(chunk: ContextChunk, attack_types: list[str]) -> int:
    if chunk.context_role not in {"tool_output", "tool_args"} or not attack_types:
        return 0
    score = {"none": 0, "low": 0, "medium": 8, "high": 16, "critical": 25}.get(
        chunk.permission_level or "none", 0
    )
    if "tool_call_hijacking" in attack_types:
        score += 20
    if "sensitive_info_request" in attack_types:
        score += 10
    return score


def _build_reason(
    chunk: ContextChunk,
    attack_types: list[str],
    evidence: list[str],
    decision: str,
    decision_source: str,
    transformer_prob: float | None,
    xgboost_prob: float | None,
) -> str:
    role_name = ROLE_NAMES.get(chunk.context_role, chunk.context_role)
    if decision_source == "hard_rule":
        names = "、".join(ATTACK_NAMES.get(item, item) for item in attack_types)
        evidence_text = "、".join(evidence[:3]) or "无明确片段"
        return f"{role_name}命中高危硬规则（{names}），证据：{evidence_text}，直接 BLOCK。"
    if not attack_types and decision == "ALLOW":
        return (
            f"{role_name}未命中高危规则；Transformer={transformer_prob:.3f}，"
            f"XGBoost={xgboost_prob:.3f}，最终允许。"
        )
    names = "、".join(ATTACK_NAMES.get(item, item) for item in attack_types) or "语义风险"
    return (
        f"{role_name}检测到{names}；Transformer={transformer_prob:.3f}，"
        f"XGBoost={xgboost_prob:.3f}，决策为 {decision}。"
    )


def analyze_chunk(chunk: ContextChunk, runtime: Any | None = None) -> ChunkRiskResult:
    """Run rule-first detection, then semantic and XGBoost fusion when needed."""
    rule_result = detect_context_rules(chunk.content)
    matched_rules = _rules_to_models(rule_result)
    attack_types = _unique(list(rule_result.get("attack_types", [])))
    if _should_mark_indirect(chunk, attack_types):
        attack_types.insert(0, "indirect_prompt_injection")
    evidence = _unique([str(item) for item in rule_result.get("evidence", [])])

    base_score = int(rule_result.get("rule_score", 0))
    has_risk = bool(HARMFUL_ATTACK_TYPES & set(attack_types))
    role_bonus = _role_bonus(chunk, attack_types)
    trust_penalty = _trust_penalty(chunk, has_risk)
    permission_bonus = _permission_bonus(chunk, attack_types)
    context_bonus = role_bonus + trust_penalty + permission_bonus

    if bool(rule_result.get("rule_block", False)):
        probability = max(0.95, min(1.0, (base_score + context_bonus) / 100.0))
        state = decide_risk(probability, rule_block=True)
        transformer_prob = None
        xgboost_prob = None
        transformer_status = "skipped_rule_block"
        risk_model_status = "skipped_rule_block"
        decision_source = "hard_rule"
    else:
        active_runtime = runtime or get_runtime()
        transformer_prediction = active_runtime.transformer.predict_chunk(chunk)
        transformer_prob = transformer_prediction.transformer_prob
        features = build_chunk_features(
            chunk=chunk,
            rule_result=rule_result,
            transformer_prob=transformer_prob,
            attack_types=attack_types,
        )
        xgboost_prob = active_runtime.risk_model.predict_proba(features)
        state = decide_risk(xgboost_prob)
        transformer_status = transformer_prediction.model_status
        risk_model_status = active_runtime.risk_model.model_status
        decision_source = "xgboost"

    decision = str(state["decision"])
    if decision_source == "xgboost" and decision != "ALLOW" and not attack_types:
        attack_types.append("semantic_prompt_injection")
    return ChunkRiskResult(
        chunk_id=chunk.chunk_id,
        context_role=chunk.context_role,
        source=chunk.source,
        source_trust=chunk.source_trust,
        risk_score=int(state["risk_score"]),
        final_risk_probability=float(state["risk_probability"]),
        risk_level=str(state["risk_level"]),
        decision=decision,
        rule_block=bool(rule_result.get("rule_block", False)),
        rule_score=base_score,
        matched_rule_count=int(rule_result.get("matched_rule_count", len(matched_rules))),
        attack_types=attack_types,
        evidence=evidence,
        reason=_build_reason(
            chunk,
            attack_types,
            evidence,
            decision,
            decision_source,
            transformer_prob,
            xgboost_prob,
        ),
        base_score=base_score,
        context_bonus=context_bonus,
        source_trust_penalty=trust_penalty,
        permission_bonus=permission_bonus,
        transformer_prob=transformer_prob,
        transformer_model_status=transformer_status,
        xgboost_prob=xgboost_prob,
        risk_model_status=risk_model_status,
        decision_source=decision_source,
        matched_rules=matched_rules,
        tool_name=chunk.tool_name,
        permission_level=chunk.permission_level,
        metadata=chunk.metadata,
    )


def aggregate_context_risk(
    chunk_results: list[ChunkRiskResult],
    request_id: str | None = None,
) -> ContextRiskResult:
    """Use the highest-risk chunk and preserve hard-rule precedence."""
    if not chunk_results:
        return ContextRiskResult(
            final_decision="ALLOW",
            final_risk_score=0,
            final_risk_probability=0.0,
            risk_level="SAFE",
            summary="没有可扫描的上下文片段。",
            request_id=request_id,
        )

    primary = max(chunk_results, key=lambda item: item.final_risk_probability)
    hard_rule_block = any(item.rule_block for item in chunk_results)
    state = decide_risk(primary.final_risk_probability, rule_block=hard_rule_block)
    risky_chunks = [item for item in chunk_results if item.decision != "ALLOW"]

    if not risky_chunks:
        summary = "上下文中未发现明显 Prompt Injection、间接注入或工具调用劫持风险。"
    else:
        attacks = _unique([attack for item in risky_chunks for attack in item.attack_types])
        attack_text = "、".join(ATTACK_NAMES.get(item, item) for item in attacks[:4]) or "语义注入"
        summary = (
            f"检测到 {len(risky_chunks)} 个风险片段，主要风险来自 "
            f"{primary.context_role} / {primary.chunk_id}，风险类型：{attack_text}。"
        )

    return ContextRiskResult(
        final_decision=str(state["decision"]),
        final_risk_score=int(state["risk_score"]),
        final_risk_probability=float(state["risk_probability"]),
        risk_level=str(state["risk_level"]),
        summary=summary,
        primary_risk_chunk_id=primary.chunk_id,
        primary_context_role=primary.context_role,
        primary_risk_source=primary.context_role,
        risky_chunk_count=len(risky_chunks),
        chunk_results=chunk_results,
        safe_chunks=[item.chunk_id for item in chunk_results if item.decision == "ALLOW"],
        blocked_chunks=[item.chunk_id for item in chunk_results if item.decision == "BLOCK"],
        request_id=request_id,
        decision_source="hard_rule" if hard_rule_block else "aggregate",
    )
