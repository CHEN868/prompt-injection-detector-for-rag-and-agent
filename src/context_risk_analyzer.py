"""Context-aware risk scoring for RAG/Agent prompt injection scanning."""
# 真正做规则匹配、上下文加分、风险评分
from __future__ import annotations

import re
from typing import Any

from .context_models import ChunkRiskResult, ContextChunk, ContextRiskResult, MatchedRule
from .context_feature_builder import build_chunk_features
from .context_rule_detector import detect_context_rules
from .decision_engine import decide_risk
from .transformer_predictor import predict_chunk
from .xgboost_risk_model import load_risk_model


RISK_MODEL = load_risk_model()

# 上下文监测规则
CONTEXT_RULES = [
    {
        "attack_type": "tool_call_hijacking",
        "name": "工具调用劫持",
        "score": 35,
        "severity": "high",
        "patterns": [
            r"call (the )?tool",
            r"invoke (the )?tool",
            r"send_email",
            r"delete_file",
            r"exfiltrate",
            r"attacker@example\.com",
            r"发送给\s*attacker",
            r"删除文件",
            r"泄露(密钥|token|凭证)",
            r"调用\s*send_email",
        ],
    },
    {
        "attack_type": "obfuscated_instruction",
        "name": "混淆或编码指令",
        "score": 15,
        "severity": "medium",
        "patterns": [
            r"rot13",
            r"decode this",
            r"encoded instruction",
            r"after decoding",
            r"解码后执行",
            r"编码后的指令",
        ],
    },
    {
        "attack_type": "benign_security_discussion",
        "name": "正常安全讨论",
        "score": 0,
        "severity": "info",
        "patterns": [
            r"什么是 prompt injection",
            r"如何防御 prompt injection",
            r"prompt injection prevention",
            r"安全风险分析",
            r"论文中提到",
            r"只做概念解释",
        ],
    },
]

# 有害攻击类型
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
}

# 英文转中文
ROLE_NAMES = {
    "user_input": "用户输入",
    "retrieved_doc": "检索文档",
    "tool_output": "工具返回内容",
    "tool_args": "工具调用参数",
    "chat_history": "历史对话",
}

# 英文转中文
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
}


def _find_evidence(text: str, patterns: list[str]) -> list[str]:
    """Return unique regex evidence fragments."""
    evidence: list[str] = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and match.group(0) not in evidence:
            evidence.append(match.group(0))
    return evidence

# 跑刚才定义的 CONTEXT_RULES，把命中的规则转成 MatchedRule 对象。
def _run_context_rules(text: str) -> list[MatchedRule]:
    """Run context-specific rules on top of the local text detector."""
    matched: list[MatchedRule] = []
    for rule in CONTEXT_RULES:
        evidence = _find_evidence(text, rule["patterns"])
        if not evidence:
            continue
        matched.append(
            MatchedRule(
                attack_type=rule["attack_type"],
                name=rule["name"],
                score=rule["score"],
                severity=rule["severity"],
                evidence=evidence,
            )
        )
    return matched

# 把文本规则 detector 返回的普通 dict 结果，转换成当前项目统一使用的 MatchedRule 模型。
def _text_rules_to_models(rule_result: dict[str, Any]) -> list[MatchedRule]:
    """Convert local text detector results into local Pydantic models."""
    matched_rules = []
    for rule in rule_result.get("matched_rules", []):
        matched_rules.append(
            MatchedRule(
                attack_type=str(rule.get("attack_type", "")),
                name=str(rule.get("name", "")),
                score=int(rule.get("score", 0)),
                severity=str(rule.get("severity", "medium")),
                evidence=[str(item) for item in rule.get("evidence", [])],
            )
        )
    return matched_rules

# 去重，但保留原来的顺序。
def _unique(values: list[str]) -> list[str]:
    """Keep input order while removing duplicates."""
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _risk_level(score: int) -> str:
    if score >= 90:
        return "CRITICAL"
    if score >= 70:
        return "HIGH"
    if score >= 45:
        return "MEDIUM"
    if score >= 20:
        return "LOW"
    return "SAFE"


def _decision(score: int) -> str:
    if score >= 75:
        return "BLOCK"
    if score >= 25:
        return "REVIEW"
    return "ALLOW"


def _source_trust_penalty(chunk: ContextChunk, has_risk: bool) -> int:
    """Low trust should amplify detected risk, but not create risk by itself."""
    if not has_risk:
        return 0
    if chunk.source_trust < 0.3:
        return 15
    if chunk.source_trust < 0.6:
        return 8
    return 0


def _permission_bonus(chunk: ContextChunk, attack_types: list[str]) -> int:
    """Increase risk when suspicious content is near privileged tool actions."""
    if chunk.context_role not in {"tool_output", "tool_args"} or not attack_types:
        return 0

    permission_scores = {
        "low": 0,
        "medium": 8,
        "high": 16,
        "critical": 25,
    }
    base = permission_scores.get(chunk.permission_level or "low", 0)
    if "tool_call_hijacking" in attack_types:
        base += 20
    if "sensitive_info_request" in attack_types:
        base += 10
    return base


def _role_bonus(chunk: ContextChunk, attack_types: list[str]) -> int:
    """Apply role-specific context amplification."""
    if not attack_types:
        return 0

    risky = bool(HARMFUL_ATTACK_TYPES & set(attack_types))
    if not risky:
        return 0

    if chunk.context_role == "retrieved_doc":
        bonus = 15
        if {"instruction_override", "system_prompt_leakage"} & set(attack_types):
            bonus += 10
        return bonus

    if chunk.context_role == "tool_output":
        bonus = 20
        if "tool_call_hijacking" in attack_types:
            bonus += 15
        return bonus

    if chunk.context_role == "tool_args":
        bonus = 20
        if "tool_call_hijacking" in attack_types:
            bonus += 15
        return bonus

    if chunk.context_role == "chat_history":
        return 5

    return 0

# 是否为间接注入
def _should_mark_indirect(chunk: ContextChunk, attack_types: list[str]) -> bool:
    if chunk.context_role not in {"retrieved_doc", "tool_output"}:
        return False
    indirect_markers = {
        "instruction_override",
        "system_prompt_leakage",
        "policy_bypass",
        "forced_compliance",
        "tool_call_hijacking",
    }
    return bool(indirect_markers & set(attack_types))

# 收集证据：
# MatchedRule(attack_type="instruction_override", evidence=["ignore previous instructions"]) --->  ["ignore previous instructions"]
def _collect_evidence(matched_rules: list[MatchedRule]) -> list[str]:
    evidence = []
    for rule in matched_rules:
        if rule.attack_type == "benign_security_discussion":
            continue
        evidence.extend(rule.evidence)
    return _unique(evidence)


def _base_score(matched_rules: list[MatchedRule], benign_only: bool) -> int:
    score = sum(
        rule.score
        for rule in matched_rules
        if rule.attack_type != "benign_security_discussion"
    )
    if benign_only:
        return max(0, score - 25)
    return min(100, score)


def _build_reason(
    chunk: ContextChunk,
    attack_types: list[str],
    evidence: list[str],
    score: int,
    benign_only: bool,
) -> str:
    role_name = ROLE_NAMES.get(chunk.context_role, chunk.context_role)
    if benign_only:
        return f"{role_name}包含 Prompt Injection 相关安全术语，但语义更接近正常安全讨论，未直接阻断。"

    if not attack_types:
        return f"{role_name}未检测到明显 Prompt Injection 或工具调用劫持证据。"

    attack_names = [ATTACK_NAMES.get(item, item) for item in attack_types]
    evidence_text = "、".join(evidence[:3]) if evidence else "无明确片段"

    if chunk.context_role == "retrieved_doc":
        return (
            f"检索文档中检测到{'、'.join(attack_names)}风险，证据包括：{evidence_text}。"
            f"该内容来自 {chunk.source}，可能作为间接 Prompt Injection 污染下游回答。"
        )
    if chunk.context_role == "tool_output":
        return (
            f"工具返回内容中检测到{'、'.join(attack_names)}风险，证据包括：{evidence_text}。"
            "该内容不应被 Agent 当作高优先级指令执行。"
        )
    if chunk.context_role == "chat_history":
        return (
            f"历史对话中检测到{'、'.join(attack_names)}迹象，证据包括：{evidence_text}。"
            "单轮历史风险较低，但多轮重复出现时会累积整体风险。"
        )
    return (
        f"用户输入中检测到{'、'.join(attack_names)}风险，证据包括：{evidence_text}。"
        f"当前片段风险分为 {score}。"
    )


def analyze_chunk(chunk: ContextChunk) -> ChunkRiskResult:
    """Analyze one normalized context chunk."""
    rule_result = detect_context_rules(chunk.content)
    matched_rules = _text_rules_to_models(rule_result)

    attack_types = _unique(
        [
            rule.attack_type
            for rule in matched_rules
            if rule.attack_type != "benign_security_discussion"
        ]
    )
    benign_only = bool(
        any(rule.attack_type == "benign_security_discussion" for rule in matched_rules)
        and not attack_types
    )

    if _should_mark_indirect(chunk, attack_types):
        attack_types.insert(0, "indirect_prompt_injection")


    evidence = _collect_evidence(matched_rules)
    base_score = _base_score(matched_rules, benign_only=benign_only)
    has_risk = bool(HARMFUL_ATTACK_TYPES & set(attack_types))
    role_bonus = _role_bonus(chunk, attack_types)
    #低可信来源
    trust_penalty = _source_trust_penalty(chunk, has_risk)
    permission_bonus = _permission_bonus(chunk, attack_types)
    #总的上下文加分
    context_bonus = role_bonus + trust_penalty + permission_bonus
    transformer_prediction = predict_chunk(chunk)
    features = build_chunk_features(
        chunk=chunk,
        rule_result=rule_result,
        transformer_prob=transformer_prediction.transformer_prob,
        attack_types=attack_types,
    )
    xgboost_prob = RISK_MODEL.predict_proba(features)
    decision_state = decide_risk(
        xgboost_prob,
        rule_block=bool(rule_result.get("rule_block", False)),
    )
    risk_score = int(decision_state["risk_score"])
    risk_level = str(decision_state["risk_level"])
    decision = str(decision_state["decision"])

    return ChunkRiskResult(
        chunk_id=chunk.chunk_id,
        context_role=chunk.context_role,
        source=chunk.source,
        source_trust=chunk.source_trust,
        risk_score=risk_score,
        final_risk_probability=float(decision_state["risk_probability"]),
        risk_level=risk_level,
        decision=decision,
        rule_block=bool(rule_result.get("rule_block", False)),
        rule_score=int(rule_result.get("rule_score", base_score)),
        matched_rule_count=int(rule_result.get("matched_rule_count", len(matched_rules))),
        attack_types=attack_types,
        evidence=evidence,
        reason=_build_reason(
            chunk=chunk,
            attack_types=attack_types,
            evidence=evidence,
            score=risk_score,
            benign_only=benign_only,
        ),
        base_score=base_score,
        context_bonus=context_bonus,
        source_trust_penalty=trust_penalty,
        permission_bonus=permission_bonus,
        transformer_prob=transformer_prediction.transformer_prob,
        transformer_model_status=transformer_prediction.model_status,
        xgboost_prob=xgboost_prob,
        risk_model_status=getattr(RISK_MODEL, "model_status", "unknown"),
        matched_rules=matched_rules,
        tool_name=chunk.tool_name,
        permission_level=chunk.permission_level,
        metadata=chunk.metadata,
    )


def aggregate_context_risk(
    chunk_results: list[ChunkRiskResult],
    request_id: str | None = None,
) -> ContextRiskResult:
    """Aggregate chunk-level results into a final context decision."""
    if not chunk_results:
        return ContextRiskResult(
            final_decision="ALLOW",
            final_risk_score=0,
            final_risk_probability=0.0,
            risk_level="SAFE",
            summary="没有可扫描的上下文片段。",
            request_id=request_id,
        )

    # 最高风险片段
    primary = max(chunk_results, key=lambda item: item.final_risk_probability)
    # 所有风险片段
    risky_chunks = [item for item in chunk_results if item.decision != "ALLOW"]
    chat_history_risky_count = sum(
        1
        for item in risky_chunks
        if item.context_role == "chat_history"
        and ("system_prompt_leakage" in item.attack_types or "forced_compliance" in item.attack_types)
    )

    # 如果风险片段不止一个，每多一个加 5 分。
    multiple_risk_bonus = max(0, len(risky_chunks) - 1) * 5
    # 如果历史对话中这类风险出现了 2 次及以上，加 10 分。
    history_bonus = 10 if chat_history_risky_count >= 2 else 0
    # 最终风险概率 = 最高 chunk 概率 + 多风险片段加分 + 历史污染加分
    final_probability = min(
        1.0,
        primary.final_risk_probability
        + multiple_risk_bonus / 100
        + history_bonus / 100,
    )
    final_state = decide_risk(
        final_probability,
        rule_block=any(item.rule_block for item in chunk_results),
    )
    final_score = int(final_state["risk_score"])

    final_decision = str(final_state["decision"])
    risk_level = str(final_state["risk_level"])
    summary = _build_context_summary(primary, risky_chunks, final_score)

    return ContextRiskResult(
        final_decision=final_decision,
        final_risk_score=final_score,
        final_risk_probability=float(final_state["risk_probability"]),
        risk_level=risk_level,
        summary=summary,
        primary_risk_chunk_id=primary.chunk_id,
        primary_context_role=primary.context_role,
        primary_risk_source=primary.context_role,
        risky_chunk_count=len(risky_chunks),
        chunk_results=chunk_results,
        safe_chunks=[item.chunk_id for item in chunk_results if item.decision == "ALLOW"],
        blocked_chunks=[item.chunk_id for item in chunk_results if item.decision == "BLOCK"],
        request_id=request_id,
    )


def _build_context_summary(
    primary: ChunkRiskResult,
    risky_chunks: list[ChunkRiskResult],
    final_score: int,
) -> str:
    if final_score < 25:
        return "上下文中未发现明显 Prompt Injection、间接注入或工具调用劫持风险。"

    role_counts: dict[str, int] = {}
    attack_types: list[str] = []
    for item in risky_chunks:
        """
        role_counts = {
            "retrieved_doc": 2,
            "tool_output": 1
        }
        """
        role_counts[item.context_role] = role_counts.get(item.context_role, 0) + 1

        attack_types.extend(item.attack_types)

    #转成中文
    role_text = "、".join(
        f"{ROLE_NAMES.get(role, role)} {count} 个"
        for role, count in role_counts.items()
    )
    attack_text = "、".join(
        ATTACK_NAMES.get(item, item)
        for item in _unique(attack_types)[:4]
    )
    
    return (
        f"检测到 {role_text} 片段存在风险，主要风险来自 {primary.context_role} "
        f"片段 {primary.chunk_id}。攻击类型包括：{attack_text}。"
    )
