"""Context hard-rule detector with explainable evidence."""

from __future__ import annotations

import re
from typing import Any

from .text_detector import detect_text_rules


CONTEXT_RULES = [
    {
        "attack_type": "tool_call_hijacking",
        "name": "工具调用劫持",
        "description": "诱导 Agent 调用、改写或滥用工具。",
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
        "description": "通过编码、加密或混淆文本隐藏真实指令。",
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
        "description": "关于 Prompt Injection 或安全风险的正常概念讨论。",
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

BLOCK_ATTACK_TYPES = {
    "instruction_override",
    "system_prompt_leakage",
    "jailbreak_roleplay",
    "policy_bypass",
    "sensitive_info_request",
    "tool_call_hijacking",
}


def _find_evidence(text: str, patterns: list[str]) -> list[str]:
    evidence: list[str] = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and match.group(0) not in evidence:
            evidence.append(match.group(0))
    return evidence


def _unique(values: list[str]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _run_context_rules(text: str) -> list[dict[str, Any]]:
    matched_rules: list[dict[str, Any]] = []
    for rule in CONTEXT_RULES:
        evidence = _find_evidence(text, rule["patterns"])
        if not evidence:
            continue
        matched_rules.append(
            {
                "attack_type": rule["attack_type"],
                "name": rule["name"],
                "description": rule["description"],
                "severity": rule["severity"],
                "score": rule["score"],
                "evidence": evidence,
            }
        )
    return matched_rules


def detect_context_rules(text: str) -> dict[str, Any]:
    """Run local text and context hard rules and preserve explainable outputs."""
    text_result = detect_text_rules(text)
    matched_rules = list(text_result.get("matched_rules", []))
    matched_rules.extend(_run_context_rules(text))

    attack_types = _unique(
        [
            str(rule.get("attack_type", ""))
            for rule in matched_rules
            if rule.get("attack_type") != "benign_security_discussion"
        ]
    )
    evidence = _unique(
        [
            str(item)
            for rule in matched_rules
            if rule.get("attack_type") != "benign_security_discussion"
            for item in rule.get("evidence", [])
        ]
    )
    rule_score = min(
        100,
        sum(
            int(rule.get("score", 0))
            for rule in matched_rules
            if rule.get("attack_type") != "benign_security_discussion"
        ),
    )
    matched_rule_count = len(
        [
            rule
            for rule in matched_rules
            if rule.get("attack_type") != "benign_security_discussion"
        ]
    )
    benign_only = bool(
        any(rule.get("attack_type") == "benign_security_discussion" for rule in matched_rules)
        and not attack_types
    )
    rule_block = bool(BLOCK_ATTACK_TYPES & set(attack_types)) and not benign_only
    reason = "未命中高危规则。"
    if benign_only:
        reason = "命中正常安全讨论 hard negative 规则，不计入攻击分。"
    elif attack_types:
        reason = "命中硬规则：" + "、".join(attack_types)

    return {
        "rule_block": rule_block,
        "rule_score": rule_score,
        "matched_rule_count": matched_rule_count,
        "matched_rules": matched_rules,
        "attack_types": attack_types,
        "evidence": evidence,
        "reason": reason,
        "is_benign_security_discussion": bool(
            text_result.get("is_benign_security_discussion", False)
            or any(rule.get("attack_type") == "benign_security_discussion" for rule in matched_rules)
        ),
    }
