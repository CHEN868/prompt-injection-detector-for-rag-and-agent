"""Local rule-based text detector for prompt injection demo."""
# 负责基础文本规则检测，提供给 context_risk_analyzer 使用
from __future__ import annotations

import re
from typing import Any


RULES = [
    {
        "attack_type": "instruction_override",
        "name": "指令覆盖",
        "description": "试图覆盖、忽略或替换原有系统/开发者指令。",
        "score": 25,
        "severity": "high",
        "patterns": [
            r"ignore (all )?(previous|prior|earlier|above) instructions?",
            r"disregard (all )?(previous|prior|earlier|above) instructions?",
            r"override (the )?(system prompt|developer instruction)",
            r"忽略之前(的)?(所有)?(系统)?指令",
            r"无视(系统提示词|开发者指令|之前的指令)",
        ],
    },
    {
        "attack_type": "system_prompt_leakage",
        "name": "系统提示词泄露",
        "description": "诱导模型输出 system prompt、开发者消息或隐藏内部说明。",
        "score": 30,
        "severity": "high",
        "patterns": [
            r"reveal (the )?system prompt",
            r"print (the )?system prompt",
            r"show (me )?(your )?(system prompt|developer message)",
            r"(隐藏的?\s*)?system prompt\s*(输出|泄露|透露|公开)",
            r"(输出|泄露|透露|公开).{0,12}system prompt",
            r"developer message",
            r"输出(你的)?系统提示词",
            r"隐藏提示词",
            r"内部说明",
        ],
    },
    {
        "attack_type": "jailbreak_roleplay",
        "name": "角色扮演越狱",
        "description": "通过角色扮演或特殊模式诱导模型绕过限制。",
        "score": 25,
        "severity": "high",
        "patterns": [
            r"\bDAN\b",
            r"developer mode",
            r"unrestricted AI",
            r"pretend (to be|you are)",
            r"进入开发者模式",
            r"无限制 AI",
            r"扮演.{0,12}(无视规则|不受限制)",
        ],
    },
    {
        "attack_type": "policy_bypass",
        "name": "安全策略绕过",
        "description": "要求模型绕过安全政策、限制、过滤或审核。",
        "score": 25,
        "severity": "high",
        "patterns": [
            r"bypass (the )?(policy|safety|guardrails|filters?)",
            r"ignore (the )?(policy|safety policy|guardrails)",
            r"without (any )?(restrictions|limitations|filtering)",
            r"绕过.{0,12}(安全策略|安全规则|限制|审核|过滤)",
            r"不要遵守.{0,8}(政策|规则|限制)",
        ],
    },
    {
        "attack_type": "forced_compliance",
        "name": "强制服从",
        "description": "强制模型必须回答、不得拒绝或无条件服从。",
        "score": 20,
        "severity": "medium",
        "patterns": [
            r"you must comply",
            r"you must answer",
            r"do not refuse",
            r"always answer",
            r"必须(回答|执行|照做)",
            r"不得拒绝",
            r"无条件(服从|执行)",
        ],
    },
    {
        "attack_type": "sensitive_info_request",
        "name": "敏感信息请求",
        "description": "请求泄露密钥、token、凭证、密码或内部机密。",
        "score": 25,
        "severity": "high",
        "patterns": [
            r"\b(api key|secret key|access token|password|credential)s?\b",
            r"private key",
            r"泄露(密钥|token|凭证|密码|机密)",
            r"输出(密钥|token|凭证|密码)",
            r"发送.{0,12}(密钥|token|凭证|密码)",
        ],
    },
    {
        "attack_type": "obfuscated_instruction",
        "name": "混淆或编码指令",
        "description": "通过编码、加密或混淆文本隐藏真实指令。",
        "score": 15,
        "severity": "medium",
        "patterns": [
            r"base64",
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


def _find_evidence(text: str, patterns: list[str]) -> list[str]:
    """Return unique regex evidence fragments."""
    evidence: list[str] = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and match.group(0) not in evidence:
            evidence.append(match.group(0))
    return evidence


def detect_text_rules(text: str) -> dict[str, Any]:
    """Detect prompt-injection signals with local transparent regex rules."""
    matched_rules: list[dict[str, Any]] = []
    matched_types: list[str] = []
    rule_score = 0
    is_benign_security_discussion = False

    for rule in RULES:
        evidence = _find_evidence(text, rule["patterns"])
        if not evidence:
            continue

        attack_type = rule["attack_type"]
        matched_rules.append(
            {
                "attack_type": attack_type,
                "name": rule["name"],
                "description": rule["description"],
                "severity": rule["severity"],
                "score": rule["score"],
                "evidence": evidence,
            }
        )

        if attack_type == "benign_security_discussion":
            is_benign_security_discussion = True
            continue

        matched_types.append(attack_type)
        rule_score += rule["score"]

    return {
        "rule_score": rule_score,
        "matched_types": matched_types,
        "matched_rules": matched_rules,
        "is_benign_security_discussion": is_benign_security_discussion,
    }
