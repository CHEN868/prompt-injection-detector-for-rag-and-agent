"""Pydantic models for context-aware RAG/Agent prompt injection scanning."""

from typing import Any, Literal

from pydantic import BaseModel, Field

# 上下文来源角色(用户输入，检索到的文档，工具输出，工具参数，聊天历史)
ContextRole = Literal["user_input", "retrieved_doc", "tool_output", "tool_args", "chat_history"]
# 场景类型
Scenario = Literal["chat", "rag", "agent", "tool_call", "multi_turn"]
# 权限级别
PermissionLevel = Literal["none", "low", "medium", "high", "critical"]
# 决策
Decision = Literal["ALLOW", "REVIEW", "BLOCK"]
# 风险级别
RiskLevel = Literal["SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

# RAG检索出来的一段文档
class RetrievedDoc(BaseModel):
    """One RAG document chunk returned by a retriever."""
    # 文档ID
    doc_id: str = Field(..., min_length=1)
    # 块ID
    chunk_id: str | None = None
    # 内容
    content: str = Field(..., min_length=1)
    # 来源
    source: str = "unknown"
    # 源信任度
    source_trust: float = Field(0.5, ge=0.0, le=1.0)
    #额外信息
    metadata: dict[str, Any] = Field(default_factory=dict)

# Agent工具返回的一段内容
class ToolOutput(BaseModel):
    """One piece of content returned by an Agent tool."""

    tool_name: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    source: str = "tool"
    source_trust: float = Field(0.5, ge=0.0, le=1.0)
    #工具权限
    permission_level: PermissionLevel = "low"
    metadata: dict[str, Any] = Field(default_factory=dict)


# Agent 即将调用工具时的参数
class ToolArgs(BaseModel):
    """Arguments prepared for an Agent tool call."""

    tool_name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    content: str | None = None
    source: str = "agent"
    source_trust: float = Field(0.7, ge=0.0, le=1.0)
    permission_level: PermissionLevel = "low"
    metadata: dict[str, Any] = Field(default_factory=dict)


# 历史对话消息
class ChatMessage(BaseModel):
    """One historical chat turn."""

    role: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

# 完整的上下文安全扫描请求
class ContextScanRequest(BaseModel):
    """Structured request for context-aware prompt injection scanning."""

    scenario: Scenario = "chat"
    user_input: str = Field(..., min_length=1)
    retrieved_docs: list[RetrievedDoc] = Field(default_factory=list)
    tool_outputs: list[ToolOutput] = Field(default_factory=list)
    tool_args: list[ToolArgs] = Field(default_factory=list)
    chat_history: list[ChatMessage] = Field(default_factory=list)
    case_id: str | None = None
    request_id: str | None = None
    history_risk_count: int = Field(0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

# 标准化之后的一段待检测文本
class ContextChunk(BaseModel):
    """Normalized text chunk with source and role metadata."""

    chunk_id: str
    case_id: str | None = None
    scenario: Scenario = "chat"
    #上下文来源角色（用户输入，检索到的文档，工具输出，聊天历史）
    context_role: ContextRole
    content: str
    source: str = "unknown"
    source_trust: float = Field(0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tool_name: str | None = None
    permission_level: PermissionLevel | None = None
    history_risk_count: int = Field(0, ge=0)

# 匹配的规则
class MatchedRule(BaseModel):
    """Rule evidence used in explanations and debugging."""

    attack_type: str
    name: str
    score: int
    #严重程度
    severity: str = "medium"
    evidence: list[str] = Field(default_factory=list)

# 单个chunk的风险监测结果
class ChunkRiskResult(BaseModel):
    """Risk result for one context chunk."""

    chunk_id: str
    context_role: ContextRole
    source: str
    source_trust: float
    risk_score: int
    final_risk_probability: float = Field(0.0, ge=0.0, le=1.0)
    risk_level: RiskLevel
    decision: Decision
    rule_block: bool = False
    rule_score: int = 0
    matched_rule_count: int = 0
    attack_types: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    #原因
    reason: str
    #基础分
    base_score: int = 0
    #上下文加分
    context_bonus: int = 0
    #源信任度惩罚
    source_trust_penalty: int = 0
    #权限加分
    permission_bonus: int = 0
    transformer_prob: float | None = Field(None, ge=0.0, le=1.0)
    transformer_model_status: str = "not_configured"
    xgboost_prob: float | None = Field(None, ge=0.0, le=1.0)
    risk_model_status: str = "not_loaded"
    decision_source: Literal["hard_rule", "xgboost", "aggregate"] = "xgboost"
    matched_rules: list[MatchedRule] = Field(default_factory=list)
    tool_name: str | None = None
    permission_level: PermissionLevel | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

# 一次完整请求的最终风险结果
class ContextRiskResult(BaseModel):
    """Aggregated context-level risk result."""

    final_decision: Decision
    final_risk_score: int
    final_risk_probability: float = Field(0.0, ge=0.0, le=1.0)
    risk_level: RiskLevel
    #总结
    summary: str
    #主要风险chunk ID
    primary_risk_chunk_id: str | None = None
    #主要风险来源类型（用户输入，检索到的文档，工具输出，聊天历史）
    primary_context_role: ContextRole | None = None
    primary_risk_source: str | None = None
    #有风险的chunk数
    risky_chunk_count: int = 0
    chunk_results: list[ChunkRiskResult] = Field(default_factory=list)
    safe_chunks: list[str] = Field(default_factory=list)
    blocked_chunks: list[str] = Field(default_factory=list)
    request_id: str | None = None
    decision_source: Literal["hard_rule", "aggregate"] = "aggregate"


class DemoContextCase(BaseModel):
    """A built-in demonstration case."""

    case_id: str
    name: str
    description: str
    request: ContextScanRequest


class ChunkScanRequest(BaseModel):
    """API request for scanning one RAG or Agent context chunk."""

    scenario: Scenario = "rag"
    context_role: ContextRole = "retrieved_doc"
    content: str = Field(..., min_length=1)
    chunk_id: str = "chunk:0"
    source: str = "unknown"
    source_trust: float = Field(0.5, ge=0.0, le=1.0)
    permission_level: PermissionLevel = "none"
    tool_name: str | None = None
    history_risk_count: int = Field(0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_context_chunk(self) -> ContextChunk:
        return ContextChunk(
            chunk_id=self.chunk_id,
            scenario=self.scenario,
            context_role=self.context_role,
            content=self.content,
            source=self.source,
            source_trust=self.source_trust,
            permission_level=self.permission_level,
            tool_name=self.tool_name,
            history_risk_count=self.history_risk_count,
            metadata=self.metadata,
        )


"""
ContextScanRequest(完整的上下文安全扫描请求)
        |
        |  标准化 / 拆分
        v
多个 ContextChunk(标准化之后的一段待检测文本)
        |
        |  逐个检测
        v
多个 ChunkRiskResult(单个chunk的风险监测结果)
        |
        |  聚合
        v
ContextRiskResult(一次完整请求的最终风险结果)
"""
