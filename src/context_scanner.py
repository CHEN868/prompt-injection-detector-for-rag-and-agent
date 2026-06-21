"""Context scanner that flattens RAG/Agent inputs and aggregates risk."""
# 把 request 拆成多个 ContextChunk，并调用风险分析器分析每个 chunk，最后聚合成整体风险结果
from __future__ import annotations

import json

from .context_models import ContextChunk, ContextRiskResult, ContextScanRequest
from .context_risk_analyzer import aggregate_context_risk, analyze_chunk


def flatten_context_request(request: ContextScanRequest) -> list[ContextChunk]:
    """Convert a structured scan request into normalized chunks."""
    chunks = [
        ContextChunk(
            chunk_id="user_input:0",
            case_id=request.case_id,
            scenario=request.scenario,
            context_role="user_input",
            content=request.user_input,
            source="user",
            source_trust=0.7,
            metadata={"request_id": request.request_id} if request.request_id else {},
            history_risk_count=request.history_risk_count,
        )
    ]

    for index, doc in enumerate(request.retrieved_docs):
        doc_chunk_id = doc.chunk_id or f"chunk_{index:03d}"
        chunks.append(
            ContextChunk(
                chunk_id=f"retrieved_doc:{doc.doc_id}:{doc_chunk_id}",
                case_id=request.case_id,
                scenario=request.scenario,
                context_role="retrieved_doc",
                content=doc.content,
                source=doc.source,
                source_trust=doc.source_trust,
                metadata={
                    **doc.metadata,
                    "doc_id": doc.doc_id,
                    "original_chunk_id": doc_chunk_id,
                    "index": index,
                },
                history_risk_count=request.history_risk_count,
            )
        )

    for index, output in enumerate(request.tool_outputs):
        chunks.append(
            ContextChunk(
                chunk_id=f"tool_output:{output.tool_name}:{index}",
                case_id=request.case_id,
                scenario=request.scenario,
                context_role="tool_output",
                content=output.content,
                source=output.source,
                source_trust=output.source_trust,
                metadata={
                    **output.metadata,
                    "tool_name": output.tool_name,
                    "index": index,
                },
                tool_name=output.tool_name,
                permission_level=output.permission_level,
                history_risk_count=request.history_risk_count,
            )
        )

    for index, tool_args in enumerate(request.tool_args):
        content = tool_args.content or json.dumps(
            tool_args.arguments,
            ensure_ascii=False,
            sort_keys=True,
        )
        chunks.append(
            ContextChunk(
                chunk_id=f"tool_args:{tool_args.tool_name}:{index}",
                case_id=request.case_id,
                scenario=request.scenario,
                context_role="tool_args",
                content=content,
                source=tool_args.source,
                source_trust=tool_args.source_trust,
                metadata={
                    **tool_args.metadata,
                    "tool_name": tool_args.tool_name,
                    "arguments": tool_args.arguments,
                    "index": index,
                },
                tool_name=tool_args.tool_name,
                permission_level=tool_args.permission_level,
                history_risk_count=request.history_risk_count,
            )
        )

    for index, message in enumerate(request.chat_history):
        chunks.append(
            ContextChunk(
                chunk_id=f"chat_history:{index}:{message.role}",
                case_id=request.case_id,
                scenario=request.scenario,
                context_role="chat_history",
                content=message.content,
                source=f"chat:{message.role}",
                source_trust=0.7,
                metadata={
                    **message.metadata,
                    "chat_role": message.role,
                    "index": index,
                },
                history_risk_count=request.history_risk_count,
            )
        )

    return chunks


def scan_context(request: ContextScanRequest, runtime=None) -> ContextRiskResult:
    """Scan a structured RAG/Agent context and return aggregate risk."""
    chunks = flatten_context_request(request)
    chunk_results = [analyze_chunk(chunk, runtime=runtime) for chunk in chunks]
    return aggregate_context_risk(
        chunk_results=chunk_results,
        request_id=request.request_id,
    )

"""
main()
    ↓
get_demo_cases()
    ↓
case.request
    ↓
scan_context(case.request)
    ↓
flatten_context_request(request)
    ↓
多个 ContextChunk
    ↓
analyze_chunk(chunk)
    ↓
多个 ChunkRiskResult
    ↓
aggregate_context_risk(chunk_results)
    ↓
ContextRiskResult
    ↓
main() 打印结果
"""
