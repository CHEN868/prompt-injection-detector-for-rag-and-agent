"""Streamlit UI for the RAG/Agent context risk scanner demo."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from src.context_models import ContextScanRequest
from src.context_scanner import scan_context
from src.demo_context_cases import get_demo_cases


def model_to_dict(model: Any) -> dict[str, Any]:
    """Serialize Pydantic models across v1/v2."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def parse_request_json(raw_json: str) -> ContextScanRequest:
    """Parse and validate the editable request JSON."""
    payload = json.loads(raw_json)
    return ContextScanRequest(**payload)


def render_summary(result) -> None:
    """Render the final context-level decision."""
    cols = st.columns(5)
    cols[0].metric("Final decision", result.final_decision)
    cols[1].metric("Risk level", result.risk_level)
    cols[2].metric("Risk score", result.final_risk_score)
    cols[3].metric("Risk probability", f"{result.final_risk_probability:.2f}")
    cols[4].metric("Risky chunks", result.risky_chunk_count)

    st.write(result.summary)
    if result.primary_risk_chunk_id:
        st.caption(
            f"Primary risk source: {result.primary_context_role} / {result.primary_risk_chunk_id}"
        )


def render_chunk_table(result) -> None:
    """Render one row per scanned context chunk."""
    rows = []
    for chunk in result.chunk_results:
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "role": chunk.context_role,
                "source": chunk.source,
                "trust": chunk.source_trust,
                "decision": chunk.decision,
                "risk_level": chunk.risk_level,
                "risk_score": chunk.risk_score,
                "risk_probability": chunk.final_risk_probability,
                "rule_block": chunk.rule_block,
                "attack_types": ", ".join(chunk.attack_types),
                "evidence": ", ".join(chunk.evidence[:3]),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_chunk_details(result) -> None:
    """Render chunk-level explanations."""
    for chunk in result.chunk_results:
        title = f"{chunk.chunk_id} | {chunk.decision} | score={chunk.risk_score}"
        with st.expander(title, expanded=chunk.decision != "ALLOW"):
            st.write(chunk.reason)
            st.json(
                {
                    "context_role": chunk.context_role,
                    "source": chunk.source,
                    "source_trust": chunk.source_trust,
                    "attack_types": chunk.attack_types,
                    "evidence": chunk.evidence,
                    "rule_block": chunk.rule_block,
                    "rule_score": chunk.rule_score,
                    "matched_rule_count": chunk.matched_rule_count,
                    "transformer_prob": chunk.transformer_prob,
                    "transformer_model_status": chunk.transformer_model_status,
                    "xgboost_prob": chunk.xgboost_prob,
                    "risk_model_status": chunk.risk_model_status,
                    "base_score": chunk.base_score,
                    "context_bonus": chunk.context_bonus,
                    "source_trust_penalty": chunk.source_trust_penalty,
                    "permission_bonus": chunk.permission_bonus,
                    "tool_name": chunk.tool_name,
                    "permission_level": chunk.permission_level,
                }
            )


st.set_page_config(
    page_title="RAG / Agent Context Risk Scanner",
    layout="wide",
)

st.title("RAG / Agent 上下文风险扫描 Demo")
st.write("扫描 user_input、retrieved_docs、tool_outputs 和 chat_history，并输出上下文感知风险。")

cases = get_demo_cases()
case_names = [f"{case.case_id} - {case.name}" for case in cases]
selected_index = st.selectbox(
    "选择内置 Demo case",
    options=list(range(len(cases))),
    format_func=lambda index: case_names[index],
)
selected_case = cases[selected_index]

st.caption(selected_case.description)

default_json = json.dumps(
    model_to_dict(selected_case.request),
    ensure_ascii=False,
    indent=2,
)

request_json = st.text_area(
    "ContextScanRequest JSON",
    value=default_json,
    height=360,
)

if st.button("执行上下文扫描", type="primary"):
    try:
        request = parse_request_json(request_json)
        result = scan_context(request)
    except Exception as error:
        st.error(f"扫描失败：{error}")
    else:
        st.divider()
        render_summary(result)
        st.subheader("Chunk risk table")
        render_chunk_table(result)
        st.subheader("Chunk explanations")
        render_chunk_details(result)
