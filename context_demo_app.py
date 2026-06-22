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
    cols[0].metric("Final decision", result.decision)
    cols[1].metric("Risk probability", f"{result.final_risk_probability:.2f}")
    cols[2].metric("Rule score", f"{result.rule_score:.2f}")
    cols[3].metric("Transformer", f"{result.transformer_prob:.2f}")
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
                "decision": chunk.decision,
                "risk_probability": chunk.final_risk_probability,
                "rule_block": chunk.rule_block,
                "rule_score": chunk.rule_score,
                "transformer_prob": chunk.transformer_prob,
                "context_risk_score": chunk.context_risk_score,
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_chunk_details(result) -> None:
    """Render chunk-level explanations."""
    for chunk in result.chunk_results:
        title = f"{chunk.chunk_id} | {chunk.decision} | p={chunk.final_risk_probability:.2f}"
        with st.expander(title, expanded=chunk.decision != "ALLOW"):
            st.json(model_to_dict(chunk))


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
