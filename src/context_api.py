"""FastAPI service for the RAG/Agent context risk demo."""

from __future__ import annotations

from fastapi import FastAPI

from .context_models import ContextRiskResult, ContextScanRequest, DemoContextCase
from .context_scanner import scan_context
from .demo_context_cases import get_demo_cases


app = FastAPI(
    title="RAG/Agent Prompt Injection Context Scanner",
    description="Context-aware demo scanner for RAG documents, tool outputs, and chat history.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health check."""
    return {
        "status": "ok",
        "message": "RAG/Agent context scanner is running",
    }


@app.post("/scan/context", response_model=ContextRiskResult)
def scan_context_endpoint(request: ContextScanRequest) -> ContextRiskResult:
    """Scan a structured RAG/Agent context request."""
    return scan_context(request)


@app.get("/demo/context-cases", response_model=list[DemoContextCase])
def demo_context_cases() -> list[DemoContextCase]:
    """Return built-in context scan demo cases."""
    return get_demo_cases()


@app.get("/demo/cases", response_model=list[DemoContextCase])
def demo_cases() -> list[DemoContextCase]:
    """Return built-in context scan demo cases."""
    return get_demo_cases()
