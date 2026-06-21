"""FastAPI service for strict five-layer RAG/Agent context scanning."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request, status

from .context_models import (
    ChunkRiskResult,
    ChunkScanRequest,
    ContextRiskResult,
    ContextScanRequest,
    DemoContextCase,
)
from .context_risk_analyzer import analyze_chunk
from .context_scanner import scan_context
from .demo_context_cases import get_demo_cases
from .model_runtime import ModelRuntime, load_runtime


RuntimeLoader = Callable[[], ModelRuntime]


def create_app(runtime_loader: RuntimeLoader | None = None) -> FastAPI:
    loader = runtime_loader or (lambda: load_runtime(allow_download=False))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            application.state.runtime = loader()
            application.state.model_error = None
        except Exception as error:
            application.state.runtime = None
            application.state.model_error = str(error)
        yield

    application = FastAPI(
        title="RAG/Agent Prompt Injection Context Scanner",
        description="Rule + multilingual Transformer + context features + XGBoost risk fusion.",
        version="1.0.0",
        lifespan=lifespan,
    )

    def ready_runtime(request: Request) -> ModelRuntime:
        runtime = getattr(request.app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "models_not_ready",
                    "message": getattr(request.app.state, "model_error", "Model runtime is unavailable."),
                },
            )
        return runtime

    @application.get("/health")
    @application.get("/health/live")
    def health_live() -> dict[str, str]:
        return {"status": "ok", "message": "Scanner process is running"}

    @application.get("/health/ready")
    def health_ready(request: Request) -> dict[str, Any]:
        runtime = ready_runtime(request)
        return {"status": "ready", **runtime.status()}

    @application.get("/models/status")
    def models_status(request: Request) -> dict[str, Any]:
        runtime = getattr(request.app.state, "runtime", None)
        if runtime is None:
            return {
                "ready": False,
                "error": getattr(request.app.state, "model_error", "Model runtime is unavailable."),
            }
        return runtime.status()

    @application.post("/v1/scan/context", response_model=ContextRiskResult)
    @application.post("/scan/context", response_model=ContextRiskResult, include_in_schema=False)
    def scan_context_endpoint(request_body: ContextScanRequest, request: Request) -> ContextRiskResult:
        return scan_context(request_body, runtime=ready_runtime(request))

    @application.post("/v1/scan/chunk", response_model=ChunkRiskResult)
    def scan_chunk_endpoint(request_body: ChunkScanRequest, request: Request) -> ChunkRiskResult:
        return analyze_chunk(request_body.to_context_chunk(), runtime=ready_runtime(request))

    @application.get("/demo/context-cases", response_model=list[DemoContextCase], include_in_schema=False)
    @application.get("/demo/cases", response_model=list[DemoContextCase])
    def demo_cases() -> list[DemoContextCase]:
        return get_demo_cases()

    return application


app = create_app()
