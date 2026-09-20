"""Construct an API instance without initializing storage at module import."""

from fastapi import FastAPI

from api.dependencies import create_container
from api.documents import router as documents_router
from api.evidence import router as evidence_router
from api.routes import router as core_router
from api.runs import router as runs_router
from config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an isolated app; omitted settings retain normal environment validation."""
    application = FastAPI(title="Scholar RAG Agent", version="0.1.0")
    application.state.container = create_container(settings)
    application.include_router(core_router)
    application.include_router(documents_router)
    application.include_router(evidence_router)
    application.include_router(runs_router)
    return application
