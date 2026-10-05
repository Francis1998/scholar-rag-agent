"""Construct an API instance without initializing storage at module import."""

import logging

from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.annotations import router as annotations_router
from api.bibliography import router as bibliography_router
from api.collections import router as collections_router
from api.comparisons import router as comparisons_router
from api.corpus_drift import router as corpus_drift_router
from api.dependencies import create_container
from api.documents import router as documents_router
from api.evidence import router as evidence_router
from api.research import router as research_router
from api.retrieval import router as retrieval_router
from api.reviews import router as reviews_router
from api.routes import router as core_router
from api.runs import router as runs_router
from api.screening import router as screening_router
from config import Settings

logger = logging.getLogger(__name__)


async def _evidence_validation_error(request: Request, exc: RequestValidationError) -> Response:
    """Do not echo unencodable numeric input; leave unrelated validation contracts intact."""
    errors = exc.errors()
    if not any(error["loc"][:2] == ("body", "near_duplicate_threshold") for error in errors):
        return await request_validation_exception_handler(request, exc)
    logger.warning("Evidence request validation failed: near_duplicate_threshold")
    detail = [
        {key: error[key] for key in ("loc", "type", "msg")}
        if error["loc"][:2] == ("body", "near_duplicate_threshold")
        else error
        for error in errors
    ]
    return JSONResponse(
        {"detail": jsonable_encoder(detail)},
        status_code=422,
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an isolated app; omitted settings retain normal environment validation."""
    application = FastAPI(title="Scholar RAG Agent", version="0.1.0")
    application.exception_handler(RequestValidationError)(_evidence_validation_error)
    application.state.container = create_container(settings)
    application.include_router(core_router)
    application.include_router(retrieval_router)
    application.include_router(documents_router)
    application.include_router(collections_router)
    application.include_router(screening_router)
    application.include_router(evidence_router)
    application.include_router(bibliography_router)
    application.include_router(runs_router)
    application.include_router(comparisons_router)
    application.include_router(corpus_drift_router)
    application.include_router(research_router)
    application.include_router(reviews_router)
    application.include_router(annotations_router)
    return application
