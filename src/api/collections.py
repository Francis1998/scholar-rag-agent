"""Local/trusted metadata management and fail-closed request scope resolution."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, Response

from api.dependencies import AppContainer
from api.schemas import QueryRequest
from storage.paper_collections import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    MAX_REVISION,
    CollectionError,
    CollectionId,
    CollectionPage,
    CollectionReplacement,
    CollectionSelection,
    PaperCollection,
)

logger = logging.getLogger(__name__)
COLLECTION_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
router = APIRouter(
    prefix="/collections",
    tags=["collections"],
    responses={
        404: {"description": "Unknown collection"},
        409: {"description": "Name/revision conflict or invalid saved collection"},
        422: {"description": "Invalid request or unknown document IDs"},
        503: {"description": "Collection storage unavailable"},
    },
)


@contextmanager
def collection_errors() -> Iterator[None]:
    """Log only a stable code, not a private name, membership, or SQLite diagnostic."""
    try:
        yield
    except CollectionError as exc:
        logger.warning("Collection operation failed: %s", exc.code)
        detail: dict[str, object] = {"code": exc.code, "message": str(exc)}
        if exc.missing_document_ids:
            detail["missing_document_ids"] = list(exc.missing_document_ids)
        raise HTTPException(
            status_code=exc.status_code, detail=detail, headers=COLLECTION_HEADERS
        ) from exc


def resolve_document_scope(
    container: AppContainer, payload: QueryRequest
) -> tuple[str, ...] | None:
    """Resolve once synchronously; neither the shared runner nor retriever owns this scope."""
    if payload.collection_id is None:
        return payload.document_ids
    with collection_errors():
        return container.paper_collections.resolve(payload.collection_id)


@router.post("", response_model=PaperCollection, status_code=201)
def create_collection(
    request: Request, payload: CollectionSelection, response: Response
) -> PaperCollection:
    container: AppContainer = request.app.state.container
    with collection_errors():
        collection = container.paper_collections.create(
            name=payload.name, document_ids=payload.document_ids
        )
    response.headers.update(COLLECTION_HEADERS)
    response.headers["Location"] = f"/collections/{collection.collection_id}"
    return collection


@router.get("", response_model=CollectionPage)
def list_collections(
    request: Request,
    response: Response,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    cursor: Annotated[CollectionId | None, Query()] = None,
) -> CollectionPage:
    container: AppContainer = request.app.state.container
    with collection_errors():
        page = container.paper_collections.list_collections(limit=limit, cursor=cursor)
    response.headers.update(COLLECTION_HEADERS)
    return page


@router.get("/{collection_id}", response_model=PaperCollection)
def get_collection(
    request: Request, collection_id: CollectionId, response: Response
) -> PaperCollection:
    container: AppContainer = request.app.state.container
    with collection_errors():
        collection = container.paper_collections.get(collection_id)
    response.headers.update(COLLECTION_HEADERS)
    return collection


@router.put("/{collection_id}", response_model=PaperCollection)
def replace_collection(
    request: Request,
    collection_id: CollectionId,
    payload: CollectionReplacement,
    response: Response,
) -> PaperCollection:
    container: AppContainer = request.app.state.container
    with collection_errors():
        collection = container.paper_collections.replace(
            collection_id,
            name=payload.name,
            document_ids=payload.document_ids,
            expected_revision=payload.expected_revision,
        )
    response.headers.update(COLLECTION_HEADERS)
    return collection


@router.delete("/{collection_id}", status_code=204)
def delete_collection(
    request: Request,
    collection_id: CollectionId,
    expected_revision: Annotated[int, Query(ge=1, le=MAX_REVISION)],
) -> Response:
    container: AppContainer = request.app.state.container
    with collection_errors():
        container.paper_collections.delete(collection_id, expected_revision=expected_revision)
    return Response(status_code=204, headers=COLLECTION_HEADERS)
