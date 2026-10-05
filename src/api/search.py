"""Strict, non-caching HTTP boundary for current-corpus literal passage search."""

import json
import logging
from collections.abc import Callable, Coroutine

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from api.dependencies import AppContainer
from storage.document_chunks import DocumentChunksError
from storage.literal_search import LiteralSearchError, LiteralSearchPage, LiteralSearchRequest
from storage.paper_collections import CollectionError

logger = logging.getLogger(__name__)
_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


class _SearchRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[None, None, Response]]:
        handler = super().get_route_handler()

        async def validated(request: Request) -> Response:
            try:
                json.loads((await request.body()).decode("utf-8-sig"))
                return await handler(request)
            except (RequestValidationError, UnicodeDecodeError, json.JSONDecodeError):
                logger.warning("Literal search request validation failed")
                return Response(
                    content=json.dumps(
                        {
                            "detail": {
                                "code": "invalid_search_request",
                                "message": "Provide a nonblank UTF-8 query (1-200 characters), "
                                "at most one nonnull scope, limit 1-50, and a returned cursor.",
                            }
                        }
                    ),
                    status_code=422,
                    media_type="application/json",
                    headers=_HEADERS,
                )

        return validated


router = APIRouter(route_class=_SearchRoute)


@router.post(
    "/research/search",
    response_model=LiteralSearchPage,
    responses={
        404: {"description": "Unknown collection"},
        409: {"description": "Invalid stored passage or stale/invalid collection"},
        413: {"description": "Stored-text read or serialized response limit exceeded"},
        422: {"description": "Invalid query, scope, limit, or mismatched cursor"},
        503: {"description": "Corpus or collection storage unavailable"},
        504: {"description": "Literal scan deadline exceeded; narrow the scope"},
    },
)
def literal_search(request: Request, payload: LiteralSearchRequest) -> Response:
    """Search exact text, not tokens, patterns, semantic similarity, or scientific relevance."""
    container: AppContainer = request.app.state.container
    try:
        result = container.literal_search.search(payload)
        content = result.to_json()
    except (LiteralSearchError, DocumentChunksError, CollectionError) as exc:
        logger.warning("Literal search failed: %s", exc.code)
        raise HTTPException(
            status_code=409 if isinstance(exc, DocumentChunksError) else exc.status_code,
            detail={"code": exc.code, "message": str(exc)},
            headers=_HEADERS,
        ) from exc
    return Response(content=content, media_type="application/json", headers=_HEADERS)
