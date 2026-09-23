"""Read-only, bounded inspection of exactly two saved completed research runs."""

from fastapi import APIRouter, HTTPException, Request, Response

from agent.comparison_models import RunComparison
from api.dependencies import AppContainer
from storage.run_comparison import RunComparisonError

router = APIRouter()
_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


@router.get(
    "/runs/{baseline_run_id}/compare/{candidate_run_id}",
    response_model=RunComparison,
    responses={
        404: {"description": "Unknown baseline or candidate run; detail.side identifies it"},
        409: {"description": "Incomplete, failed, legacy, or invalid evidence on either side"},
    },
)
def compare_runs(
    request: Request, response: Response, baseline_run_id: str, candidate_run_id: str
) -> RunComparison:
    """Inspect baseline -> candidate changes, without rerunning retrieval or generation."""
    container: AppContainer = request.app.state.container
    response.headers.update(_HEADERS)
    try:
        return container.run_comparator.compare(baseline_run_id, candidate_run_id)
    except RunComparisonError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": str(exc), "side": exc.side},
            headers=_HEADERS,
        ) from exc
