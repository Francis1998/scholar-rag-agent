"""Saved-run comparisons backed exclusively by the authoritative evidence exporter."""

from typing import Literal

from agent.comparison_models import RunComparison
from agent.evidence import EvidenceBundle
from agent.run_comparison import compare_bundles
from storage.evidence_export import EvidenceExporter, EvidenceExportError

ComparisonSide = Literal["baseline", "candidate"]


class RunComparisonError(EvidenceExportError):
    """Preserve export failure semantics while identifying the failing side safely."""

    def __init__(self, side: ComparisonSide, error: EvidenceExportError) -> None:
        super().__init__(error.code, str(error), error.status_code)
        self.side = side


class SavedRunComparator:
    """No corpus, model, settings, or writable run state is needed for comparison."""

    def __init__(self, exporter: EvidenceExporter) -> None:
        self._exporter = exporter

    def _export(self, run_id: str, side: ComparisonSide) -> EvidenceBundle:
        try:
            return self._exporter.export(run_id)
        except EvidenceExportError as exc:
            raise RunComparisonError(side, exc) from exc

    def compare(self, baseline_run_id: str, candidate_run_id: str) -> RunComparison:
        """Validate baseline first; any failure rejects the entire operation."""
        baseline = self._export(baseline_run_id, "baseline")
        candidate = (
            baseline
            if candidate_run_id == baseline_run_id
            else self._export(candidate_run_id, "candidate")
        )
        return compare_bundles(baseline, candidate)
