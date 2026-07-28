"""Paper ingestion connectors and normalization pipeline."""

from ingestion.orcid import OrcidConnector
from ingestion.pmc import PmcConnector
from ingestion.retraction_watch import RetractionWatchConnector
from ingestion.semantic_scholar import SemanticScholarConnector
from ingestion.unpaywall import UnpaywallConnector

__all__ = [
    "OrcidConnector",
    "PmcConnector",
    "RetractionWatchConnector",
    "SemanticScholarConnector",
    "UnpaywallConnector",
]
