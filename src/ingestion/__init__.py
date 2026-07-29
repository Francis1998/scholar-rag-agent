"""Paper ingestion connectors and normalization pipeline."""

from ingestion.crossref_events import CrossrefEventsConnector
from ingestion.dryad import DryadConnector
from ingestion.orcid import OrcidConnector
from ingestion.pmc import PmcConnector
from ingestion.retraction_watch import RetractionWatchConnector
from ingestion.semantic_scholar import SemanticScholarConnector
from ingestion.unpaywall import UnpaywallConnector

__all__ = [
    "CrossrefEventsConnector",
    "DryadConnector",
    "OrcidConnector",
    "PmcConnector",
    "RetractionWatchConnector",
    "SemanticScholarConnector",
    "UnpaywallConnector",
]
