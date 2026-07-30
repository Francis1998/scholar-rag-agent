"""Paper ingestion connectors and normalization pipeline."""

from ingestion.clinicaltrials import ClinicalTrialsConnector
from ingestion.crossref_events import CrossrefEventsConnector
from ingestion.dryad import DryadConnector
from ingestion.openalex_topics import OpenAlexTopicsConnector
from ingestion.orcid import OrcidConnector
from ingestion.pmc import PmcConnector
from ingestion.retraction_watch import RetractionWatchConnector
from ingestion.semantic_scholar import SemanticScholarConnector
from ingestion.unpaywall import UnpaywallConnector

__all__ = [
    "ClinicalTrialsConnector",
    "CrossrefEventsConnector",
    "DryadConnector",
    "OpenAlexTopicsConnector",
    "OrcidConnector",
    "PmcConnector",
    "RetractionWatchConnector",
    "SemanticScholarConnector",
    "UnpaywallConnector",
]
