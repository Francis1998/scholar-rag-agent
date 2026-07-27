"""Paper ingestion connectors and normalization pipeline."""

from ingestion.orcid import OrcidConnector
from ingestion.pmc import PmcConnector
from ingestion.semantic_scholar import SemanticScholarConnector

__all__ = ["OrcidConnector", "PmcConnector", "SemanticScholarConnector"]
