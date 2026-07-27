"""Paper ingestion connectors and normalization pipeline."""

from ingestion.orcid import OrcidConnector
from ingestion.pmc import PmcConnector

__all__ = ["OrcidConnector", "PmcConnector"]
