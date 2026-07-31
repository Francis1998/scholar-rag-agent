"""Paper ingestion connectors and normalization pipeline."""

from ingestion.biorxiv_collections import BioRxivCollectionsConnector
from ingestion.clinicaltrials import ClinicalTrialsConnector
from ingestion.crossref_events import CrossrefEventsConnector
from ingestion.dryad import DryadConnector
from ingestion.openalex_authors import OpenAlexAuthorsConnector
from ingestion.openalex_topics import OpenAlexTopicsConnector
from ingestion.orcid import OrcidConnector
from ingestion.pmc import PmcConnector
from ingestion.pmc_oa import PmcOaPackageConnector
from ingestion.retraction_watch import RetractionWatchConnector
from ingestion.semantic_scholar import SemanticScholarConnector
from ingestion.unpaywall import UnpaywallConnector
from ingestion.wikidata_scholarly import WikidataScholarlyConnector

__all__ = [
    "BioRxivCollectionsConnector",
    "ClinicalTrialsConnector",
    "CrossrefEventsConnector",
    "DryadConnector",
    "OpenAlexAuthorsConnector",
    "OpenAlexTopicsConnector",
    "OrcidConnector",
    "PmcConnector",
    "PmcOaPackageConnector",
    "RetractionWatchConnector",
    "SemanticScholarConnector",
    "UnpaywallConnector",
    "WikidataScholarlyConnector",
]
