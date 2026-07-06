"""OpenAlex API ingestion connector.

OpenAlex (https://openalex.org) is a large, open catalog of scholarly works and
a common complement to arXiv and Semantic Scholar. Its ``works`` endpoint stores
the abstract as an *inverted index* (a mapping of word to the positions where it
occurs) rather than plain text, so this connector reconstructs the readable
abstract before normalizing the record into a :class:`Document`.
"""

import httpx

from ingestion.chunking import stable_id
from retrieval.models import Document

OPENALEX_BASE_URL = "https://api.openalex.org/works"


class OpenAlexConnector:
    """Fetch and normalize OpenAlex work records."""

    def __init__(self, mailto: str | None = None) -> None:
        """Create a connector.

        Args:
            mailto: Optional contact email added to requests so OpenAlex routes
                traffic to its faster, polite API pool.
        """
        self._mailto = mailto

    async def fetch_work(self, work_id: str) -> Document:
        """Return one normalized OpenAlex work by id.

        Args:
            work_id: OpenAlex work id (e.g. ``W2741809807``), a full OpenAlex
                URL, or a DOI accepted by the works endpoint.

        Returns:
            Normalized document with the reconstructed abstract as its text.
        """
        params = {"mailto": self._mailto} if self._mailto else None
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{OPENALEX_BASE_URL}/{work_id}", params=params)
            response.raise_for_status()
        data = response.json()
        title = str(data.get("title") or data.get("display_name") or "Untitled OpenAlex work")
        abstract = self._reconstruct_abstract(data.get("abstract_inverted_index"))
        source = str(data.get("id") or data.get("doi") or work_id)
        return Document(
            document_id=stable_id(source, "doc"),
            title=" ".join(title.split()),
            text=abstract,
            source=source,
            metadata={
                "source_type": "openalex",
                "year": str(data.get("publication_year") or ""),
            },
        )

    @staticmethod
    def _reconstruct_abstract(inverted_index: object) -> str:
        """Reconstruct abstract text from an OpenAlex inverted index.

        OpenAlex encodes abstracts as ``{word: [positions, ...]}``. The words are
        placed at their recorded positions and joined with single spaces. Missing
        or malformed indexes yield an empty string, and gaps left by
        non-contiguous positions are dropped rather than rendered as blanks.

        Args:
            inverted_index: The ``abstract_inverted_index`` field, expected to be
                a mapping of word to a list of integer positions.

        Returns:
            The reconstructed abstract, or an empty string when unavailable.
        """
        if not isinstance(inverted_index, dict) or not inverted_index:
            return ""
        positioned: dict[int, str] = {}
        for word, positions in inverted_index.items():
            if not isinstance(word, str) or not isinstance(positions, list):
                continue
            for position in positions:
                if isinstance(position, int) and not isinstance(position, bool):
                    positioned[position] = word
        ordered_words = [positioned[index] for index in sorted(positioned)]
        return " ".join(ordered_words)
