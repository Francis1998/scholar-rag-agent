"""Deterministic dense embedding utilities."""

import hashlib
import math

from retrieval.sparse import tokenize


class HashEmbeddingModel:
    """Small deterministic embedding model for local demos and tests."""

    def __init__(self, dimensions: int = 64) -> None:
        """Create a hashing embedder with a fixed dimensionality."""
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        """Embed text into a normalized hashing vector.

        Tokens are produced by the shared :func:`retrieval.sparse.tokenize`
        helper so a term is bucketed identically here and in the BM25 sparse
        retriever. A raw ``str.split`` left surrounding punctuation attached
        (``"retrieval."`` hashed to a different dimension than ``"retrieval"``),
        so the same word was a hit in sparse retrieval but a miss in the dense
        vector the two are fused with.
        """
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / magnitude for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two already normalized vectors."""
    return sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )
