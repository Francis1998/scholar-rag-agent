"""Offline author-name near-duplicate grouping hints.

Groups near-duplicate author strings such as ``J Smith`` vs ``John Smith``
using deterministic initial/surname heuristics. Never calls the network.
Replaces a redundant VenuePrestigeCalibrator because
:class:`~retrieval.venue_tier_boost.VenueTierBooster` already maps venues to
prestige tiers with soft score blending. Fills an OpenAlex / Semantic Scholar
author-disambiguation hint gap without an LLM call. Optional later narrative
can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[.,]+")


@dataclass(frozen=True)
class AuthorDisambiguationGroup:
    """One group of near-duplicate author name strings."""

    canonical: str
    members: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class _ParsedName:
    original: str
    surname: str
    given_tokens: tuple[str, ...]
    initials: tuple[str, ...]


class AuthorNameDisambiguationHint:
    """Group near-duplicate author strings offline.

    Offline author-name disambiguation hints for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 pipelines — OpenAlex/Semantic Scholar author-merge
    gap. Distinct from :class:`~retrieval.venue_tier_boost.VenueTierBooster`
    (venue prestige tiers already covered) and from network ORCID connectors.
    Never mutates inputs.
    """

    def group(
        self,
        authors: Sequence[str],
    ) -> tuple[AuthorDisambiguationGroup, ...]:
        """Return near-duplicate author groups for ``authors``.

        Names that share a surname and compatible given-name initials
        (``J`` / ``J.`` with ``John``) are merged. Distinct surnames never
        merge. Singletons are included. Empty input yields an empty tuple.
        Inputs are not mutated.
        """
        if not authors:
            return ()

        parsed = [self._parse(name) for name in authors if str(name).strip()]
        if not parsed:
            return ()

        parent = list(range(len(parsed)))

        def find_root(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find_root(i), find_root(j)
            if ri != rj:
                parent[rj] = ri

        for i in range(len(parsed)):
            for j in range(i + 1, len(parsed)):
                if self._compatible(parsed[i], parsed[j]):
                    union(i, j)

        buckets: dict[int, list[int]] = {}
        for i in range(len(parsed)):
            buckets.setdefault(find_root(i), []).append(i)

        groups: list[AuthorDisambiguationGroup] = []
        for members in buckets.values():
            originals = tuple(parsed[i].original for i in members)
            canonical = self._pick_canonical([parsed[i] for i in members])
            if len(members) == 1:
                reason = "Singleton author string."
            else:
                reason = (
                    "Near-duplicate via shared surname and compatible "
                    "given-name initial/full form."
                )
            groups.append(
                AuthorDisambiguationGroup(
                    canonical=canonical,
                    members=originals,
                    reason=reason,
                )
            )

        groups.sort(key=lambda g: (-len(g.members), g.canonical.casefold()))
        return tuple(groups)

    @staticmethod
    def _parse(name: str) -> _ParsedName:
        cleaned = _PUNCT.sub(" ", name.strip())
        cleaned = _WHITESPACE.sub(" ", cleaned).strip()
        parts = cleaned.split(" ")
        if len(parts) == 1:
            return _ParsedName(
                original=name.strip(),
                surname=parts[0].casefold(),
                given_tokens=(),
                initials=(),
            )
        surname = parts[-1].casefold()
        given = tuple(parts[:-1])
        initials = tuple(
            token[0].casefold() for token in given if token and token[0].isalpha()
        )
        return _ParsedName(
            original=name.strip(),
            surname=surname,
            given_tokens=tuple(t.casefold() for t in given),
            initials=initials,
        )

    @staticmethod
    def _compatible(left: _ParsedName, right: _ParsedName) -> bool:
        if not left.surname or not right.surname:
            return False
        if left.surname != right.surname:
            return False
        if not left.initials or not right.initials:
            return not left.given_tokens or not right.given_tokens
        shorter, longer = (
            (left.initials, right.initials)
            if len(left.initials) <= len(right.initials)
            else (right.initials, left.initials)
        )
        if shorter != longer[: len(shorter)]:
            return False
        for a, b in zip(left.given_tokens, right.given_tokens, strict=False):
            if len(a) == 1 or len(b) == 1:
                if a[0] != b[0]:
                    return False
            elif a != b:
                return False
        return True

    @staticmethod
    def _pick_canonical(members: list[_ParsedName]) -> str:
        return max(
            members,
            key=lambda m: (
                sum(len(t) for t in m.given_tokens),
                len(m.original),
                -members.index(m),
            ),
        ).original
