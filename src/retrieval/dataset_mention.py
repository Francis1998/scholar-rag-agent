"""Extract curated dataset mentions from paper title and abstract text.

Offline lexicon matching plus nearby ``dataset`` heuristics surfaces
benchmarks such as ImageNet, CIFAR, MIMIC, SQuAD, GLUE, and PubMedQA without
a network call. Fills a PapersWithCode dataset-surfacing gap. Optional later
narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

Distinct from live PapersWithCode connectors and from generic keyword boosters
that do not canonicalize dataset names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WHITESPACE = re.compile(r"\s+")

# Canonical name -> alternate spellings / common variants (case-insensitive).
_DATASET_LEXICON: dict[str, tuple[str, ...]] = {
    "ImageNet": ("imagenet", "image-net", "image net"),
    "CIFAR": ("cifar", "cifar-10", "cifar10", "cifar-100", "cifar100"),
    "MIMIC": ("mimic", "mimic-iii", "mimic-iv", "mimic iii", "mimic iv"),
    "SQuAD": ("squad", "squad 1.1", "squad 2.0", "squad2"),
    "GLUE": ("glue", "glue benchmark"),
    "SuperGLUE": ("superglue", "super-glue", "super glue"),
    "PubMedQA": ("pubmedqa", "pubmed qa", "pubmed-qa"),
    "MS MARCO": ("ms marco", "msmarco", "ms-marco"),
    "WikiText": ("wikitext", "wiki-text", "wikitext-103", "wikitext-2"),
    "MNIST": ("mnist", "fashion-mnist", "fashion mnist"),
    "COCO": ("coco", "ms coco", "mscoco", "ms-coco"),
    "KITTI": ("kitti",),
    "ADE20K": ("ade20k", "ade20", "ade 20k"),
    "LibriSpeech": ("librispeech", "libri-speech", "libri speech"),
    "HotpotQA": ("hotpotqa", "hotpot qa", "hotpot-qa"),
    "TriviaQA": ("triviaqa", "trivia qa", "trivia-qa"),
    "Natural Questions": ("natural questions", "nq dataset", "nq-open"),
    "BioASQ": ("bioasq", "bio-asq"),
    "MedQA": ("medqa", "med-qa", "med qa"),
    "CheXpert": ("chexpert", "chex-pert"),
}

_DATASET_NEARBY = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_-]{1,40})\s+(?:dataset|benchmark|corpus)\b"
    r"|"
    r"\b(?:dataset|benchmark|corpus)\s+([A-Za-z][A-Za-z0-9_-]{1,40})\b",
    re.IGNORECASE,
)

# Names that are too generic to treat as datasets when only nearby heuristics fire.
_NEARBY_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "our",
        "new",
        "public",
        "large",
        "small",
        "open",
        "training",
        "test",
        "validation",
        "eval",
        "evaluation",
        "standard",
        "popular",
        "existing",
        "following",
        "proposed",
        "same",
        "other",
        "several",
        "multiple",
        "various",
        "available",
        "released",
        "used",
        "using",
        "based",
        "related",
        "benchmark",
        "corpus",
        "dataset",
    }
)


@dataclass(frozen=True)
class DatasetMentionReport:
    """Dataset mentions extracted from a paper title/abstract."""

    datasets: tuple[str, ...]
    reasons: tuple[str, ...]


class DatasetMentionIndexer:
    """Index dataset mentions via lexicon and nearby ``dataset`` cues.

    Offline indexer for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2
    pipelines — PapersWithCode-style dataset surfacing without network access.
    Matches a curated lexicon (ImageNet, CIFAR, MIMIC, SQuAD, GLUE, PubMedQA,
    and related benchmarks) and optional nearby ``dataset`` / ``benchmark`` /
    ``corpus`` heuristics. Distinct from live PapersWithCode connectors and
    from generic keyword boosters. Inputs are not mutated.
    """

    def __init__(self, *, nearby_heuristic: bool = True) -> None:
        """Create a dataset-mention indexer.

        Args:
            nearby_heuristic: When ``True``, also capture tokens adjacent to
                ``dataset`` / ``benchmark`` / ``corpus`` that are not already
                covered by the curated lexicon.
        """
        self._nearby_heuristic = nearby_heuristic
        self._patterns: list[tuple[str, re.Pattern[str]]] = []
        for canonical, aliases in _DATASET_LEXICON.items():
            # Prefer longer aliases first so CIFAR-100 wins over CIFAR.
            ordered = sorted(aliases, key=len, reverse=True)
            parts = [re.escape(alias) for alias in ordered]
            joined = "|".join(parts)
            pattern = re.compile(
                rf"(?<![A-Za-z0-9])(?:{joined})(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
            self._patterns.append((canonical, pattern))

    def index(self, title: str, abstract: str = "") -> DatasetMentionReport:
        """Return dataset mentions found in ``title`` and ``abstract``.

        Empty title and abstract yield an empty report. Lexicon hits are listed
        before nearby-heuristic names. Reasons explain each match source.
        """
        haystack = _WHITESPACE.sub(" ", f"{title or ''} {abstract or ''}").strip()
        if not haystack:
            return DatasetMentionReport(datasets=(), reasons=("no title or abstract text",))

        datasets: list[str] = []
        reasons: list[str] = []
        seen: set[str] = set()

        for canonical, pattern in self._patterns:
            match = pattern.search(haystack)
            if match is None:
                continue
            key = canonical.casefold()
            if key in seen:
                continue
            seen.add(key)
            datasets.append(canonical)
            reasons.append(f"lexicon match ({canonical} via '{match.group(0)}')")

        alias_keys = {
            alias.casefold() for aliases in _DATASET_LEXICON.values() for alias in aliases
        } | {name.casefold() for name in _DATASET_LEXICON}

        if self._nearby_heuristic:
            for match in _DATASET_NEARBY.finditer(haystack):
                raw = (match.group(1) or match.group(2) or "").strip()
                if not raw:
                    continue
                key = raw.casefold()
                if key in _NEARBY_STOP or key in alias_keys or key in seen:
                    continue
                seen.add(key)
                datasets.append(raw)
                reasons.append(f"nearby dataset cue ('{match.group(0)}')")

        if not datasets:
            return DatasetMentionReport(datasets=(), reasons=("no dataset mentions found",))
        return DatasetMentionReport(datasets=tuple(datasets), reasons=tuple(reasons))
