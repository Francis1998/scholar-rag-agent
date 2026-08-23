# Citation Groundedness Score Guide

![Citation groundedness score demo](../assets/citation_groundedness_score.gif)

`CitationGroundednessScorer` checks whether the inline citation markers in a
draft answer actually point to evidence that supports the sentence they
appear in. It is inspired by RAGAS `context_precision` and TruLens
groundedness metrics, and requires no LLM or network call. Verified answers
can be refined or explained with GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, or
Kimi K2.

## How it differs from `ClaimVerificationGate`

- `ClaimVerificationGate` splits an answer into claim sentences and checks
  whether *any* retrieved chunk lexically supports each claim, independent
  of citation syntax.
- `CitationGroundednessScorer` only scores sentences that contain an
  explicit citation marker, and resolves each marker to the *specific*
  source(s) it names before measuring overlap. A sentence can be lexically
  supported by the corpus in general yet still fail this check if it cites
  the wrong source.

The two are complementary: run `ClaimVerificationGate` for general
groundedness, and `CitationGroundednessScorer` to audit whether citations are
attributed correctly.

## Supported citation markers

- Numeric bracket markers, resolved to the *n*-th retrieved result
  (1-indexed): `[1]`, `[2]`, or comma-separated lists like `[1, 2]`.
- Author-year markers, resolved by matching the `authors` and `year` chunk
  metadata populated by ingestion connectors: `(Smith, 2020)`,
  `(Smith et al., 2020)`, `(Smith and Jones, 2020)`, `(Smith & Jones, 2020)`.

## How scoring works

1. Split the answer into sentences on `.`, `!`, `?`, or newlines.
2. Find every citation marker in each sentence.
3. Resolve numeric markers to result indices; resolve author-year markers by
   matching the surname against `authors` metadata and the year prefix
   against `year` metadata.
4. Strip citation markers from the sentence and extract its non-stopword
   terms.
5. Measure lexical overlap between the sentence terms and each resolved
   candidate's title and text; keep the best overlap across candidates.
6. Mark the citation `grounded` when it resolves to at least one candidate,
   the sentence has content terms, and the best overlap meets
   `overlap_threshold`.

```text
overlap_score = sentence terms in cited chunk / all sentence terms
groundedness  = grounded citations / all citations found
```

The default `overlap_threshold` is `0.3`. Sentences without any citation
marker are excluded from the report entirely.

## Example

```python
from retrieval.citation_groundedness_score import CitationGroundednessScorer

scorer = CitationGroundednessScorer(overlap_threshold=0.3)
report = scorer.score(draft_answer, retrieved_results)

if report.groundedness < 1.0:
    for citation in report.citations:
        if not citation.grounded:
            flag_misattributed_citation(citation.marker, citation.sentence)
```

Each mention includes the citation marker text, the sentence it appeared in,
the resolved candidate chunk ids, a boolean grounded flag, and the overlap
score. Unresolved indices (out of range) or unmatched author-year markers
yield an empty candidate tuple and are always ungrounded.
