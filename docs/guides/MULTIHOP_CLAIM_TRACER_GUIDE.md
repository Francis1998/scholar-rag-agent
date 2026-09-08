# Multi-Hop Claim Tracer Guide

![Multi-hop claim tracer demo](../assets/multihop-claim-tracer.gif)

`MultiHopClaimTracer` builds an inspectable claim path:
**claim → supporting spans → cited documents**. It is a report/path producer,
not a retrieval gate.

Distinct from:

- `MultiHopRetriever` — entity-chain retrieval over a graph store
- `ClaimVerificationGate` — support / refuse groundedness gate
- `CitationGraphExpander` — citation-edge expansion of seed hits

Fills a PaperQA-style claim verification UI gap with a local deterministic hop
report. Optional later narrative can use GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.

## Usage

```python
from retrieval.multihop_claim_tracer import MultiHopClaimTracer

tracer = MultiHopClaimTracer(support_threshold=0.35, max_span_hops=3)
trace = tracer.trace(claim, ranked_hits)
print(trace.to_markdown())
```

Each hop records type (`claim`, `supporting_span`, `cited_document`), label,
lexical coverage score, and optional `document_id` / `chunk_id` provenance.
