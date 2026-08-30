# Claim Density Boost Guide

![Claim Density Boost](../assets/claim_density_boost.gif)

Inspired by claim-centric reranking in scholarly RAG (PaperQA/LlamaIndex); boosts hits whose chunk text has higher density of claim-like sentences (not a DOI connector).

Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 answer stacks.

## Usage

```python
from retrieval.claim_density_boost import ClaimDensityBooster

processor = ClaimDensityBooster()
# call .boost(...) on hybrid SearchResult lists
```

## Safety

Local chunk text only — no network calls and no DOI connector side effects.
