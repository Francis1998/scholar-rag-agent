# SourceAuthorityGate Guide

![SourceAuthorityGate demo](../assets/source-authority-gate.gif)

Local retrieval postprocessor for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines (not a DOI connector).

Inspired by LlamaIndex/Haystack metadata boost and filter postprocessors. Boosts (and optionally filters) hits using ``source_authority`` / venue-tier ``high`` / ``medium`` / ``low`` metadata, with an optional venue map and ``min_authority`` floor.

## Usage

```python
from retrieval.source_authority_gate import SourceAuthorityGate
```

See unit tests for edge cases.
