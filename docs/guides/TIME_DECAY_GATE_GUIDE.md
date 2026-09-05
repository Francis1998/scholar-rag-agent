# TimeDecayGate Guide

![TimeDecayGate demo](../assets/time-decay-gate.gif)

Local retrieval postprocessor for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines (not a DOI connector).

Inspired by Haystack/Elasticsearch temporal decay postprocessors and LlamaIndex freshness rerankers. Multiplies relevance by ``0.5 ** (age_days / half_life_days)`` using ``published_at`` / ``year`` / ``date`` metadata (default half-life 365 days), then re-ranks.

## Usage

```python
from retrieval.time_decay_gate import TimeDecayGate
```

See unit tests for edge cases.
