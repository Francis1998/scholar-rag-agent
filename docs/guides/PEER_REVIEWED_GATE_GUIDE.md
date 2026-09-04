# PeerReviewedGate Guide

![PeerReviewedGate demo](../assets/peer-reviewed-gate.gif)

Local retrieval postprocessor for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines (not a DOI connector).

Inspired by scholarly RAG stacks that prefer peer-reviewed sources over preprints when summarizing evidence.

## Usage

```python
from retrieval.peer_reviewed_gate import PeerReviewedGate
```

See unit tests for edge cases.
