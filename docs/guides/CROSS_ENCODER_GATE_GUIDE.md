# CrossEncoderGate Guide

![demo](../assets/cross_encoder_gate.gif)

Gap fill vs SentenceTransformer cross-encoder gates without requiring model downloads.

Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 scholarly RAG pipelines.
Not a DOI connector.

## Usage

```python
from retrieval.cross_encoder_gate import CrossEncoderGate

stage = CrossEncoderGate()
# call stage.gate(...)
```

See unit tests for edge cases.
