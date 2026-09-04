# MinAbstractLengthGate Guide

![MinAbstractLengthGate demo](../assets/min-abstract-length-gate.gif)

Local retrieval postprocessor for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines (not a DOI connector).

Inspired by LlamaIndex/Haystack length filters that drop stub abstracts before generation.

## Usage

```python
from retrieval.min_abstract_length_gate import MinAbstractLengthGate
```

See unit tests for edge cases.
