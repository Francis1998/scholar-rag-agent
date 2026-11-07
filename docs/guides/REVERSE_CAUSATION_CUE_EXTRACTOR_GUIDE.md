# ReverseCausationCueExtractor Guide

![ReverseCausationCueExtractor](../assets/reverse-causation-cues.gif)

Offline cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE reverse-causation cue extractors.

Optional narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `IndicationBiasCueExtractor` / `ConfoundingByIndicationCueExtractor`.

## Usage

```python
from retrieval.reverse_causation_cues import ReverseCausationCueExtractor

rows = [{"paper_id": "p1", "abstract": "We discuss reverse causation."}]
print(ReverseCausationCueExtractor().extract(rows)[0].flagged)
```

## Safety

Advisory cues only. Humans decide.
