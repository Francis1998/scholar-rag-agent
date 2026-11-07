# ProtopathicBiasCueExtractor Guide

![ProtopathicBiasCueExtractor](../assets/protopathic-bias-cues.gif)

Offline cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE protopathic-bias cue extractors.

Optional narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `ReverseCausationCueExtractor` / `IndicationBiasCueExtractor`.

## Usage

```python
from retrieval.protopathic_bias_cues import ProtopathicBiasCueExtractor

rows = [{"paper_id": "p1", "abstract": "We discuss protopathic bias."}]
print(ProtopathicBiasCueExtractor().extract(rows)[0].flagged)
```

## Safety

Advisory cues only. Humans decide.
