# WillRogersPhenomenonCueExtractor Guide

![WillRogersPhenomenonCueExtractor](../assets/will-rogers-phenomenon-cues.gif)

Offline cue extractor for deterministic title/abstract/methods patterns.
Gap vs Elicit/Consensus/Connected Papers Will-Rogers / stage-migration cue extractors. Distinct from `LeadTimeBiasCueExtractor` / `LengthBiasCueExtractor`.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.will_rogers_phenomenon_cues import WillRogersPhenomenonCueExtractor

rows = WillRogersPhenomenonCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Authors describe will rogers phenomenon after restaging."}]
)
assert rows[0].flagged is True
```

## Safety

Advisory only. No network I/O. Humans decide.
