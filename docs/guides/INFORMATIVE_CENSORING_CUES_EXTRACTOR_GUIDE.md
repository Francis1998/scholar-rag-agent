# InformativeCensoringCueExtractor Guide

![InformativeCensoringCueExtractor](../../assets/demo/informative-censoring-cues.gif)

Offline advisory bias/method cue extractor.
Gap vs Elicit/Consensus/Connected Papers informative-censoring cue extractors. Distinct from `IntervalCensoringCueExtractor` / `LossToFollowupBiasCueExtractor`.

Optional narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.informative_censoring_cues import InformativeCensoringCueExtractor

rows = InformativeCensoringCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss informative censoring."}]
)
print(rows[0].cues)
```

## Safety

Advisory only. No network I/O.
