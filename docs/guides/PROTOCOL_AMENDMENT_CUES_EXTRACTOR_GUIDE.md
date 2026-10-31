# ProtocolAmendmentCueExtractor Guide

![ProtocolAmendmentCueExtractor](../../assets/demo/protocol-amendment-cues.gif)

Offline deterministic cue extractor.
Gap vs Elicit/Consensus/Connected Papers protocol-amendment cue extractors. Distinct from `OutcomeSwitchingCueExtractor` / `WinnerCurseCueExtractor`.

Optional narrative polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.protocol_amendment_cues import ProtocolAmendmentCueExtractor

rows = ProtocolAmendmentCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss protocol amendment."}]
)
print(rows[0].cues)
```

## Safety

Advisory only. No network I/O.
