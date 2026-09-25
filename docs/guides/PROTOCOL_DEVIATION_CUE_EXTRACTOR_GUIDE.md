# ProtocolDeviationCueExtractor Guide

![ProtocolDeviationCueExtractor](../assets/protocol-deviation-cue-extractor.gif)

Offline evidence cue extractor. Never calls the network.

Optional later narrative via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from AttritionRateCueExtractor and IntentionToTreatCueExtractor.

## Usage

```python
from retrieval.protocol_deviation_cues import ProtocolDeviationCueExtractor

rows = ProtocolDeviationCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Several protocol deviations were recorded."}]
)
assert rows[0].flagged is True
```
