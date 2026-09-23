# SurrogateEndpointCueExtractor Guide

![SurrogateEndpointCueExtractor](../assets/surrogate-endpoint-cue-extractor.gif)

Offline evidence cue extractor. Never calls the network.

Optional later narrative via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from PrimaryEndpointCueExtractor and EffectSizeHintExtractor.

## Usage

```python
from retrieval.surrogate_endpoint_cues import SurrogateEndpointCueExtractor

rows = SurrogateEndpointCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "The primary surrogate endpoint was ORR."}]
)
assert rows[0].flagged is True
```
