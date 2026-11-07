# FollowUpDurationCueExtractor Guide

![FollowUpDurationCueExtractor](../assets/follow-up-duration-cue-extractor.gif)

Offline evidence cue extractor. Never calls the network.

Optional later narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from AttritionRateCueExtractor and SampleSizeHintExtractor.

## Usage

```python
from retrieval.follow_up_duration_cues import FollowUpDurationCueExtractor

rows = FollowUpDurationCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Median follow-up was 24 months."}]
)
assert rows[0].flagged is True
```
