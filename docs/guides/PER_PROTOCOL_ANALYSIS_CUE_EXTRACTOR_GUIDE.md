# PerProtocolAnalysisCueExtractor Guide

![PerProtocolAnalysisCueExtractor](../assets/per-protocol-analysis-cue-extractor.gif)

Offline advisory cue extractor. Never calls the network.
Closes Elicit / Consensus / CONSORT per-protocol analysis gaps.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from IntentionToTreatCueExtractor / ProtocolDeviationCueExtractor.

## Usage

```python
from retrieval.per_protocol_analysis_cues import PerProtocolAnalysisCueExtractor

rows = PerProtocolAnalysisCueExtractor().extract(
    [{"paper_id": "p1", "methods": "Primary analysis used a per-protocol population."}]
)
print(rows[0].flagged, rows[0].cues)
```
