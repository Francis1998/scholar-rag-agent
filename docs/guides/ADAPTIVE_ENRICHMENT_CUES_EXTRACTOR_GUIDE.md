# AdaptiveEnrichmentCueExtractor Guide

![AdaptiveEnrichmentCueExtractor](../../assets/demo/adaptive-enrichment-cues.gif)

Offline cue extractor.
Gap vs Elicit/Consensus/Connected Papers adaptive-enrichment cue extractors. Distinct from `SubgroupAnalysisCueExtractor` / `ProtocolAmendmentCueExtractor`.

Optional narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.adaptive_enrichment_cues import AdaptiveEnrichmentCueExtractor

rows = AdaptiveEnrichmentCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss adaptive enrichment in the cohort."}]
)
print(rows[0].cues)
```

## Safety

Advisory only. No network I/O.
