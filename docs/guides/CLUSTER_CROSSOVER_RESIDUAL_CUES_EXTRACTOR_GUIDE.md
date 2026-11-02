# ClusterCrossoverResidualCueExtractor Guide

![ClusterCrossoverResidualCueExtractor](../../assets/demo/cluster-crossover-residual-cues.gif)

Offline cue extractor.
Gap vs Elicit/Consensus/Connected Papers cluster-crossover residual cue extractors. Distinct from `ClusterRandomizationCueExtractor` / `CrossoverDesignCueExtractor`.

Optional narrative via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.cluster_crossover_residual_cues import ClusterCrossoverResidualCueExtractor

rows = ClusterCrossoverResidualCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss cluster crossover residual."}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O.
