# MendelianRandomizationWeakCueExtractor Guide

![MendelianRandomizationWeakCueExtractor](../../assets/demo/mendelian-randomization-weak-cues.gif)

Offline cue extractor.
Gap vs Elicit/Consensus/Connected Papers weak Mendelian-randomization cue extractors. Distinct from `InstrumentStrengthCueExtractor` / `GComputationCueExtractor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.mendelian_randomization_weak_cues import MendelianRandomizationWeakCueExtractor

rows = MendelianRandomizationWeakCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss weak instrument."}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O.
