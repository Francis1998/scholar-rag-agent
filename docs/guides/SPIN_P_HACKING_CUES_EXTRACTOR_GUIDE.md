# SpinPHackingCueExtractor Guide

![SpinPHackingCueExtractor](../../assets/demo/spin-p-hacking-cues.gif)

Offline cue extractor for scholarly evidence screening.
Gap vs Elicit/Consensus/Connected Papers spin / p-hacking / forking-paths cue extractors. Distinct from `NoveltyBiasCueExtractor` / `PublicationBiasCueExtractor`.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.spin_p_hacking_cues import SpinPHackingCueExtractor

rows = SpinPHackingCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "abstract": "Authors show spin reporting and possible p-hacking concerns.",
        }
    ],
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory cues only. No network I/O. Humans decide.
