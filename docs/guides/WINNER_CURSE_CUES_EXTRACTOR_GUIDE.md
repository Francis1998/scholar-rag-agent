# WinnerCurseCueExtractor Guide

![WinnerCurseCueExtractor](../../assets/demo/winner-curse-cues.gif)

Offline advisory bias/method cue extractor.
Gap vs Elicit/Consensus/Connected Papers winner's-curse cue extractors. Distinct from `PublicationBiasCueExtractor` / `NoveltyBiasCueExtractor`.

Optional narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.winner_curse_cues import WinnerCurseCueExtractor

rows = WinnerCurseCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss winner curse."}]
)
print(rows[0].cues)
```

## Safety

Advisory only. No network I/O.
