# Preregistration Flag Detector Guide

![Preregistration flag detector demo](../assets/preregistration-flag-detector.gif)

`PreregistrationFlagDetector` scans abstract / methods / registration text for
offline preregistration cues (`clinicaltrials.gov`, `NCT########`, OSF /
`osf.io`, `ISRCTN…`, `preregistration` / `pre-registration` /
`preregistered`). It returns advisory flags with reasons and matched cues. It
never calls the network.

Distinct from `FundingDisclosureFlagger` (funder/grant cues) and
`PrismaScreeningChecklist` (HITL PRISMA screening rows). Fills an Elicit /
Consensus / SciSpace / PaperQA preregistration-cue surfacing gap with
deterministic heuristics. Optional later narrative can use GPT-6 Astra /
Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro.

## Usage

```python
from retrieval.preregistration_flag import PreregistrationFlagDetector

flags = PreregistrationFlagDetector().detect(
    [
        {
            "paper_id": "p1",
            "abstract": "Registered at clinicaltrials.gov (NCT01234567).",
        },
        {
            "paper_id": "clear",
            "abstract": "We study transformers without registry identifiers.",
        },
    ]
)
for flag in flags:
    print(flag.paper_id, flag.flagged, flag.reasons, flag.matched_cues)
```

Empty paper lists return an empty tuple. Inputs are not mutated.
