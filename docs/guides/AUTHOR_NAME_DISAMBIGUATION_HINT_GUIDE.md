# Author Name Disambiguation Hint Guide

![Author name disambiguation hint demo](../assets/author-name-disambiguation-hint.gif)

`AuthorNameDisambiguationHint` groups near-duplicate author strings offline
(for example `J Smith` / `J. Smith` with `John Smith`) using shared-surname and
compatible given-name initial heuristics. Singletons are retained. Distinct
surnames never merge.

This module replaces a redundant VenuePrestigeCalibrator: `VenueTierBooster`
already maps venues to prestige tiers with soft score blending. Fills an
OpenAlex / Semantic Scholar author-disambiguation hint gap. Optional later
narrative can use GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro.

## Usage

```python
from retrieval.author_disambiguation import AuthorNameDisambiguationHint

groups = AuthorNameDisambiguationHint().group(["J Smith", "John Smith", "Jane Doe", "J. Smith"])
for group in groups:
    print(group.canonical, group.members, group.reason)
```

Empty author lists return an empty tuple. Inputs are not mutated.
