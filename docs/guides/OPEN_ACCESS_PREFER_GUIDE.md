# Open Access Prefer Guide

![Open access prefer demo](../assets/open_access_prefer.gif)

`OpenAccessPreferencer` prefers open-access retrieved results either by
blending an OA signal into the score or by soft-filtering closed hits when any
OA evidence exists. It is inspired by LlamaIndex/Haystack metadata
postprocessors and Unpaywall / OpenAlex OA preference in scholarly search, but
is a local retrieval postprocessor (not a DOI connector). Preferred evidence
can then feed GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, or Kimi K2.

## How preference works

Open-access is truthy when any of `open_access`, `is_oa`, or `oa` metadata
values (case-insensitive) is one of: `true`, `1`, `yes`, `y`, `oa`, `open`,
`open_access`.

### `mode="boost"` (default)

```text
oa_signal = 1.0 if open_access else 0.0
new_score = (1 - alpha) * old_score + alpha * oa_signal
```

Results are sorted by `new_score` descending (stable for ties). Returned rows
use `retriever="open_access_prefer"` and append the prior retriever to `path`.

### `mode="filter"`

- If at least one hit is OA, keep only OA rows (original identity/order/score).
- If none are OA, keep all inputs unchanged.
- Input objects are never mutated.

## Example

```python
from retrieval.open_access_prefer import OpenAccessPreferencer

preferencer = OpenAccessPreferencer(alpha=0.3, mode="boost")
preferred = preferencer.prefer(retrieved_results, top_k=10)

filtered = OpenAccessPreferencer(mode="filter").prefer(retrieved_results)
```

## Configuration notes

- `alpha` must be a finite number in `[0.0, 1.0]` (default `0.3`; boost mode).
- `mode` must be `"boost"` or `"filter"`.
- `top_k=None` keeps every survivor; empty input returns `[]`.
