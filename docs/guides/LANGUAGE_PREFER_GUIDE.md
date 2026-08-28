# Language Prefer Guide

![Language prefer demo](../assets/language_prefer.gif)

`LanguagePreferencer` prefers retrieved results whose metadata language matches
a configured preferred set. It is inspired by Haystack `LanguageClassifier`
signals and LlamaIndex metadata filters, but is a local retrieval postprocessor
(not a DOI connector). Preferred evidence can then feed GPT-5.5, Claude Sonnet
4.6, Gemini 3.x, or Kimi K2.

## How preference works

Language is read from `language` or `lang` metadata (case-insensitive). The
default preferred set is `en`.

### `mode="boost"` (default)

```text
lang_signal = 1.0 if language in preferred else 0.0
new_score = (1 - alpha) * old_score + alpha * lang_signal
```

Results are sorted by `new_score` descending (stable for ties). Returned rows
use `retriever="language_prefer"` and append the prior retriever to `path`.

### `mode="filter"`

- If at least one hit matches a preferred language, keep only matching rows
  (original identity/order/score).
- If none match, keep all inputs unchanged.
- Input objects are never mutated.

## Example

```python
from retrieval.language_prefer import LanguagePreferencer

preferencer = LanguagePreferencer(
    preferred_languages=["en"],
    alpha=0.3,
    mode="boost",
)
preferred = preferencer.prefer(retrieved_results, top_k=10)

filtered = LanguagePreferencer(mode="filter").prefer(retrieved_results)
```

## Configuration notes

- `preferred_languages` defaults to `("en",)` and must contain at least one
  non-blank value after stripping.
- `alpha` must be a finite number in `[0.0, 1.0]` (default `0.3`; boost mode).
- `mode` must be `"boost"` or `"filter"`.
- `top_k=None` keeps every survivor; empty input returns `[]`.
