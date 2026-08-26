# Required Metadata Gate Guide

![Required metadata gate demo](../assets/required_metadata_gate.gif)

`RequiredMetadataGate` drops retrieved results that are missing any of a
configured set of chunk metadata keys. It is inspired by Haystack
`MetadataRouter` and LlamaIndex `MetadataFilters`, but is a simple boolean
presence/non-empty check with no network or LLM call. Unlike
`TemporalFreshnessCutoff`, which interprets date-shaped fields, this gate only
verifies that each required key exists with a non-empty string value. Surviving
evidence can then feed GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, or Kimi K2.

## How filtering works

- Constructor takes `required_keys: Sequence[str]`.
- Empty `required_keys` is a pass-through (all results kept, including empty
  input → `[]`).
- A result is dropped when any required key is missing from `chunk.metadata`
  **or** the stored value strips to an empty string.
- Survivors keep original identity, score, and input order.

## Example

```python
from retrieval.required_metadata_gate import RequiredMetadataGate

gate = RequiredMetadataGate(required_keys=["doi", "year"])
filtered = gate.filter(retrieved_results)
```

## Configuration notes

- Pass a list or tuple of metadata key names (e.g. `doi`, `year`,
  `source_type`).
- Whitespace-only values are treated as missing.
- The gate does not mutate input `SearchResult` objects or their metadata.
