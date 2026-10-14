# Duplicate Paper Cluster Guide

![Duplicate paper cluster demo](../assets/duplicate-paper-cluster.gif)

`DuplicatePaperClusterer` creates inspectable **paper-level candidate duplicate
groups** using normalized DOI matches, literal `document_id` matches, and fuzzy
title similarity via `difflib.SequenceMatcher`. These are local heuristics, not
proof that references identify the same scientific work. Review candidates before
making identity or deduplication decisions.

Distinct from `NearDuplicateCollapser` and `ParaphraseCollapser`, which collapse
chunk-level retrieval hits. Clustering does not call a model or DOI service.
For optional downstream synthesis configuration, see the central
[provider models and compatibility guide](PROVIDER_MODELS_GUIDE.md).

## Matching and identity

- Nonblank document IDs match only the same raw string. Case and surrounding
  whitespace are significant: `"Study-A"`, `"study-a"`, and `" Study-A "` are
  distinct IDs. Repeated identical IDs, including identical padding, still match.
- Empty and whitespace-only IDs never create a document-ID match. A reference
  with a blank ID can still participate through its DOI or title. References with
  only blank IDs/titles and no DOI are omitted, as before.
- DOI normalization and fuzzy title matching remain independent heuristics.
  Distinct document IDs can still belong to one candidate group through these
  signals. Titles are lowercased and whitespace-normalized before comparison;
  `title_threshold` is inclusive.
- Grouping is transitive: an ID match linking two references and a DOI or title
  match linking one of them to a third yields one candidate group. Not every
  pair in that group necessarily matches directly.

Inputs are not mutated or merged into a canonical paper. Member order follows
input order; cluster IDs derive from the earliest retained input index, and
clusters are sorted by descending size, then cluster ID.

## Usage

```python
from retrieval.duplicate_paper_cluster import DuplicatePaperClusterer, PaperRef

clusterer = DuplicatePaperClusterer(title_threshold=0.92)
papers = [
    PaperRef(document_id="a", title="Graph RAG Survey", doi="10.1000/xyz"),
    PaperRef(document_id="b", title="Graph RAG Survey", doi="https://doi.org/10.1000/XYZ"),
]
clusters = clusterer.duplicate_clusters(papers)
for cluster in clusters:
    print(cluster.cluster_id, cluster.size, cluster.match_reasons)
```

`Document`, `Chunk`, and `SearchResult` inputs are also accepted. Use
`cluster(...)` when you need the full partition including singletons.
