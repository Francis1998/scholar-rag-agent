# Changelog

All notable changes to **scholar-rag-agent** are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- DBLP ingestion connector (`ingestion/dblp.py`) that queries the DBLP public `search/publ` endpoint by keyword and normalizes each publication (title, authors, venue, year, DOI, electronic-edition link) into a `Document`. It adds a ninth scholarly source and the first dedicated computer-science bibliography alongside PDF, arXiv, Semantic Scholar, OpenAlex, PubMed, Crossref, Europe PMC, and DOAJ. As DBLP is a metadata index without abstracts, a concise author/venue/year descriptor is synthesized as the document text so sparse and entity retrieval retain real signal.
- DOAJ ingestion connector (`ingestion/doaj.py`) that queries the DOAJ public `search/articles` endpoint by keyword and normalizes each `bibjson` article (title, abstract, DOI, year, full-text link) into a `Document`. It adds an eighth scholarly source that guarantees a freely readable open access full text alongside PDF, arXiv, Semantic Scholar, OpenAlex, PubMed, Crossref, and Europe PMC.

### Fixed
- Dense hashing embeddings tokenized on raw whitespace while the BM25 sparse retriever strips surrounding punctuation, so a term such as `retrieval.` embedded into a different dimension than `retrieval` and the same word was a hit in sparse retrieval but a miss in the dense vector the two are fused with. The embedder now uses the shared `retrieval.sparse.tokenize` helper, making dense and sparse retrieval bucket a term identically.
- Crossref abstracts left XML/HTML entities undecoded: `_strip_jats` removed JATS tags but not entity references, so `&lt;`, `&amp;`, and numeric character references such as `&#945;` (α) leaked into the stored abstract as raw markup. Entities are now decoded via `html.unescape`, yielding readable, searchable prose.

### Added (earlier this cycle)
- Europe PMC ingestion connector (`ingestion/europepmc.py`) that queries the Europe PMC REST `search` endpoint (`resultType=core`) by keyword and normalizes each result (title, abstract, DOI, year, PMID) into a `Document`. It adds a seventh scholarly source that federates PubMed/MEDLINE, PubMed Central, preprints, and patents alongside PDF, arXiv, Semantic Scholar, OpenAlex, PubMed, and Crossref.
- Crossref ingestion connector (`ingestion/crossref.py`) that queries the Crossref REST `works` endpoint by keyword and normalizes each work (title, JATS-stripped abstract, DOI, year) into a `Document`. It adds the largest cross-disciplinary DOI index as a sixth scholarly source alongside PDF, arXiv, Semantic Scholar, OpenAlex, and PubMed.
- PubMed ingestion connector (`ingestion/pubmed.py`) that runs the NCBI E-utilities `esearch`+`efetch` flow to resolve a free-text query to PMIDs and normalize each article (title, structured multi-section abstract, PMID, year) into a `Document`. It is the first keyword-search connector, so one call can ingest several biomedical papers for a topic.
- OpenAlex ingestion connector (`ingestion/openalex.py`) that fetches works from the OpenAlex API and reconstructs the abstract from its `abstract_inverted_index` representation, adding a fourth open scholarly source alongside PDF, arXiv, and Semantic Scholar.
- Maximal Marginal Relevance (MMR) diversity re-ranker (`retrieval/mmr.py`) that reduces near-duplicate chunks by balancing query relevance against novelty (dependency-free lexical Jaccard similarity); available as an optional, default-off `HybridRetriever` stage.
- Public repository scaffold with Python 3.11+ packaging, CI, Docker, docs, and tests.
- Observe -> Decide -> Act agent runtime with an explicit persisted state machine.
- SQLite event log for state transitions and decision traces.
- PDF, arXiv, and Semantic Scholar ingestion connectors.
- Dense, sparse, HyDE, RRF, GraphRAG, multi-hop, and reranking retrieval components.
- LLM adapter contracts and optional provider adapters for OpenAI, Anthropic, Gemini, and Kimi.
- Citation grounding and `[UNGROUNDED]` hallucination guard.
- FastAPI endpoints for health, text ingestion, query execution, and run events.
- Demo GIF gallery and deterministic local demo scripts.

### Changed
- README restored to project-specific use cases, demos, architecture, and setup instructions.
- CI matrix aligned with the Python 3.11+ package requirement.
- Utility scripts and docs corrected to avoid generated placeholder APIs.

### Fixed
- PubMed connector no longer truncates abstracts that contain inline formatting elements (for example `<i>` around gene names or `<sup>` for exponents). The previous `node.text` read captured only the run of text before the first inline child, silently dropping the remainder of the abstract; every `AbstractText` segment's full text is now reconstructed with `itertext`.
- arXiv connector now recognizes versioned new-style identifiers (for example `2301.00001v2`) as ids and resolves them via the `id_list` parameter. The previous `replace('.', '').isdigit()` check failed on the trailing `vN` suffix, so versioned ids were misrouted to a keyword `search_query` and returned search hits instead of the requested paper.
- Gemini provider adapter no longer raises `AttributeError` when the `candidates` list holds a non-object first element (for example a blocked/`null` candidate or a malformed gateway payload); it now guards `candidates[0]` like the OpenAI adapter and degrades to an empty completion instead of failing the request.
- OpenAI (and inherited Kimi) provider adapter now extracts text from a structured `message.content` part list returned by OpenAI-compatible gateways (LiteLLM, vLLM, OpenRouter) instead of coercing the list with `str(...)`, which previously emitted a Python repr (`[{'type': 'text', ...}]`) as the answer.
- Anthropic provider adapter now concatenates all `content` text blocks (skipping non-text blocks such as `thinking`/`tool_use`) instead of reading only `content[0].text`, which raised `KeyError` when a non-text block came first and truncated multi-block answers.
- `SafetyLimits.clamp_hops`/`clamp_sources` and `MultiHopRetriever` no longer apply hardcoded literal ceilings (`5`/`50`) alongside the configurable `max_hops`/`max_source_docs`/`max_depth`, which silently capped any limit raised above the historical defaults; the configured values are now the sole authoritative bounds.
- Gemini provider adapter now concatenates all `content.parts` text segments (skipping non-text parts) instead of reading only the first, preventing silent truncation of multi-part answers and dropped citations.
- Removed fabricated release notes and ADRs generated by automation.
- Replaced undeclared `requests` health check dependency with existing `httpx`.
- Removed Redis/distributed-mode references from docs and scripts.
- Repaired broken `scholar-rag-ingest` and `scholar-rag-eval` console scripts by packaging `scripts/`.
- Aligned CI and security scans with full `src/` mypy and bandit coverage.
