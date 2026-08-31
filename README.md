# Scholar RAG Agent

[![CI](https://github.com/Francis1998/scholar-rag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Francis1998/scholar-rag-agent/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-pytest--cov-blue)](tests)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Scholar RAG Agent is a production-grade, local-first Agentic RAG system for scientific literature. It ingests papers from PDFs, arXiv, arXiv HTML abs abstracts, Semantic Scholar search and recommendations, OpenAlex, OpenAlex authors, OpenAlex author works, OpenAlex concepts, OpenAlex concepts ancestors, OpenAlex institutions, OpenAlex sources/venues, OpenAlex source hierarchies, OpenAlex sources host org, PubMed MeSH, OpenAlex topics hierarchy, Semantic Scholar bulk, OpenAlex publishers, OpenAlex funders, OpenAlex keywords, OpenAlex topics, PubMed, PubMed Central (PMC), PMC OA packages, Crossref, Crossref types filter, Crossref members, Crossref relations, Crossref Funder Registry, Crossref works-by-funder, Crossref works-by-license, Crossref works type+license, Crossref works ISSN+type, Crossref works-by-ISBN, Crossref Event Data, Crossref journals, Europe PMC, Europe PMC grants, DOAJ, DBLP, HAL, OpenAIRE, OpenAIRE projects, Zenodo, Figshare, CORE, bioRxiv/medRxiv, bioRxiv/medRxiv collections, NASA ADS, DataCite, DataCite related identifiers, DataCite reports, DataCite DOIs-by-prefix, DataCite Event Data, OpenCitations, OSF, ORCID, ORCID works filter, ORCID works summaries, ORCID education, Unpaywall, Dryad, Wikidata scholarly entities, SSRN preprints, OpenAlex retraction alerts, and ClinicalTrials.gov; builds hybrid dense, sparse, and entity-relationship retrieval indexes; and answers research questions with multi-hop reasoning and citation-backed evidence.

Scholar RAG Agent supports reproducible scientific knowledge synthesis, helping researchers accelerate literature review, hypothesis validation, and grounded comparison across large corpora while preserving source provenance.

## Why Researchers Need This

Most literature workflows break down when the corpus grows beyond a few papers:

- Issue: keyword search misses papers that use different terminology.
- : Inspired by bibliometric priors in scholarly RAG (Haystack-style metadata boosts); softly prefers mid-sized author lists over single-author or extreme mega-author rows (not a DOI connector).
  Scholar RAG Agent combines dense semantic retrieval, BM25 sparse search, HyDE expansion, and RRF fusion so a query can match both exact terms and related scientific phrasing.

- Issue: one hypothetical answer can overfit retrieval to a single framing.
  Multi-HyDE generates deterministic background, methods, findings, and
  limitations abstracts, retrieves each expansion, and fuses shared hits with
  RRF. Optional generation supports GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, and
  Kimi K2.

- Issue: compound questions bury multiple retrieval intents in one string.
  A deterministic query decomposer splits on conjunctions and question marks,
  deduplicates sub-queries, and keeps the original question first for fusion.

- Issue: domain synonyms fragment lexical retrieval across different terms.
  A deterministic query rewriter drops stopwords, expands a provided synonym
  map, and emits bounded query variants for reciprocal-rank fusion.

- Issue: fused results are dominated by near-duplicate passages that waste the context window.
  An optional Maximal Marginal Relevance (MMR) re-ranker balances relevance against novelty, dropping redundant chunks so the model sees complementary evidence.

- Issue: near-duplicate passages from overlapping sections still waste context after fusion.
  A deterministic near-duplicate collapser drops textually similar chunks above a Jaccard threshold, keeping the highest-scoring representative of each cluster.

- Issue: dense or fused rankings can under-weight chunks that share exact query terms.
  A deterministic lexical-overlap booster blends prior relevance with Jaccard
  query-chunk term overlap and re-sorts stably by the blended score.

- Issue: fused rankings can under-weight papers whose titles match the query even when body overlap is weak.
  A deterministic title-match booster blends prior relevance with Jaccard overlap against `chunk.title` only (distinct from title+text lexical overlap) and re-sorts stably by the blended score.

- Issue: relevance-only rankings can bury recent findings in fast-moving fields.
  A deterministic freshness booster blends normalized relevance with exponential
  publication-date decay from chunk metadata.

- Issue: fused rankings can under-weight recent papers when only a publication year is available.
  A deterministic recency half-life booster blends prior relevance with ``0.5 ** ((ref_year - year) / half_life)`` decay from year metadata and re-sorts stably.

- Issue: relevant chunks still contain sentences unrelated to the current query.
  A deterministic contextual compressor extracts bounded lexical-overlap spans,
  reducing token use without an LLM or network call.

- Issue: fixed-size character windows split mid-sentence and separate headings from their content.
  A deterministic agentic chunk-boundary splitter prefers markdown headings,
  then paragraph breaks, then sentence and word boundaries, falling back to
  raw characters only when a single token exceeds `max_chars`, then merges
  any resulting chunk under `min_chars` into a neighbor when it still fits.

- Issue: precise child chunks omit neighboring sentences that clarify methods or results.
  A deterministic sentence-window expander widens each hit by ±N sentences from
  `document_text` / `full_text` metadata without swapping in the entire parent.

- Issue: small chunks retrieve precisely but can omit the surrounding evidence needed for synthesis.
  A deterministic parent-document expander replaces child hits with deduplicated
  full parent text from a provided in-memory store.

- Issue: greetings and meta capability questions waste retrieval budget.
  A deterministic adaptive retrieval gate chooses RETRIEVE or SKIP from
  lexical chitchat versus knowledge-seeking cues before any corpus lookup.

- Issue: retrieval can surface chunks that still cannot support a grounded answer.
  A deterministic answerability gate scores lexical query coverage per chunk,
  refuses the whole batch when mean coverage is too low, and otherwise drops
  weak hits before synthesis.

- Issue: relevance-only rankings can under-weight highly cited papers that anchor a field.
  A deterministic citation-count booster blends prior relevance with batch-normalized `log1p(citation_count)` / `cited_by_count` from chunk metadata and re-sorts stably by the blended score.

- Issue: fused rankings can over-promote preprint servers relative to peer-reviewed venues.
  A deterministic preprint demoter soft-blends prior relevance with a demote score
  (`0.2` for arXiv/bioRxiv/medRxiv/SSRN/preprint metadata, else `1.0`) and re-sorts stably.

- Issue: multilingual corpora can bury preferred-language evidence under higher-scoring foreign-language hits.
  A deterministic language preferencer boosts or soft-filters results whose `language`/`lang` metadata matches a preferred set (default `en`).

- Issue: fused rankings treat introduction and results chunks equally even when the query needs methods or findings.
  A deterministic section-type booster blends prior relevance with preferred `section`/`section_type` scores (default results/methods/conclusion/abstract) and re-sorts stably.

- Issue: dense rankings can bury chunks that state explicit findings and conclusions.
  A deterministic claim-density booster blends prior relevance with the fraction of claim-like sentences in chunk text (reporting verbs or ``we`` / ``our results`` heuristics) and re-sorts stably.

- Issue: relevance-only rankings ignore soft authority cues such as peer review and impact.
  A deterministic authority booster blends prior relevance with `source_authority` / `venue_rank` / `is_peer_reviewed` / `impact_factor` metadata (neutral when missing).


- Issue: fused rankings can promote internally disjoint passages that jump topics mid-chunk.
  A deterministic coherence booster blends prior relevance with adjacent-sentence token
  overlap and query-term continuity across sentences, then re-sorts stably.


- Issue: synthesis stages may require provenance fields that some hits lack.
  A deterministic required-metadata gate drops chunks missing any configured non-empty metadata keys (empty key list is a pass-through).

- Issue: retrieval can return evidence too weak to support grounded synthesis.
  A deterministic corrective-RAG gate grades lexical query coverage, filters
  weak hits, and signals when a retry with rewritten terminology is needed.

- Issue: single-hop RAG retrieves isolated snippets but misses evidence chains.
  The GraphRAG layer extracts entities and relationships, then follows bounded multi-hop paths to connect methods, datasets, findings, and limitations across papers.

- Issue: draft answers can mix grounded sentences with unsupported claims.
  A deterministic claim-verification gate splits the answer into claim
  sentences, scores lexical support against retrieved chunks, and reports
  per-claim groundedness before synthesis is trusted.

- Issue: generated summaries sound plausible but are hard to audit.
  Every answer is mapped back to retrieved chunk IDs, and unsupported claims are flagged with `[UNGROUNDED]` instead of being silently trusted.

- Issue: an answer can cite a source that does not actually support the sentence next to it.
  A deterministic citation-groundedness scorer resolves `[n]` and
  `(Author, Year)` markers to specific retrieved chunks and measures whether
  each cited sentence lexically overlaps the source it names.

- Issue: citation interfaces need exact source offsets, not approximate excerpts.
  A deterministic evidence-span aligner maps query terms to Unicode-aware
  half-open character spans in retrieved chunk text for reliable highlighting.

- Issue: rankings ignore whether a query needs background, methods, results, or comparisons.
  A deterministic citation-intent classifier labels the query and attaches that
  intent to result metadata for downstream citation-aware ranking.

- Issue: research questions often need a plan, not just one search call.
  The Observe -> Decide -> Act runtime classifies intent, decomposes the query into retrieval sub-tasks, and persists a JSON rationale trace for every decision.

- Issue: teams need reproducible evidence trails for reviews, grants, and publications.
  The SQLite event log records state transitions, timestamps, agent IDs, run IDs, plans, retrieval payloads, and final answer provenance.

## Example Use Cases

- Systematic literature review: ingest a folder of PDFs plus arXiv IDs, ask for the strongest themes, and receive cited claims grouped by supporting chunks.

- Research and grant evidence synthesis: collect papers around a research question or contribution, assess novelty claims, and export citation-backed reasoning traces showing the evidence for each claim.

- Hypothesis validation: ask whether the literature supports or refutes a hypothesis, then inspect supporting and counter-evidence retrieval tasks separately.

- Method comparison: compare approaches such as GraphRAG, dense retrieval, and BM25 across papers while preserving the source chunks behind each contrast.

- Research onboarding: give a new lab member a paper corpus and let them ask grounded factual, synthesis, comparison, and hypothesis questions without manually reading every PDF first.

- Prior-art triage: search Semantic Scholar and arXiv records, expand a trusted seed through Semantic Scholar recommendations, ingest abstracts, then identify overlapping methods, datasets, and claims before deeper manual review.

- Citation QA for drafts: paste draft claims as questions and flag statements that are not supported by the ingested source chunks.

- Multi-provider LLM evaluation: route reasoning, speed, cost, and default tasks to different adapters while keeping output validation and citation grounding consistent.

## Demo Gallery

![End-to-end local demo](docs/assets/demo.gif)

![Use case walkthrough](docs/assets/use_cases.gif)

![Planning trace demo](docs/assets/planning_trace.gif)

![Grounded answer demo](docs/assets/grounded_answer.gif)

```text
                 +---------------------------+
                 | Observe: Query Analyzer   |
                 +-------------+-------------+
                               |
                               v
+---------+      +-------------+-------------+      +-------------------+
| Papers  +----->| Decide: Planner           +----->| Act: Executor     |
+---------+      +-------------+-------------+      +---------+---------+
 PDF/arXiv/S2                  |                              |
                               v                              v
                   +-----------+-----------+       +----------+----------+
                   | SQLite Durable Events |       | Hybrid Retrieval    |
                   +-----------------------+       | Dense + BM25 + RRF  |
                                                   +----------+----------+
                                                              |
                                                              v
                                                   +----------+----------+
                                                   | GraphRAG Multi-hop  |
                                                   +----------+----------+
                                                              |
                                                              v
                                                   +----------+----------+
                                                   | LLM Router + Guard  |
                                                   +----------+----------+
                                                              |
                                                              v
                                                   Citation-backed answer
```

## Install In 3 Commands

```bash
git clone https://github.com/Francis1998/scholar-rag-agent.git
cd scholar-rag-agent && uv sync --extra dev
uv run pytest tests/ -v
```

## Local Demo

```bash
uv run python scripts/demo_local.py
uv run uvicorn api.main:app --reload
```

The deterministic demo ingests a small fixture paper, executes an Observe -> Decide -> Act run, prints the planner trace, and returns a cited answer. A generated demo asset is available at `docs/assets/demo.gif`.

Additional GIFs in `docs/assets/` show the problem-to-solution flow, planner trace, and citation grounding guard.

## Documentation

| Document | Description |
| --- | --- |
| [Quickstart](QUICKSTART.md) | Install, demo, and API in three steps. |
| [Architecture](ARCHITECTURE.md) | Agent state machine, retrieval pipeline, and data flow. |
| [Configuration](CONFIGURATION.md) | Environment variables and provider keys. |
| [Configuration (extended)](docs/CONFIGURATION.md) | Full configuration reference with examples. |
| [Safety](SAFETY.md) | Timeout policy, scope bounds, cancellation, and hallucination guard design. |
| [Demo](docs/DEMO.md) | Demo GIFs and reproducible local demo commands. |
| [Multi-HyDE fusion guide](docs/guides/MULTI_HYDE_FUSION_GUIDE.md) | Retrieve deterministic hypothetical abstracts and fuse their rankings with RRF. |
| [Near duplicate collapse guide](docs/guides/NEAR_DUPLICATE_COLLAPSE_GUIDE.md) | Collapse near-duplicate chunks by text Jaccard similarity, keeping top scorers. |
| [Citation count boost guide](docs/guides/CITATION_COUNT_BOOST_GUIDE.md) | Re-rank results by blending relevance with normalized log1p citation counts. |
| [Required metadata gate guide](docs/guides/REQUIRED_METADATA_GATE_GUIDE.md) | Drop results missing any required non-empty chunk metadata keys. |
| [Title match boost guide](docs/guides/TITLE_MATCH_BOOST_GUIDE.md) | Re-rank results by blending relevance with Jaccard query-title term overlap. |
| [Lexical overlap boost guide](docs/guides/LEXICAL_OVERLAP_BOOST_GUIDE.md) | Re-rank results by blending relevance with Jaccard query-chunk term overlap. |
| [Freshness boost guide](docs/guides/FRESHNESS_BOOST_GUIDE.md) | Re-rank results with configurable exponential publication-date decay. |
| [Recency half life boost guide](docs/guides/RECENCY_HALF_LIFE_GUIDE.md) | Re-rank results by blending relevance with publication-year half-life decay. |
| [Contextual compression guide](docs/guides/CONTEXTUAL_COMPRESSION_GUIDE.md) | Extract bounded query-relevant sentence spans from retrieved chunks. |
| [Sentence window expand guide](docs/guides/SENTENCE_WINDOW_EXPAND_GUIDE.md) | Expand retrieved chunks with ±N neighboring sentences from full document text. |
| [Parent document guide](docs/guides/PARENT_DOCUMENT_GUIDE.md) | Expand child chunk hits to deduplicated full parent documents. |
| [Claim verification gate guide](docs/guides/CLAIM_VERIFICATION_GATE_GUIDE.md) | Split draft answers into claims and score lexical groundedness against retrieved chunks. |
| [Citation groundedness score guide](docs/guides/CITATION_GROUNDEDNESS_SCORE_GUIDE.md) | Resolve `[n]` / `(Author, Year)` citation markers and score lexical alignment to the cited source. |
| [Answerability gate guide](docs/guides/ANSWERABILITY_GATE_GUIDE.md) | Score lexical query coverage and refuse batches that cannot support an answer. |
| [Temporal freshness cutoff guide](docs/guides/TEMPORAL_FRESHNESS_CUTOFF_GUIDE.md) | Drop chunks older than a configured maximum age before synthesis. |
| [Agentic chunk boundary guide](docs/guides/AGENTIC_CHUNK_BOUNDARY_GUIDE.md) | Split long text on headings, paragraphs, and sentences before falling back to fixed-size cuts. |
| [Adaptive retrieval gate guide](docs/guides/ADAPTIVE_RETRIEVAL_GATE_GUIDE.md) | Decide RETRIEVE vs SKIP before lookup using chitchat and knowledge-seeking cues. |
| [Corrective RAG gate guide](docs/guides/CORRECTIVE_RAG_GUIDE.md) | Grade lexical relevance and signal keep, filter, or retry with query rewriting. |
| [Query decomposition guide](docs/guides/QUERY_DECOMPOSITION_GUIDE.md) | Split compound questions into distinct retrieval sub-queries for multi-query fusion. |
| [Query rewrite guide](docs/guides/QUERY_REWRITE_GUIDE.md) | Expand provided synonyms and generate deterministic multi-query retrieval variants. |
| [Citation intent guide](docs/guides/CITATION_INTENT_GUIDE.md) | Label background, method, result, comparison, or unknown evidence needs. |
| [arXiv HTML abstract source guide](docs/guides/ARXIV_HTML_ABSTRACT_SOURCE_GUIDE.md) | arXiv abs HTML abstract enrichment connector. |
| [Examples](docs/EXAMPLES.md) | Usage examples for ingestion, querying, and retrieval evaluation. |
| [Performance](docs/PERFORMANCE.md) | Performance tuning notes. |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common setup and runtime fixes. |
| [Contributing](CONTRIBUTING.md) | Development and PR workflow. |
| [Security](SECURITY.md) | Vulnerability reporting policy. |
| [Changelog](CHANGELOG.md) | Version history. |
| [bioRxiv / medRxiv source guide](docs/guides/BIORXIV_SOURCE_GUIDE.md) | bioRxiv and medRxiv preprint connector. |
| [bioRxiv / medRxiv collections guide](docs/guides/BIORXIV_COLLECTIONS_SOURCE_GUIDE.md) | bioRxiv and medRxiv subject-category collection connector. |
| [NASA ADS source guide](docs/guides/ADS_SOURCE_GUIDE.md) | NASA ADS astronomy/physics connector. |
| [PMC source guide](docs/guides/PMC_SOURCE_GUIDE.md) | PubMed Central full-text connector. |
| [PMC OA package guide](docs/guides/PMC_OA_PACKAGE_GUIDE.md) | NCBI PMC Open Access package/PDF link discovery connector. |
| [DataCite source guide](docs/guides/DATACITE_SOURCE_GUIDE.md) | DataCite DOI registry connector. |
| [DataCite related identifiers source guide](docs/guides/DATACITE_RELATED_SOURCE_GUIDE.md) | DataCite related-identifier enrichment connector. |
| [DataCite Event Data source guide](docs/guides/DATACITE_EVENTS_SOURCE_GUIDE.md) | DataCite DOI citation, usage, and relationship events connector. |
| [OpenCitations source guide](docs/guides/OPENCITATIONS_SOURCE_GUIDE.md) | OpenCitations DOI metadata and citation-count connector. |
| [Semantic Scholar recommendations guide](docs/guides/SEMANTIC_SCHOLAR_RECOMMENDATIONS_GUIDE.md) | Related-paper expansion from a seed Semantic Scholar id or DOI. |
| [OSF source guide](docs/guides/OSF_SOURCE_GUIDE.md) | Open Science Framework preprint and registration connector. |
| [OpenAIRE projects source guide](docs/guides/OPENAIRE_PROJECTS_SOURCE_GUIDE.md) | OpenAIRE funded-projects registry connector. |
| [ORCID works filter source guide](docs/guides/ORCID_WORKS_FILTER_SOURCE_GUIDE.md) | ORCID works year/type deep-filter connector. |
| [ORCID works summaries source guide](docs/guides/ORCID_WORKS_SUMMARIES_SOURCE_GUIDE.md) | ORCID iD public work-summaries connector. |
| [ORCID source guide](docs/guides/ORCID_SOURCE_GUIDE.md) | ORCID public record works connector. |
| [Unpaywall source guide](docs/guides/UNPAYWALL_SOURCE_GUIDE.md) | Unpaywall DOI open-access landing/PDF lookup connector. |
| [OpenAlex topics source guide](docs/guides/OPENALEX_TOPICS_SOURCE_GUIDE.md) | OpenAlex research-topic taxonomy connector. |
| [OpenAlex concepts source guide](docs/guides/OPENALEX_CONCEPTS_SOURCE_GUIDE.md) | OpenAlex legacy concepts taxonomy connector. |
| [OpenAlex concepts ancestors source guide](docs/guides/OPENALEX_CONCEPTS_ANCESTORS_SOURCE_GUIDE.md) | OpenAlex concepts with ancestors hierarchy connector. |
| [OpenAlex institutions source guide](docs/guides/OPENALEX_INSTITUTIONS_SOURCE_GUIDE.md) | OpenAlex research-institution connector. |
| [OpenAlex sources source guide](docs/guides/OPENALEX_SOURCES_SOURCE_GUIDE.md) | OpenAlex journal/venue sources connector. |
| [OpenAlex sources hierarchy source guide](docs/guides/OPENALEX_SOURCES_HIERARCHY_SOURCE_GUIDE.md) | OpenAlex venues with host, type, and ISSN ancestry paths. |
| [OpenAlex sources host org source guide](docs/guides/OPENALEX_SOURCES_HOST_ORG_SOURCE_GUIDE.md) | OpenAlex venues filtered by host organization. |
| [PubMed MeSH source guide](docs/guides/PUBMED_MESH_SOURCE_GUIDE.md) | NCBI MeSH vocabulary descriptor connector. |
| [OpenAlex topics hierarchy source guide](docs/guides/OPENALEX_TOPICS_HIERARCHY_SOURCE_GUIDE.md) | OpenAlex topics with domain/field/subfield ancestry. |
| [Semantic Scholar bulk source guide](docs/guides/SEMANTIC_SCHOLAR_BULK_SOURCE_GUIDE.md) | Semantic Scholar paper/batch bulk connector. |
| [OpenAlex publishers source guide](docs/guides/OPENALEX_PUBLISHERS_SOURCE_GUIDE.md) | OpenAlex publisher-organization connector. |
| [OpenAlex funders source guide](docs/guides/OPENALEX_FUNDERS_SOURCE_GUIDE.md) | OpenAlex funding-organization connector. |
| [OpenAlex keywords source guide](docs/guides/OPENALEX_KEYWORDS_SOURCE_GUIDE.md) | OpenAlex research-keyword taxonomy connector. |
| [Europe PMC grants source guide](docs/guides/EUROPEPMC_GRANTS_SOURCE_GUIDE.md) | Europe PMC GRIST grants connector. |
| [Crossref relations source guide](docs/guides/CROSSREF_RELATIONS_SOURCE_GUIDE.md) | Crossref works relation-types connector. |
| [OpenAlex authors source guide](docs/guides/OPENALEX_AUTHORS_SOURCE_GUIDE.md) | OpenAlex researcher-profile connector. |
| [Retraction check guide](docs/guides/RETRACTION_CHECK_GUIDE.md) | OpenAlex retracted-works alert connector. |
| [Crossref Event Data source guide](docs/guides/CROSSREF_EVENTS_SOURCE_GUIDE.md) | Crossref Event Data altmetrics/events connector. |
| [Crossref journals source guide](docs/guides/CROSSREF_JOURNALS_SOURCE_GUIDE.md) | Crossref journal metadata / ISSN connector. |
| [Crossref members source guide](docs/guides/CROSSREF_MEMBERS_SOURCE_GUIDE.md) | Crossref publisher/registrant member connector. |
| [Crossref Funder Registry source guide](docs/guides/CROSSREF_FUNDER_SOURCE_GUIDE.md) | Crossref Open Funder Registry connector. |
| [CORE source guide](docs/guides/CORE_SOURCE_GUIDE.md) | CORE open-access works connector. |
| [Figshare source guide](docs/guides/FIGSHARE_SOURCE_GUIDE.md) | Figshare research-output connector. |
| [Dryad source guide](docs/guides/DRYAD_SOURCE_GUIDE.md) | Dryad research-data repository connector. |
| [ClinicalTrials.gov source guide](docs/guides/CLINICALTRIALS_SOURCE_GUIDE.md) | ClinicalTrials.gov clinical-study registry connector. |
| [Wikidata scholarly source guide](docs/guides/WIKIDATA_SCHOLARLY_SOURCE_GUIDE.md) | Wikidata scholarly-entity search connector. |
| [SSRN source guide](docs/guides/SSRN_SOURCE_GUIDE.md) | SSRN preprint DOI bridge via Crossref connector. |
| [ORCID employments source guide](docs/guides/ORCID_EMPLOYMENTS_SOURCE_GUIDE.md) | ORCID public employment affiliations connector. |
| [ORCID education source guide](docs/guides/ORCID_EDUCATION_SOURCE_GUIDE.md) | ORCID public education affiliations connector. |
| [OpenAlex works n-grams source guide](docs/guides/OPENALEX_WORKS_NGRAMS_SOURCE_GUIDE.md) | OpenAlex salient phrase and frequency connector for individual works. |
| [DataCite client and prefix source guide](docs/guides/DATACITE_CLIENT_PREFIX_SOURCE_GUIDE.md) | DataCite client-id scoped DOI listing with prefix compatibility. |
| [Crossref works-by-funder source guide](docs/guides/CROSSREF_WORKS_FUNDER_SOURCE_GUIDE.md) | Crossref funded-works / funder-filter connector. |
| [Crossref works-by-license source guide](docs/guides/CROSSREF_WORKS_LICENSE_SOURCE_GUIDE.md) | Crossref licensed-works / license-URL filter connector. |
| [Crossref works type+license source guide](docs/guides/CROSSREF_WORKS_TYPE_LICENSE_SOURCE_GUIDE.md) | Crossref type+license filter connector. |
| [Crossref works ISSN+type source guide](docs/guides/CROSSREF_WORKS_ISSN_TYPE_SOURCE_GUIDE.md) | Crossref ISSN+type filter connector. |
| [Crossref works ISBN source guide](docs/guides/CROSSREF_WORKS_ISBN_SOURCE_GUIDE.md) | Crossref ISBN filter connector for books and other ISBN-bearing works. |
| [OpenAlex author works source guide](docs/guides/OPENALEX_AUTHOR_WORKS_SOURCE_GUIDE.md) | OpenAlex author→works citations blend connector. |
| [DataCite reports source guide](docs/guides/DATACITE_REPORTS_SOURCE_GUIDE.md) | DataCite research-report DOI connector. |
| [DataCite DOIs-by-prefix source guide](docs/guides/DATACITE_DOIS_PREFIX_SOURCE_GUIDE.md) | DataCite DOI prefix filter connector. |
| [Europe PMC preprints source guide](docs/guides/EUROPEPMC_PREPRINTS_SOURCE_GUIDE.md) | Europe PMC PPR-filtered preprint connector. |
| [Open access prefer guide](docs/guides/OPEN_ACCESS_PREFER_GUIDE.md) | Prefer open-access hits via score boost or soft filter when any OA exists. |
| [Venue tier boost guide](docs/guides/VENUE_TIER_BOOST_GUIDE.md) | Re-rank results by blending relevance with venue prestige tier scores. |
| [Preprint demote guide](docs/guides/PREPRINT_DEMOTE_GUIDE.md) | Soft-demote preprint venues via blended demote scores from publication_type/type/venue metadata. |
| [Language prefer guide](docs/guides/LANGUAGE_PREFER_GUIDE.md) | Prefer preferred-language hits via score boost or soft filter when any match exists. |
| [Section type boost guide](docs/guides/SECTION_TYPE_BOOST_GUIDE.md) | Re-rank results by blending relevance with preferred section/section_type scores. |
| [Claim density boost guide](docs/guides/CLAIM_DENSITY_BOOST_GUIDE.md) | Re-rank results by blending relevance with claim-like sentence density in chunk text. |
| [Authority boost guide](docs/guides/AUTHORITY_BOOST_GUIDE.md) | Re-rank results by blending relevance with source_authority / venue_rank / peer-review / impact_factor signals. |
| [Coherence boost guide](docs/guides/COHERENCE_BOOST_GUIDE.md) | Re-rank results by blending relevance with adjacent-sentence overlap and query-term continuity. |

## Provider Keys

All live providers are optional. Without keys the system uses deterministic fakes for tests and demos. Configure keys in `.env` or your shell:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export MOONSHOT_API_KEY=...
export UNPAYWALL_EMAIL=dev@example.org
```

When enabled, downstream synthesis can route through GPT-5.5, Claude Sonnet 4.6,
Gemini 3.x, and Kimi K2 while deterministic connectors such as Unpaywall keep
source lookup reproducible.
For downstream synthesis and evaluation, the preferred frontier model families are GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, and Kimi K2.
Gemini 3.x, and Kimi K2 while deterministic connectors such as Unpaywall and
retraction checks keep source lookup reproducible.

## Quality Gates

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src/
uv run pytest tests/ -v --cov=src --cov-fail-under=70
```

## License

Apache-2.0. See `LICENSE`.
