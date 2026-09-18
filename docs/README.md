- **NumberNeededToTreatHintExtractor**: offline evidence cue extractor — see `docs/guides/NNT_HINT_EXTRACTOR_GUIDE.md`

# Documentation catalog

Start with a working local workflow, then choose extensions deliberately.
The guides below remain at their existing URLs; this catalog replaces the
accumulated feature list in the project README.

## Start here

| Goal | Guide |
| --- | --- |
| Install and run without model credentials | [Quickstart](../QUICKSTART.md) |
| Work through a small corpus and prepare an honest portfolio demonstration | [Research workflow](guides/RESEARCH_WORKFLOW_GUIDE.md) |
| Save and review the exact evidence from a recorded run | [Evidence export](guides/EVIDENCE_EXPORT_GUIDE.md) |
| Copy public API requests or try connector examples | [Examples](EXAMPLES.md) |
| Reproduce the showcased animation | [Demo](DEMO.md) |
| Understand what the API actually runs | [Architecture](../ARCHITECTURE.md) |
| Choose provider models and understand routing | [Provider models](guides/PROVIDER_MODELS_GUIDE.md) |

## Configuration, operation, and contribution

| Document | Scope |
| --- | --- |
| [Configuration](../CONFIGURATION.md) | Environment variables and defaults |
| [Extended configuration](CONFIGURATION.md) | Configuration examples |
| [Safety](../SAFETY.md) | Execution bounds, grounding limitations, and data handling |
| [Troubleshooting](TROUBLESHOOTING.md) | Setup and runtime troubleshooting |
| [Performance](PERFORMANCE.md) | Tuning notes, not measured product benchmarks |
| [Contributing](../CONTRIBUTING.md) | Development, quality gates, and PR workflow |
| [Security](../SECURITY.md) | Vulnerability reporting |
| [Migrations](../migrations/README.md) | Storage migration notes |
| [Changelog](../CHANGELOG.md) | Version history |
| [License](../LICENSE) | Apache-2.0 terms for this repository |

## How to read the extension guides

**The following are opt-in Python utilities and source adapters, not a checklist
of stages automatically executed by `/query`.** Import and wire them in your own
application as their usage examples show. Installing optional dependencies alone
does not enable every helper, a learned embedding model, or cross-encoder reranking
in the API. See [the runtime wiring](../src/api/dependencies.py) and
[architecture](../ARCHITECTURE.md).

Most scoring, screening, extraction, and "verification" helpers use deterministic
lexical or metadata heuristics. Their labels are advisory, not proof of factual
support, study quality, causality, novelty, or completeness. Connector calls may
need network access, service credentials, and permission to use the returned text.

Older guides and their generated animations retain historical model names and
illustrative examples. Use the dated, source-linked
[provider model guide](guides/PROVIDER_MODELS_GUIDE.md) for current defaults and
routing; an old provider name in a helper guide is not a compatibility promise.

### Query planning and context

| Guide | Utility |
| --- | --- |
| [Adaptive retrieval gate](guides/ADAPTIVE_RETRIEVAL_GATE_GUIDE.md) | Lexical retrieve-or-skip decision |
| [Query decomposition](guides/QUERY_DECOMPOSITION_GUIDE.md) | Split compound queries |
| [Query rewrite](guides/QUERY_REWRITE_GUIDE.md) | Caller-supplied synonym expansion |
| [Multi-HyDE fusion](guides/MULTI_HYDE_FUSION_GUIDE.md) | Fuse multiple hypothetical query expansions |
| [Citation intent](guides/CITATION_INTENT_GUIDE.md) | Label background, method, result, or comparison intent |
| [Agentic chunk boundaries](guides/AGENTIC_CHUNK_BOUNDARY_GUIDE.md) | Heading, paragraph, and sentence-aware splitting |
| [Contextual compression](guides/CONTEXTUAL_COMPRESSION_GUIDE.md) | Extract query-relevant sentence spans |
| [Sentence windows](guides/SENTENCE_WINDOW_EXPAND_GUIDE.md) | Expand hits with neighboring sentences |
| [Parent documents](guides/PARENT_DOCUMENT_GUIDE.md) | Expand child hits to provided parent text |
| [PDF OCR hook](guides/PDF_OCR_HOOK_GUIDE.md) | Supply an OCR backend for short PDF extraction |
| [Figure captions](guides/FIGURE_CAPTION_GUIDE.md) | Extract figure and table caption text and offsets |

### Ranking and diversity

These signals alter retrieval order; a higher score is not stronger scientific
evidence. Keep a baseline and evaluate on labeled queries before adopting them.

| Guide | Signal or operation |
| --- | --- |
| [Lexical overlap](guides/LEXICAL_OVERLAP_BOOST_GUIDE.md) | Query/chunk token overlap |
| [Title match](guides/TITLE_MATCH_BOOST_GUIDE.md) | Query/title overlap |
| [Abstract overlap](guides/ABSTRACT_OVERLAP_BOOST_GUIDE.md) | Query/abstract overlap |
| [Abstract keyword boost](guides/ABSTRACT_KEYWORD_BOOST_GUIDE.md) | Query keywords in abstracts |
| [Entity overlap](guides/ENTITY_OVERLAP_BOOST_GUIDE.md) | Entity overlap |
| [Term coverage](guides/TERM_COVERAGE_BOOST_GUIDE.md) | Fraction of query terms present |
| [Claim density](guides/CLAIM_DENSITY_BOOST_GUIDE.md) | Claim-like sentence cues |
| [Section type](guides/SECTION_TYPE_BOOST_GUIDE.md) | Preferred section metadata |
| [Coherence](guides/COHERENCE_BOOST_GUIDE.md) | Adjacent-sentence overlap |
| [Citation count](guides/CITATION_COUNT_BOOST_GUIDE.md) | Citation-count metadata |
| [Authority boost](guides/AUTHORITY_BOOST_GUIDE.md) | Source-authority metadata |
| [Author count](guides/AUTHOR_COUNT_BOOST_GUIDE.md) | Author-list-size prior |
| [Venue tier](guides/VENUE_TIER_BOOST_GUIDE.md) | Caller-defined venue tiers |
| [Freshness](guides/FRESHNESS_BOOST_GUIDE.md) | Publication-date decay |
| [Recency half-life](guides/RECENCY_HALF_LIFE_GUIDE.md) | Publication-year decay |
| [Time decay gate](guides/TIME_DECAY_GATE_GUIDE.md) | Multiply scores by an age-dependent weight |
| [Open-access preference](guides/OPEN_ACCESS_PREFER_GUIDE.md) | Open-access metadata |
| [Language preference](guides/LANGUAGE_PREFER_GUIDE.md) | Preferred-language metadata |
| [Preprint demotion](guides/PREPRINT_DEMOTE_GUIDE.md) | Preprint metadata |
| [Code availability](guides/CODE_AVAILABILITY_BOOSTER_GUIDE.md) | Code-link cues |
| [Novelty diversity](guides/NOVELTY_DIVERSIFY_GUIDE.md) | Reduce textual redundancy, not establish research novelty |
| [Near-duplicate collapse](guides/NEAR_DUPLICATE_COLLAPSE_GUIDE.md) | Token-similarity deduplication |
| [Paraphrase collapse](guides/PARAPHRASE_COLLAPSE_GUIDE.md) | Character n-gram similarity deduplication |
| [Reciprocal rank fusion gate](guides/RECIPROCAL_RANK_FUSION_GATE_GUIDE.md) | Fuse supplied ranked lists |

### Gates and filters

| Guide | Decision input |
| --- | --- |
| [Answerability](guides/ANSWERABILITY_GATE_GUIDE.md) | Lexical query coverage |
| [Corrective RAG](guides/CORRECTIVE_RAG_GUIDE.md) | Keep, filter, or request a retry based on lexical relevance |
| [Self-RAG reflection](guides/SELF_RAG_REFLECTION_GATE_GUIDE.md) | Advisory reflection checks |
| [Score threshold](guides/SCORE_THRESHOLD_GATE_GUIDE.md) | Minimum supplied relevance score |
| [Keyword match](guides/KEYWORD_MATCH_GATE_GUIDE.md) | Query-keyword coverage |
| [Cross-encoder gate](guides/CROSS_ENCODER_GATE_GUIDE.md) | Local lexical proxy, not a learned cross-encoder |
| [Required metadata](guides/REQUIRED_METADATA_GATE_GUIDE.md) | Presence of required fields |
| [Metadata equality](guides/METADATA_EQUALS_GATE_GUIDE.md) | Exact key/value matches |
| [Minimum abstract length](guides/MIN_ABSTRACT_LENGTH_GATE_GUIDE.md) | Abstract length |
| [Minimum unique sources](guides/MIN_UNIQUE_SOURCES_GATE_GUIDE.md) | Distinct source count |
| [Diversity cap](guides/DIVERSITY_CAP_GATE_GUIDE.md) | Maximum hits per source |
| [Source authority](guides/SOURCE_AUTHORITY_GATE_GUIDE.md) | Source-authority metadata |
| [Peer-reviewed gate](guides/PEER_REVIEWED_GATE_GUIDE.md) | Peer-review metadata, not independent verification |
| [Retracted filter](guides/RETRACTED_FILTER_GUIDE.md) | Supplied retraction flags |
| [Year range](guides/YEAR_RANGE_GATE_GUIDE.md) | Publication-year bounds |
| [Temporal freshness cutoff](guides/TEMPORAL_FRESHNESS_CUTOFF_GUIDE.md) | Maximum publication age |

### Evidence inspection and corpus organization

| Guide | Utility |
| --- | --- |
| [Evidence span alignment](guides/EVIDENCE_SPAN_ALIGN_GUIDE.md) | Locate matching terms in chunk text |
| [Claim support scorer](guides/CLAIM_SUPPORT_SCORER_GUIDE.md) | Lexical claim/passage support labels |
| [Claim verification gate](guides/CLAIM_VERIFICATION_GATE_GUIDE.md) | Draft-claim lexical checks |
| [Citation groundedness](guides/CITATION_GROUNDEDNESS_SCORE_GUIDE.md) | Citation-marker resolution and lexical alignment |
| [Evidence conflict](guides/EVIDENCE_CONFLICT_GUIDE.md) | Polarity and negation cues |
| [Contradiction clusters](guides/CONTRADICTION_CLUSTER_FINDER_GUIDE.md) | Group apparent supporting and opposing passages |
| [Multi-hop claim tracing](guides/MULTIHOP_CLAIM_TRACER_GUIDE.md) | Claim/evidence paths |
| [Citation graph](guides/CITATION_GRAPH_GUIDE.md) | Citation-neighbor expansion |
| [Co-citation clusters](guides/CO_CITATION_CLUSTER_FINDER_GUIDE.md) | Shared citation-neighbor clusters |
| [Duplicate paper clusters](guides/DUPLICATE_PAPER_CLUSTER_GUIDE.md) | Paper-level duplicate hints |
| [Preprint version comparison](guides/PREPRINT_VERSION_DIFFER_GUIDE.md) | Compare supplied preprint and published text |
| [Author-name disambiguation](guides/AUTHOR_NAME_DISAMBIGUATION_HINT_GUIDE.md) | Surname and initial grouping hints |
| [Author expertise](guides/AUTHOR_EXPERTISE_GUIDE.md) | Metadata-based expertise proxies |
| [Dataset mentions](guides/DATASET_MENTION_INDEXER_GUIDE.md) | Curated dataset-name cues |
| [Method extraction cards](guides/METHOD_EXTRACT_CARD_GUIDE.md) | Study-design and methods cues |
| [Sample-size hints](guides/SAMPLE_SIZE_HINT_EXTRACTOR_GUIDE.md) | Sample-size integers in text |
| [Effect-size hints](guides/EFFECT_SIZE_HINT_EXTRACTOR_GUIDE.md) | Labeled effect-size values |
| [P-value hints](guides/P_VALUE_HINT_EXTRACTOR_GUIDE.md) | Significance-value patterns |
| [Confidence-interval hints](guides/CONFIDENCE_INTERVAL_HINT_EXTRACTOR_GUIDE.md) | Interval bounds in text |
| [Heterogeneity I2 hints](guides/HETEROGENEITY_I2_HINT_EXTRACTOR_GUIDE.md) | Meta-analysis I2 / heterogeneity cues |
| [Study limitations](guides/STUDY_LIMITATION_CUE_EXTRACTOR_GUIDE.md) | Limitation-language cues |
| [Funding disclosures](guides/FUNDING_DISCLOSURE_FLAGGER_GUIDE.md) | Funding and grant cues |
| [Conflict-of-interest flags](guides/CONFLICT_OF_INTEREST_FLAGGER_GUIDE.md) | Disclosure cues |
| [Preregistration flags](guides/PREREGISTRATION_FLAG_DETECTOR_GUIDE.md) | Registry and preregistration cues |
| [Open-data availability](guides/OPEN_DATA_AVAILABILITY_FLAGGER_GUIDE.md) | Data-availability cues |
| [Retraction watch flagger](guides/RETRACTION_FLAGGER_GUIDE.md) | Advisory flags from caller-supplied offline sets |
| [PRISMA screening checklist](guides/PRISMA_SCREENING_CHECKLIST_GUIDE.md) | Pending human-review rows, never automatic include/exclude |
| [Reading-list prioritization](guides/READING_LIST_PRIORITIZER_GUIDE.md) | Explainable unread-queue triage |
| [Paper chat memory](guides/PAPER_CHAT_MEMORY_GUIDE.md) | Library-managed paper-scoped chat turns |
| [Literature review outlines](guides/LITERATURE_REVIEW_OUTLINE_GUIDE.md) | Deterministic section scaffolding |
| [Related works](guides/RELATED_WORKS_GUIDE.md) | Theme-based outline scaffolding |
| [Survey gaps](guides/SURVEY_GAP_GUIDE.md) | Coverage of caller-provided themes, not proof of novelty |
| [BibTeX export](guides/BIBTEX_EXPORT_GUIDE.md) | Bibliography entries from supplied metadata |
| [Evaluation harness](guides/EVALUATION_HARNESS_GUIDE.md) | Labeled retrieval cases and lexical answer metrics |

### Source adapters: papers, repositories, and registries

Connectors return normalized records. They are not automatic web searches during
`/query`, and fetching a record is not the same as indexing it into the API's
corpus. See the [network examples](EXAMPLES.md#network-connectors) and
[ingestion modules](../src/ingestion) for the base PDF, arXiv, Semantic Scholar,
OpenAlex, PubMed, Crossref, Europe PMC, DOAJ, DBLP, HAL, OpenAIRE, and Zenodo adapters.

| Guide | Source |
| --- | --- |
| [arXiv HTML abstracts](guides/ARXIV_HTML_ABSTRACT_SOURCE_GUIDE.md) | arXiv abstract-page enrichment |
| [Semantic Scholar recommendations](guides/SEMANTIC_SCHOLAR_RECOMMENDATIONS_GUIDE.md) | Related papers from a seed |
| [Semantic Scholar bulk](guides/SEMANTIC_SCHOLAR_BULK_SOURCE_GUIDE.md) | Paper batch lookup |
| [bioRxiv and medRxiv](guides/BIORXIV_SOURCE_GUIDE.md) | Preprint records |
| [bioRxiv and medRxiv collections](guides/BIORXIV_COLLECTIONS_SOURCE_GUIDE.md) | Subject-category collections |
| [NASA ADS](guides/ADS_SOURCE_GUIDE.md) | Astronomy and physics records |
| [PMC](guides/PMC_SOURCE_GUIDE.md) | PubMed Central article text |
| [PMC OA packages](guides/PMC_OA_PACKAGE_GUIDE.md) | Open-access package and PDF link discovery |
| [PubMed MeSH](guides/PUBMED_MESH_SOURCE_GUIDE.md) | Vocabulary descriptors |
| [Europe PMC preprints](guides/EUROPEPMC_PREPRINTS_SOURCE_GUIDE.md) | Preprint-filtered results |
| [Europe PMC grants](guides/EUROPEPMC_GRANTS_SOURCE_GUIDE.md) | Grant records |
| [CORE](guides/CORE_SOURCE_GUIDE.md) | Open-access works |
| [Figshare](guides/FIGSHARE_SOURCE_GUIDE.md) | Research outputs |
| [Dryad](guides/DRYAD_SOURCE_GUIDE.md) | Research datasets |
| [OSF](guides/OSF_SOURCE_GUIDE.md) | Preprints and registrations |
| [OpenAIRE projects](guides/OPENAIRE_PROJECTS_SOURCE_GUIDE.md) | Funded projects |
| [ClinicalTrials.gov](guides/CLINICALTRIALS_SOURCE_GUIDE.md) | Study registry records |
| [Wikidata scholarly entities](guides/WIKIDATA_SCHOLARLY_SOURCE_GUIDE.md) | Scholarly entity search |
| [SSRN](guides/SSRN_SOURCE_GUIDE.md) | DOI bridge through Crossref |
| [Unpaywall](guides/UNPAYWALL_SOURCE_GUIDE.md) | DOI open-access location lookup |
| [OpenCitations](guides/OPENCITATIONS_SOURCE_GUIDE.md) | DOI citation metadata |

### Source adapters: OpenAlex

| Guide | Source |
| --- | --- |
| [Authors](guides/OPENALEX_AUTHORS_SOURCE_GUIDE.md) | Researcher profiles |
| [Author works](guides/OPENALEX_AUTHOR_WORKS_SOURCE_GUIDE.md) | Author-to-works lookup |
| [Institutions](guides/OPENALEX_INSTITUTIONS_SOURCE_GUIDE.md) | Institution records |
| [Publishers](guides/OPENALEX_PUBLISHERS_SOURCE_GUIDE.md) | Publisher organizations |
| [Funders](guides/OPENALEX_FUNDERS_SOURCE_GUIDE.md) | Funding organizations |
| [Sources](guides/OPENALEX_SOURCES_SOURCE_GUIDE.md) | Venues |
| [Source hierarchies](guides/OPENALEX_SOURCES_HIERARCHY_SOURCE_GUIDE.md) | Venue ancestry |
| [Sources by host organization](guides/OPENALEX_SOURCES_HOST_ORG_SOURCE_GUIDE.md) | Host-scoped venues |
| [Topics](guides/OPENALEX_TOPICS_SOURCE_GUIDE.md) | Research topics |
| [Topic hierarchies](guides/OPENALEX_TOPICS_HIERARCHY_SOURCE_GUIDE.md) | Domain, field, and subfield ancestry |
| [Concepts](guides/OPENALEX_CONCEPTS_SOURCE_GUIDE.md) | Legacy concept taxonomy |
| [Concept ancestors](guides/OPENALEX_CONCEPTS_ANCESTORS_SOURCE_GUIDE.md) | Legacy concept ancestry |
| [Keywords](guides/OPENALEX_KEYWORDS_SOURCE_GUIDE.md) | Keyword taxonomy |
| [Works n-grams](guides/OPENALEX_WORKS_NGRAMS_SOURCE_GUIDE.md) | Work-level phrases |
| [Retraction check](guides/RETRACTION_CHECK_GUIDE.md) | Retraction alerts from OpenAlex records |

### Source adapters: Crossref, DataCite, and ORCID

| Guide | Source |
| --- | --- |
| [Crossref types](guides/CROSSREF_TYPES_SOURCE_GUIDE.md) | Work-type filters |
| [Crossref relations](guides/CROSSREF_RELATIONS_SOURCE_GUIDE.md) | Work relationships |
| [Crossref journals](guides/CROSSREF_JOURNALS_SOURCE_GUIDE.md) | Journal and ISSN metadata |
| [Crossref members](guides/CROSSREF_MEMBERS_SOURCE_GUIDE.md) | Publisher and registrant metadata |
| [Crossref Funder Registry](guides/CROSSREF_FUNDER_SOURCE_GUIDE.md) | Funder records |
| [Crossref works by funder](guides/CROSSREF_WORKS_FUNDER_SOURCE_GUIDE.md) | Funded-work filters |
| [Crossref works by license](guides/CROSSREF_WORKS_LICENSE_SOURCE_GUIDE.md) | License filters |
| [Crossref works by type and license](guides/CROSSREF_WORKS_TYPE_LICENSE_SOURCE_GUIDE.md) | Combined filters |
| [Crossref works by ISSN and type](guides/CROSSREF_WORKS_ISSN_TYPE_SOURCE_GUIDE.md) | Combined filters |
| [Crossref works by ISBN](guides/CROSSREF_WORKS_ISBN_SOURCE_GUIDE.md) | ISBN filters |
| [Crossref Event Data](guides/CROSSREF_EVENTS_SOURCE_GUIDE.md) | Scholarly activity events |
| [DataCite](guides/DATACITE_SOURCE_GUIDE.md) | DOI metadata |
| [DataCite related identifiers](guides/DATACITE_RELATED_SOURCE_GUIDE.md) | Identifier relationships |
| [DataCite reports](guides/DATACITE_REPORTS_SOURCE_GUIDE.md) | Report DOIs |
| [DataCite DOIs by prefix](guides/DATACITE_DOIS_PREFIX_SOURCE_GUIDE.md) | DOI-prefix filters |
| [DataCite clients and prefixes](guides/DATACITE_CLIENT_PREFIX_SOURCE_GUIDE.md) | Client-scoped DOI lookup |
| [DataCite Event Data](guides/DATACITE_EVENTS_SOURCE_GUIDE.md) | DOI relationship and usage events |
| [ORCID works](guides/ORCID_SOURCE_GUIDE.md) | Public-record works |
| [ORCID works filters](guides/ORCID_WORKS_FILTER_SOURCE_GUIDE.md) | Year and type filters |
| [ORCID works summaries](guides/ORCID_WORKS_SUMMARIES_SOURCE_GUIDE.md) | Public work summaries |
| [ORCID employments](guides/ORCID_EMPLOYMENTS_SOURCE_GUIDE.md) | Public employment affiliations |
| [ORCID education](guides/ORCID_EDUCATION_SOURCE_GUIDE.md) | Public education affiliations |

## Historical project records

[Daily improvements](../DAILY_IMPROVEMENTS.md) and
[the activity log](PULL_SHARK_LOG.md) are historical records, not onboarding
instructions or evidence of current scientific performance.
