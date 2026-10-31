# Documentation catalog

Start with a working local workflow, then choose extensions deliberately.
The guides below remain at their existing URLs; this catalog replaces the
accumulated feature list in the project README.

## Start here

| Goal | Guide |
| --- | --- |
| Install and run without model credentials | [Quickstart](../QUICKSTART.md) |
| Extract a local PDF, review before indexing, and explicitly wire optional OCR | [Executable local PDF ingestion and restart workflow](guides/PDF_OCR_HOOK_GUIDE.md) |
| Work through a small corpus and prepare an honest portfolio demonstration | [Research workflow](guides/RESEARCH_WORKFLOW_GUIDE.md) |
| Browse the current paper catalog and bounded passages with native browser forms/links | [Read-only local corpus explorer and actual browser demo](guides/CORPUS_EXPLORER_GUIDE.md) |
| Browse saved papers, recover IDs after restart, and select a query corpus | [Document catalog and offline demo](guides/DOCUMENT_CATALOG_GUIDE.md) |
| Read bounded passages from a stored paper before choosing query scope | [Stored chunk reader and offline demo](guides/DOCUMENT_CHUNKS_GUIDE.md) |
| Read neighboring passages around an exact current chunk in validated source order | [Source-context reader, API/Python/browser guide and reproducible GIF](guides/SOURCE_CONTEXT_GUIDE.md) |
| Search exact wording in a browser, inspect Unicode highlights and return from source context | [Native passage search, API/Python linkage and actual-output GIF](guides/BROWSER_PASSAGE_SEARCH_GUIDE.md) |
| Find literal wording across current stored papers without ranking or generation | [Literal passage search and measured offline demo](guides/LITERAL_SEARCH_GUIDE.md) |
| Save named paper selections for queries and previews across restarts | [Paper collections and offline demo](guides/PAPER_COLLECTIONS_GUIDE.md) |
| Create a reusable selection in a native browser with explicit exact-ID checkboxes | [Native collection builder: offline/browser/API/Python/portfolio guide and GIF](guides/NATIVE_COLLECTION_BUILDER_GUIDE.md) |
| Rename and explicitly add/remove members without dropping hidden papers | [Native collection editor: browser/API/Python/portfolio guide, executed demo and GIF/still/text](guides/NATIVE_COLLECTION_EDITOR_GUIDE.md) |
| Browse saved membership, human screening and revision-bound downloads without writes | [Native collection browser: end-user/API/Python/portfolio guide and executed GIF](guides/COLLECTION_BROWSER_GUIDE.md) |
| Inspect complete duplicate candidates and singletons in a revision-checked saved selection, without paper loss | [Collection duplicate API/Python guide and measured offline GIF](guides/COLLECTION_DUPLICATES_GUIDE.md) |
| Save human paper labels, recover a screening queue, and select only current included IDs | [Paper screening and measured offline demo](guides/PAPER_SCREENING_GUIDE.md) |
| Hand current human inclusions directly to preview/query without manually copying IDs | [Screened collection scope: API/Python/portfolio guide and executed GIF](guides/SCREENED_COLLECTION_GUIDE.md) |
| Explicitly submit include/exclude/unsure and a reason for an exact member in a native browser, then recover persisted history | [Human screening workbench: user/API/Python/portfolio guide and measured GIF/still/text](guides/HUMAN_SCREENING_WORKBENCH_GUIDE.md) |
| Inspect append-only human decisions, original reasons/timestamps and an explicit legacy boundary without writes | [Screening history: user/API/Python/browser/portfolio guide and executed GIF](guides/SCREENING_HISTORY_GUIDE.md) |
| Download all current screening results as atomic, bounded JSON or spreadsheet-safe CSV | [Screening exports and measured offline demo](guides/SCREENING_EXPORT_GUIDE.md) |
| Save resolved/needs-review totals and auditable exact exclusion-reason groups from one complete snapshot | [Human screening report: browser/API/Python/portfolio guide and executed GIF/still](guides/SCREENING_REPORT_GUIDE.md) |
| Restrict every query retrieval path to selected ingested papers | [Document scope and offline demo](guides/DOCUMENT_SCOPE_GUIDE.md) |
| Inspect the actual prepared evidence without generation or agent-event writes | [Retrieval preview and offline demo](guides/RETRIEVAL_PREVIEW_GUIDE.md) |
| Submit a scoped question in the browser and inspect ordered prepared passages, policies and count diagnostics | [Native retrieval workbench: offline/browser/API/Python/portfolio guide and GIF/still/text](guides/NATIVE_RETRIEVAL_WORKBENCH_GUIDE.md) |
| Compare two evidence policies on one shared query/scope before any generation | [Retrieval policy comparison and measured offline demo](guides/RETRIEVAL_COMPARISON_GUIDE.md) |
| Collapse near-duplicate passages in query/preview with preserved survivor provenance | [Near-duplicate evidence collapse and zero-model demo](guides/NEAR_DUPLICATE_COLLAPSE_GUIDE.md) |
| Apply a per-request passage quota to each paper in query and preview evidence | [Per-paper evidence limits and measured demo](guides/PER_PAPER_EVIDENCE_LIMITS_GUIDE.md) |
| Stop generation below a final-context document minimum while retaining inspectable evidence | [Minimum evidence documents and measured demo](guides/MINIMUM_EVIDENCE_DOCUMENTS_GUIDE.md) |
| Inspect a question-by-paper grid in a native browser form, follow exact sources, and export without generation | [Browser/API/Python research worksheets and measured demos](guides/RESEARCH_WORKSHEET_GUIDE.md) |
| Save and review the exact evidence from a recorded run | [Evidence export](guides/EVIDENCE_EXPORT_GUIDE.md) |
| Read a completed saved answer offline with local links to exact frozen passages | [Offline HTML evidence reader and measured demo](guides/OFFLINE_EVIDENCE_READER_GUIDE.md) |
| Export only a completed saved answer's cited references as BibTeX with provenance | [Saved bibliography and measured offline demo](guides/SAVED_BIBLIOGRAPHY_GUIDE.md) |
| Record human judgments and recover append-only saved-answer review history | [Saved answer reviews and offline demo](guides/ANSWER_REVIEWS_GUIDE.md) |
| Preview an explicit verdict/comment/frozen citations in a native browser, save and reopen the precise opinion | [Saved-answer review workbench: user/API/Python/portfolio guide and actual browser GIF/still/text](guides/SAVED_ANSWER_REVIEW_WORKBENCH_GUIDE.md) |
| Save immutable human notes on exact character spans in frozen run evidence | [Exact quote annotations and measured offline demo](guides/EVIDENCE_ANNOTATIONS_GUIDE.md) |
| Preview and explicitly save a note on one exact frozen occurrence from the native reader/notebook | [Frozen annotation workbench: user/API/Python/portfolio guide and actual browser GIF/still/text](guides/FROZEN_ANNOTATION_WORKBENCH_GUIDE.md) |
| Read independent human-review histories and follow exact frozen quote highlights from a saved run | [Native human-review notebook: user/API/Python/portfolio guide and actual browser GIF](guides/HUMAN_REVIEW_NOTEBOOK_GUIDE.md) |
| Download a whole frozen answer plus complete human-review and annotation histories in one bounded read snapshot | [Portable notebook: API/Python/browser/portfolio guide and executed-output GIF/still](guides/PORTABLE_REVIEW_NOTEBOOK_GUIDE.md) |
| Recover forgotten run IDs and inspect recorded states after restart | [Run history](guides/RUN_HISTORY_GUIDE.md) |
| Find a remembered literal in the complete saved original question, not just its preview | [Saved-question search: browser/API/Python/portfolio guide and measured GIF/still/text](guides/SAVED_QUESTION_SEARCH_GUIDE.md) |
| Diagnose a failed or unfinished saved run without paid retries or rewriting history | [Failed-run triage, safe next actions, and executable offline exercises](guides/FAILED_RUN_TRIAGE_GUIDE.md) |
| Discover saved runs in a browser and follow their validated frozen evidence without rerunning | [Saved-run browser, API/Python guide and actual browser GIF](guides/SAVED_RUN_BROWSER_GUIDE.md) |
| Choose an explicit saved baseline, compare recorded outputs side by side, swap direction and open precise frozen sources | [Native comparison workbench: browser/API/Python/portfolio and measured illustration](guides/SAVED_RUN_COMPARISON_WORKBENCH_GUIDE.md) |
| Inspect exact changes between two saved evidence bundles without rerunning models | [Existing JSON/Python comparison contract](guides/RUN_COMPARISON_GUIDE.md) |
| Check saved source chunks against today's persisted corpus without regeneration | [Corpus drift and measured offline demo](guides/CORPUS_DRIFT_GUIDE.md) |
| Copy public API requests or try connector examples | [Examples](EXAMPLES.md) |
| Reproduce the showcased animation | [Demo](DEMO.md) |
| Understand what the API actually runs | [Architecture](../ARCHITECTURE.md) |
| Choose provider models, evaluate explicit overrides, and plan legacy K2 migration | [Provider models, routing, and migration checklist](guides/PROVIDER_MODELS_GUIDE.md) |
| Inspect every task/fallback before enabling providers and save synthetic fake evidence | [Executable model-routing preflight and provider-free verification](guides/MODEL_ROUTING_PREFLIGHT_GUIDE.md) |
| Compare labeled retrieval runs and enforce local CI quality gates | [Offline retrieval benchmarks](guides/RETRIEVAL_BENCHMARK_GUIDE.md) |

## Configuration, operation, and contribution

| Document | Scope |
| --- | --- |
| [Configuration](../CONFIGURATION.md) | Environment variables and defaults |
| [Extended configuration](CONFIGURATION.md) | Configuration examples |
| [SQLite backup and recovery](guides/SQLITE_BACKUP_RESTORE_GUIDE.md) | Safe new-path backups/restores and an offline synthetic recovery exercise |
| [Safety](../SAFETY.md) | Execution bounds, grounding limitations, and data handling |
| [Troubleshooting](TROUBLESHOOTING.md) | Setup/runtime symptoms and the credential-free saved-failure workflow |
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

The explicitly requested API evidence controls listed above are exceptions:
near-duplicate collapse and per-paper caps run only when their request fields
are supplied. Their underlying helpers remain available independently.

Most scoring, screening, extraction, and "verification" helpers use deterministic
lexical or metadata heuristics. Their labels are advisory, not proof of factual
support, study quality, causality, novelty, or completeness. Connector calls may
need network access, service credentials, and permission to use the returned text.

Older guides and their generated animations retain historical model names and
illustrative examples. Use the dated, source-linked
[provider model guide](guides/PROVIDER_MODELS_GUIDE.md) for current defaults and
routing; an old provider name in a helper guide is not a compatibility promise.

Choose [planning and context](#query-planning-and-context),
[ranking and diversity](#ranking-and-diversity), [gates and filters](#gates-and-filters),
[evidence and corpus organization](#evidence-inspection-and-corpus-organization),
[clinical and statistical cues](#clinical-and-statistical-evidence-cues), or
[source adapters](#source-adapters-papers-repositories-and-registries).

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
| [Local PDF ingestion and explicit OCR](guides/PDF_OCR_HOOK_GUIDE.md) | Extract, inspect, persist, and recover saved IDs; standalone OCR is not enabled by the connector |
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
| [Retraction watch flagger](guides/RETRACTION_FLAGGER_GUIDE.md) | Advisory flags from caller-supplied offline sets |
| [PRISMA screening checklist](guides/PRISMA_SCREENING_CHECKLIST_GUIDE.md) | Pending human-review rows, never automatic include/exclude |
| [Reading-list prioritization](guides/READING_LIST_PRIORITIZER_GUIDE.md) | Explainable unread-queue triage |
| [Paper chat memory](guides/PAPER_CHAT_MEMORY_GUIDE.md) | Library-managed paper-scoped chat turns |
| [Literature review outlines](guides/LITERATURE_REVIEW_OUTLINE_GUIDE.md) | Deterministic section scaffolding |
| [Related works](guides/RELATED_WORKS_GUIDE.md) | Theme-based outline scaffolding |
| [Survey gaps](guides/SURVEY_GAP_GUIDE.md) | Coverage of caller-provided themes, not proof of novelty |
| [BibTeX export](guides/BIBTEX_EXPORT_GUIDE.md) | Bibliography entries from supplied metadata |
| [Evaluation harness](guides/EVALUATION_HARNESS_GUIDE.md) | Labeled retrieval cases and lexical answer metrics |

### Clinical and statistical evidence cues

These offline helpers locate reported wording or numbers in caller-supplied
paper text. They do not calculate missing statistics, assess treatment benefit,
or replace human study-quality review. A missing cue is not evidence that a
study has no bias. They require explicit Python integration; `/query` does not
run them automatically. Usage examples and individual GIFs remain in each guide.

#### Study design, conduct, and outcomes

| Guide | Reported cues |
| --- | --- |
| [Method extraction cards](guides/METHOD_EXTRACT_CARD_GUIDE.md) | Study-design and methods cues |
| [Allocation concealment](guides/ALLOCATION_CONCEALMENT_CUE_EXTRACTOR_GUIDE.md) | Allocation-concealment wording |
| [Blinding status](guides/BLINDING_STATUS_CUE_EXTRACTOR_GUIDE.md) | Blinding and masking descriptions |
| [Cluster randomization](guides/CLUSTER_RANDOMIZATION_CUE_EXTRACTOR_GUIDE.md) | Cluster-randomized, stepped-wedge, and ICC wording |
| [Crossover design](guides/CROSSOVER_DESIGN_CUE_EXTRACTOR_GUIDE.md) | Crossover and treatment-sequence descriptions |
| [Estimands](guides/ESTIMAND_ICH_E9_CUE_EXTRACTOR_GUIDE.md) | ICH-E9(R1) estimand strategy wording |
| [Per-protocol analysis](guides/PER_PROTOCOL_ANALYSIS_CUE_EXTRACTOR_GUIDE.md) | Per-protocol / as-treated analysis wording |
| [Bayesian interim priors](guides/BAYESIAN_INTERIM_PRIOR_CUE_EXTRACTOR_GUIDE.md) | Bayesian interim / prior / posterior wording |
| [Transportability](guides/TRANSPORTABILITY_CUE_EXTRACTOR_GUIDE.md) | Transportability / external-validity / generalizability wording |
| [Competing risks](guides/COMPETING_RISK_CUE_EXTRACTOR_GUIDE.md) | Competing-risk / Fine-Gray / cumulative-incidence wording |
| [Regression discontinuity](guides/REGRESSION_DISCONTINUITY_CUE_EXTRACTOR_GUIDE.md) | RDD / running-variable / bandwidth wording |
| [Interrupted time series](guides/INTERRUPTED_TIME_SERIES_CUE_EXTRACTOR_GUIDE.md) | ITS / level-change / slope-change wording |
| [Event study](guides/EVENT_STUDY_CUE_EXTRACTOR_GUIDE.md) | offline cue extractor |
| [Heterogeneous treatment effects](guides/HETEROGENEOUS_TREATMENT_EFFECT_CUE_EXTRACTOR_GUIDE.md) | HTE / CATE / effect-modifier wording |
| [E-value sensitivity](guides/EVALUE_SENSITIVITY_CUE_EXTRACTOR_GUIDE.md) | E-value / sensitivity / unmeasured-confounding wording |
| [Measurement-error bias](guides/MEASUREMENT_ERROR_BIAS_CUE_EXTRACTOR_GUIDE.md) | Measurement error / misclassification / information-bias wording |
| [Multiple-testing correction](guides/MULTIPLE_TESTING_CORRECTION_CUE_EXTRACTOR_GUIDE.md) | Bonferroni / FDR / family-wise error wording |
| [Healthy-worker effect](guides/HEALTHY_WORKER_EFFECT_CUE_EXTRACTOR_GUIDE.md) | Healthy-worker effect / employed-population bias wording |
| [Recall bias](guides/RECALL_BIAS_CUE_EXTRACTOR_GUIDE.md) | Recall / reporting / retrospective-recall bias wording |
| [Interviewer bias](guides/INTERVIEWER_BIAS_CUE_EXTRACTOR_GUIDE.md) | Interviewer / observer / ascertainment bias wording |
| [Left truncation](guides/LEFT_TRUNCATION_CUE_EXTRACTOR_GUIDE.md) | Left-truncation / delayed-entry / late-entry wording |
| [Collider stratification](guides/COLLIDER_STRATIFICATION_CUE_EXTRACTOR_GUIDE.md) | Collider-stratification / Berkson bias wording |
| [Immortal-time bias](guides/IMMORTAL_TIME_BIAS_CUE_EXTRACTOR_GUIDE.md) | Immortal-time / guarantee-time bias wording |
| [Selection bias](guides/SELECTION_BIAS_CUE_EXTRACTOR_GUIDE.md) | Selection-bias / healthy-user / volunteer-bias wording |
| [Dose-response](guides/DOSE_RESPONSE_CUE_EXTRACTOR_GUIDE.md) | Dose-response / exposure-response / ED50 wording |
| [Spillover / interference](guides/SPILLOVER_INTERFERENCE_CUE_EXTRACTOR_GUIDE.md) | Spillover / interference / SUTVA wording |
| [Placebo tests](guides/PLACEBO_TEST_CUE_EXTRACTOR_GUIDE.md) | Placebo-test / falsification / null-effect wording |
| [Negative control exposures](guides/NEGATIVE_CONTROL_EXPOSURE_CUE_EXTRACTOR_GUIDE.md) | Negative-control exposure / falsification-exposure wording |
| [Fuzzy RDD](guides/FUZZY_RDD_CUE_EXTRACTOR_GUIDE.md) | offline cue extractor |
| [Doubly robust / IPTW](guides/DOUBLY_ROBUST_IPTW_CUE_EXTRACTOR_GUIDE.md) | offline cue extractor |
| [Propensity score matching](guides/PROPENSITY_SCORE_MATCHING_CUE_EXTRACTOR_GUIDE.md) | PSM / caliper / SMD balance wording |
| [Synthetic control](guides/SYNTHETIC_CONTROL_CUE_EXTRACTOR_GUIDE.md) | Synthetic-control / donor-pool / placebo-in-space wording |
| [Difference in differences](guides/DIFFERENCE_IN_DIFFERENCES_CUE_EXTRACTOR_GUIDE.md) | Difference-in-differences / parallel-trends / TWFE wording |
| [Negative control outcomes](guides/NEGATIVE_CONTROL_OUTCOME_CUE_EXTRACTOR_GUIDE.md) | Negative-control / falsification-outcome wording |
| [Mendelian randomization](guides/MENDELIAN_RANDOMIZATION_CUE_EXTRACTOR_GUIDE.md) | MR / genetic instrument / pleiotropy wording |
| [IV strength](guides/INSTRUMENTAL_VARIABLE_STRENGTH_CUE_EXTRACTOR_GUIDE.md) | Instrumental-variable / F-stat / weak-instrument wording |
| [Time-varying confounding](guides/TIME_VARYING_CONFOUNDING_CUE_EXTRACTOR_GUIDE.md) | Time-varying / MSM / g-methods confounding wording |
| [Mediation analysis](guides/MEDIATION_ANALYSIS_CUE_EXTRACTOR_GUIDE.md) | Mediator / indirect-effect / path-analysis wording |
| [Confounding adjustment](guides/CONFOUNDING_ADJUSTMENT_CUE_EXTRACTOR_GUIDE.md) | Propensity / IPTW / confounding-adjustment wording |
| [Missing-data mechanisms](guides/MISSING_DATA_MECHANISM_CUE_EXTRACTOR_GUIDE.md) | MCAR/MAR/MNAR missingness wording |
| [Adaptive designs](guides/ADAPTIVE_DESIGN_CUE_EXTRACTOR_GUIDE.md) | SSR, RAR, and seamless adaptive-design wording |
| [Washout periods](guides/WASHOUT_PERIOD_CUE_EXTRACTOR_GUIDE.md) | Washout, run-in, and wash-in periods |
| [Baseline imbalance](guides/BASELINE_IMBALANCE_CUE_EXTRACTOR_GUIDE.md) | Baseline, covariate, and Table-1 imbalance wording |
| [Intention to treat](guides/INTENTION_TO_TREAT_CUE_EXTRACTOR_GUIDE.md) | Analysis-population descriptions |
| [Protocol deviations](guides/PROTOCOL_DEVIATION_CUE_EXTRACTOR_GUIDE.md) | Protocol-deviation wording |
| [Interim analysis](guides/INTERIM_ANALYSIS_CUE_EXTRACTOR_GUIDE.md) | Interim analysis, monitoring, and early stopping |
| [Adaptive designs](guides/ADAPTIVE_DESIGN_CUE_EXTRACTOR_GUIDE.md) | Sample-size re-estimation, adaptive randomization, and seamless-design wording |
| [Noninferiority margins](guides/NONINFERIORITY_MARGIN_CUE_EXTRACTOR_GUIDE.md) | Noninferiority and equivalence-margin wording |
| [Multiplicity adjustment](guides/MULTIPLICITY_ADJUSTMENT_CUE_EXTRACTOR_GUIDE.md) | Multiple-testing and adjustment descriptions |
| [Primary endpoints](guides/PRIMARY_ENDPOINT_CUE_EXTRACTOR_GUIDE.md) | Primary-outcome and endpoint wording |
| [Surrogate endpoints](guides/SURROGATE_ENDPOINT_CUE_EXTRACTOR_GUIDE.md) | Surrogate-outcome descriptions |
| [Follow-up duration](guides/FOLLOW_UP_DURATION_CUE_EXTRACTOR_GUIDE.md) | Follow-up and observation periods |
| [Attrition rates](guides/ATTRITION_RATE_CUE_EXTRACTOR_GUIDE.md) | Dropout and loss-to-follow-up reporting |
| [Subgroup analysis](guides/SUBGROUP_ANALYSIS_CUE_EXTRACTOR_GUIDE.md) | Subgroup and interaction-analysis wording |
| [Adverse events](guides/ADVERSE_EVENT_CUE_EXTRACTOR_GUIDE.md) | Adverse-event and safety-reporting wording |

#### Statistical reporting hints

| Guide | Reported cues |
| --- | --- |
| [Sample-size hints](guides/SAMPLE_SIZE_HINT_EXTRACTOR_GUIDE.md) | Sample-size integers in text |
| [Effect-size hints](guides/EFFECT_SIZE_HINT_EXTRACTOR_GUIDE.md) | Labeled effect-size values |
| [P-value hints](guides/P_VALUE_HINT_EXTRACTOR_GUIDE.md) | Significance-value patterns |
| [Confidence-interval hints](guides/CONFIDENCE_INTERVAL_HINT_EXTRACTOR_GUIDE.md) | Interval bounds in text |
| [Heterogeneity I2 hints](guides/HETEROGENEITY_I2_HINT_EXTRACTOR_GUIDE.md) | Meta-analysis I2 / heterogeneity cues |
| [Statistical power](guides/STATISTICAL_POWER_HINT_EXTRACTOR_GUIDE.md) | Power percentages and power-calculation wording |
| [Absolute risk reduction](guides/ARR_HINT_EXTRACTOR_GUIDE.md) | ARR and risk-difference numeric hints |
| [Number needed to treat](guides/NNT_HINT_EXTRACTOR_GUIDE.md) | NNT, NNH, and ARR metric/value hints |

#### Limitations, bias, and disclosures

| Guide | Reported cues |
| --- | --- |
| [Study limitations](guides/STUDY_LIMITATION_CUE_EXTRACTOR_GUIDE.md) | Limitation-language cues |
| [Risk of bias](guides/RISK_OF_BIAS_CUE_EXTRACTOR_GUIDE.md) | Advisory risk-of-bias domain wording, not a completed assessment |
| [Healthy-adherer bias](guides/HEALTHY_ADHERER_BIAS_CUES_EXTRACTOR_GUIDE.md) | Healthy-adherer bias wording |
| [Measurement scale shift](guides/MEASUREMENT_SCALE_SHIFT_CUES_EXTRACTOR_GUIDE.md) | Measurement-scale shift / recalibration wording |
| [Overadjustment bias](guides/OVERADJUSTMENT_BIAS_CUES_EXTRACTOR_GUIDE.md) | Overadjustment / overcontrol / mediator-adjustment wording |
| [Competing-risk informative censoring](guides/COMPETING_RISK_INFORMATIVE_CENSORING_CUES_EXTRACTOR_GUIDE.md) | Competing-risk informative / dependent censoring wording |
| [Target Trial Emulation](guides/TARGET_TRIAL_EMULATION_CUES_EXTRACTOR_GUIDE.md) | target trial emulation wording |
| [Instrument Strength](guides/INSTRUMENT_STRENGTH_CUES_EXTRACTOR_GUIDE.md) | instrument strength wording |
| [Healthy Worker Survivor](guides/HEALTHY_WORKER_SURVIVOR_CUES_EXTRACTOR_GUIDE.md) | healthy worker survivor wording |
| [Survey mode effect](guides/SURVEY_MODE_EFFECT_CUES_EXTRACTOR_GUIDE.md) | Survey-mode effect / mixed-mode survey bias wording |
| [Collider bias interaction](guides/COLLIDER_BIAS_INTERACTION_CUES_EXTRACTOR_GUIDE.md) | Collider-bias interaction / collider stratification interaction wording |
| [Ascertainment differential residual](guides/ASCERTAINMENT_DIFFERENTIAL_RESIDUAL_CUES_EXTRACTOR_GUIDE.md) | Ascertainment-differential residual / residual ascertainment bias wording |
| [Neyman bias](guides/NEYMAN_BIAS_CUES_EXTRACTOR_GUIDE.md) | Neyman bias / selective-survival wording |
| [Observer-expectancy bias](guides/OBSERVER_EXPECTANCY_BIAS_CUES_EXTRACTOR_GUIDE.md) | Observer-expectancy bias wording |
| [Loss-to-follow-up bias](guides/LOSS_TO_FOLLOWUP_BIAS_CUES_EXTRACTOR_GUIDE.md) | Loss-to-follow-up wording |
| [Late-looker bias](guides/LATE_LOOKER_BIAS_CUES_EXTRACTOR_GUIDE.md) | Late-looker bias wording |
| [Right-truncation bias](guides/RIGHT_TRUNCATION_BIAS_CUES_EXTRACTOR_GUIDE.md) | Right-truncation wording |
| [Publication bias](guides/PUBLICATION_BIAS_CUE_EXTRACTOR_GUIDE.md) | Funnel plots, Egger's test, and small-study effects |
| [Funding disclosures](guides/FUNDING_DISCLOSURE_FLAGGER_GUIDE.md) | Funding and grant cues |
| [Funding conflicts](guides/FUNDING_CONFLICT_CUE_EXTRACTOR_GUIDE.md) | Funding and conflict-of-interest wording |
| [Conflict-of-interest flags](guides/CONFLICT_OF_INTEREST_FLAGGER_GUIDE.md) | Disclosure cues |
| [Preregistration flags](guides/PREREGISTRATION_FLAG_DETECTOR_GUIDE.md) | Registry and preregistration cues |
| [Open-data availability](guides/OPEN_DATA_AVAILABILITY_FLAGGER_GUIDE.md) | Data-availability cues |

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

| Extract offline ConfoundingByIndication cues | [ConfoundingByIndicationCueExtractor](guides/CONFOUNDING_BY_INDICATION_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline LengthBias cues | [LengthBiasCueExtractor](guides/LENGTH_BIAS_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline LeadTimeBias cues | [LeadTimeBiasCueExtractor](guides/LEAD_TIME_BIAS_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline BerksonBias cues | [BerksonBiasCueExtractor](guides/BERKSON_BIAS_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline VolunteerBias cues | [VolunteerBiasCueExtractor](guides/VOLUNTEER_BIAS_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline ReportingBias cues | [ReportingBiasCueExtractor](guides/REPORTING_BIAS_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline IndicationBias cues | [IndicationBiasCueExtractor](guides/INDICATION_BIAS_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline ReverseCausation cues | [ReverseCausationCueExtractor](guides/REVERSE_CAUSATION_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline SurvivorBias cues | [SurvivorBiasCueExtractor](guides/SURVIVOR_BIAS_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline IntervalCensoring cues | [IntervalCensoringCueExtractor](guides/INTERVAL_CENSORING_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline EcologicalFallacy cues | [EcologicalFallacyCueExtractor](guides/ECOLOGICAL_FALLACY_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline DetectionBias cues | [DetectionBiasCueExtractor](guides/DETECTION_BIAS_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline HealthyUserBias cues | [HealthyUserBiasCueExtractor](guides/HEALTHY_USER_BIAS_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline InformationBias cues | [InformationBiasCueExtractor](guides/INFORMATION_BIAS_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline NoveltyBias cues | [NoveltyBiasCueExtractor](guides/NOVELTY_BIAS_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline ChannelingBias cues | [ChannelingBiasCueExtractor](guides/CHANNELING_BIAS_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline PrevalenceIncidenceBias cues | [PrevalenceIncidenceBiasCueExtractor](guides/PREVALENCE_INCIDENCE_BIAS_CUES_EXTRACTOR_GUIDE.md) |
| Extract offline IncorporationBias cues | [IncorporationBiasCueExtractor](guides/INCORPORATION_BIAS_CUES_EXTRACTOR_GUIDE.md) |
| Extract offline WillRogersPhenomenon cues | [WillRogersPhenomenonCueExtractor](guides/WILL_ROGERS_PHENOMENON_CUES_EXTRACTOR_GUIDE.md) |
| Extract offline NonDifferentialMisclassification cues | [NonDifferentialMisclassificationCueExtractor](guides/NON_DIFFERENTIAL_MISCLASSIFICATION_CUES_EXTRACTOR_GUIDE.md) |
| Extract offline SpinPHacking cues | [SpinPHackingCueExtractor](guides/SPIN_P_HACKING_CUES_EXTRACTOR_GUIDE.md) |
| Extract offline DifferentialMisclassification cues | [DifferentialMisclassificationCueExtractor](guides/DIFFERENTIAL_MISCLASSIFICATION_CUES_EXTRACTOR_GUIDE.md) |
| Extract offline ConfoundingByContraindication cues | [ConfoundingByContraindicationCueExtractor](guides/CONFOUNDING_BY_CONTRAINDICATION_CUES_EXTRACTOR_GUIDE.md) |
| Extract offline SocialDesirabilityBias cues | [SocialDesirabilityBiasCueExtractor](guides/SOCIAL_DESIRABILITY_BIAS_CUES_EXTRACTOR_GUIDE.md) |
| Extract offline IndexEventBias cues | [IndexEventBiasCueExtractor](guides/INDEX_EVENT_BIAS_CUES_EXTRACTOR_GUIDE.md) |
| Extract offline CompetingEventBias cues | [CompetingEventBiasCueExtractor](guides/COMPETING_EVENT_BIAS_CUES_EXTRACTOR_GUIDE.md) |
| Extract offline DiagnosticAccessBias cues | [DiagnosticAccessBiasCueExtractor](guides/DIAGNOSTIC_ACCESS_BIAS_CUES_EXTRACTOR_GUIDE.md) |
| Extract offline ReferralBias cues | [ReferralBiasCueExtractor](guides/REFERRAL_BIAS_CUES_EXTRACTOR_GUIDE.md) |
| Extract offline HawthorneEffect cues | [HawthorneEffectCueExtractor](guides/HAWTHORNE_EFFECT_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline RightCensoring cues | [RightCensoringCueExtractor](guides/RIGHT_CENSORING_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline ResidualConfounding cues | [ResidualConfoundingCueExtractor](guides/RESIDUAL_CONFOUNDING_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline SimpsonParadox cues | [SimpsonParadoxCueExtractor](guides/SIMPSON_PARADOX_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline ProtopathicBias cues | [ProtopathicBiasCueExtractor](guides/PROTOPATHIC_BIAS_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline DepletionOfSusceptibles cues | [DepletionOfSusceptiblesCueExtractor](guides/DEPLETION_OF_SUSCEPTIBLES_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline OverdiagnosisBias cues | [OverdiagnosisBiasCueExtractor](guides/OVERDIAGNOSIS_BIAS_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline AscertainmentBias cues | [AscertainmentBiasCueExtractor](guides/ASCERTAINMENT_BIAS_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline AttritionInformativeCensoring cues | [AttritionInformativeCensoringCueExtractor](guides/ATTRITION_INFORMATIVE_CENSORING_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline VerificationBias cues | [VerificationBiasCueExtractor](guides/VERIFICATION_BIAS_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline SpectrumBias cues | [SpectrumBiasCueExtractor](guides/SPECTRUM_BIAS_CUE_EXTRACTOR_GUIDE.md) |
| Extract offline AtomisticFallacy cues | [AtomisticFallacyCueExtractor](guides/ATOMISTIC_FALLACY_CUES_EXTRACTOR_GUIDE.md) |
| Extract offline OutcomeReportingBias cues | [OutcomeReportingBiasCueExtractor](guides/OUTCOME_REPORTING_BIAS_CUES_EXTRACTOR_GUIDE.md) |
| Extract offline NonresponseBias cues | [NonresponseBiasCueExtractor](guides/NONRESPONSE_BIAS_CUES_EXTRACTOR_GUIDE.md) |
| Extract offline PrevalentUserBias cues | [PrevalentUserBiasCueExtractor](guides/PREVALENT_USER_BIAS_CUES_EXTRACTOR_GUIDE.md) |
| Extract offline TimeDependentBias cues | [TimeDependentBiasCueExtractor](guides/TIME_DEPENDENT_BIAS_CUES_EXTRACTOR_GUIDE.md) |
| Extract offline LengthTimeBias cues | [LengthTimeBiasCueExtractor](guides/LENGTH_TIME_BIAS_CUES_EXTRACTOR_GUIDE.md) |
- [GComputationCueExtractor](guides/G_COMPUTATION_CUES_EXTRACTOR_GUIDE.md)
- [InformativeCensoringCueExtractor](guides/INFORMATIVE_CENSORING_CUES_EXTRACTOR_GUIDE.md)
- [OutcomeSwitchingCueExtractor](guides/OUTCOME_SWITCHING_CUES_EXTRACTOR_GUIDE.md)
- [SelectionIntoTreatmentCueExtractor](guides/SELECTION_INTO_TREATMENT_CUES_EXTRACTOR_GUIDE.md)
- [MendelianRandomizationWeakCueExtractor](guides/MENDELIAN_RANDOMIZATION_WEAK_CUES_EXTRACTOR_GUIDE.md)
- [WinnerCurseCueExtractor](guides/WINNER_CURSE_CUES_EXTRACTOR_GUIDE.md)
- [CompetingRisksResidualCueExtractor](guides/COMPETING_RISKS_RESIDUAL_CUES_EXTRACTOR_GUIDE.md)
- [MediationBiasCueExtractor](guides/MEDIATION_BIAS_CUES_EXTRACTOR_GUIDE.md)
- [ProtocolAmendmentCueExtractor](guides/PROTOCOL_AMENDMENT_CUES_EXTRACTOR_GUIDE.md)
