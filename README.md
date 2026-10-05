# Scholar RAG Agent

[![CI](https://github.com/Francis1998/scholar-rag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Francis1998/scholar-rag-agent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Load a small paper corpus, ask comparison or hypothesis questions, and inspect
the passages and run history behind the response. Scholar RAG Agent is a
**local-first Python toolkit and FastAPI service** for building inspectable
literature workflows, with SQLite persistence and optional model-provider adapters.

The default setup works without model credentials. Its fake adapter demonstrates
the workflow; it does **not** produce a scientific summary or validate a hypothesis.

## Start offline

With Python 3.11+ and [uv](https://docs.astral.sh/uv/) installed:

```bash
git clone https://github.com/Francis1998/scholar-rag-agent.git
cd scholar-rag-agent
uv sync --extra dev
uv run python scripts/demo_local.py
```

The demo explicitly uses the fake model, ingests a synthetic fixture into a
temporary database, and prints a run ID, `DONE` state, planner trace, and cited
placeholder answer. Its temporary database is removed on exit.

Next, use the [Quickstart](QUICKSTART.md) to start the API with an isolated
database and empty provider keys. The interactive API documentation is at
`http://127.0.0.1:8000/docs`; this is not a paper-chat or PDF-upload UI.

## Choose a workflow

| What you can do | Integrated path | Next guide |
| --- | --- | --- |
| Explore a corpus you own or may process | Ingest text, ask a question, inspect source IDs and snippets | [Quickstart](QUICKSTART.md) |
| Recover paper IDs after restart | Browse bounded document summaries, filter by source/title, and select papers for queries | [Document catalog](docs/guides/DOCUMENT_CATALOG_GUIDE.md) |
| Find exact wording across stored papers | `POST /research/search` returns literal matches, Unicode offsets, and bounded excerpts without retrieval, generation, or run writes | [Literal passage search](docs/guides/LITERAL_SEARCH_GUIDE.md) |
| Reuse a named paper selection after restart | Save a collection, then pass `collection_id` to `/query` or `/retrieve`; revisioned edits preserve old evidence | [Paper collections](docs/guides/PAPER_COLLECTIONS_GUIDE.md) |
| Screen papers before choosing evidence | Save human include/exclude/unsure decisions, resume the queue, and explicitly preview current included IDs | [Human paper screening](docs/guides/PAPER_SCREENING_GUIDE.md) |
| Download complete human screening results | Export every current collection member in one read snapshot as bounded JSON or spreadsheet-safe CSV, including stale and unscreened states | [Screening exports and measured offline demo](docs/guides/SCREENING_EXPORT_GUIDE.md) |
| Restrict a query to selected ingested papers | Pass `document_ids` through hybrid and graph retrieval; preserve scope in the saved evidence | [Document scope](docs/guides/DOCUMENT_SCOPE_GUIDE.md) |
| Inspect evidence before generating | `POST /retrieve` returns the actual prepared chunks and plan without any live/fake LLM call or agent-event writes | [Retrieval preview](docs/guides/RETRIEVAL_PREVIEW_GUIDE.md) |
| Collapse overlapping evidence before generating | Opt in to `near_duplicate_threshold` on `/query` or `/retrieve`; preserve exact survivors and reviewable transformation paths | [Near-duplicate evidence collapse](docs/guides/NEAR_DUPLICATE_COLLAPSE_GUIDE.md) |
| Limit how many passages one paper contributes | Opt in to `max_chunks_per_document` on `/query` or `/retrieve`; retain the quota and gate provenance in saved evidence | [Per-paper evidence limits](docs/guides/PER_PAPER_EVIDENCE_LIMITS_GUIDE.md) |
| Require a minimum number of evidence documents before generating | Opt in to `min_evidence_documents`; inspect count diagnostics with `/retrieve`, or retain exact evidence in an `ERROR` run without generation | [Minimum evidence documents](docs/guides/MINIMUM_EVIDENCE_DOCUMENTS_GUIDE.md) |
| Inspect the same questions against each selected paper | Build a bounded, model-free question-by-paper worksheet and save JSON/Markdown with passage provenance | [Research worksheets](docs/guides/RESEARCH_WORKSHEET_GUIDE.md) |
| Compare methods or explore a hypothesis | Inspect comparison or supporting/counter-evidence retrieval tasks, then review the merged evidence | [Research workflow](docs/guides/RESEARCH_WORKFLOW_GUIDE.md) |
| Preserve a reviewable answer and its context | Export a completed run as JSON or Markdown with its exact recorded source chunks | [Evidence export](docs/guides/EVIDENCE_EXPORT_GUIDE.md) |
| Import only a saved answer's cited references | Download frozen-source BibTeX plus exact document/chunk provenance and metadata warnings; no generation or DOI lookup | [Saved bibliography and measured demo](docs/guides/SAVED_BIBLIOGRAPHY_GUIDE.md) |
| Record a human judgment on a saved answer | Append `accepted`, `needs_revision`, or `rejected` opinions with comments and frozen chunk references; recover history after restart | [Saved answer reviews](docs/guides/ANSWER_REVIEWS_GUIDE.md) |
| Attach a human note to an exact saved passage | Validate frozen source IDs, SHA-256, Unicode offsets, and quote; keep immutable retry-safe notes after corpus changes and restart | [Exact frozen-evidence annotations](docs/guides/EVIDENCE_ANNOTATIONS_GUIDE.md) |
| Find a previous run after restart | Page through saved query previews and recorded states, then follow events/export links | [Run history](docs/guides/RUN_HISTORY_GUIDE.md) |
| Review changes between two completed runs | Compare frozen queries, scope, configuration, answers, and evidence without retrieval or generation | [Saved-run comparison](docs/guides/RUN_COMPARISON_GUIDE.md) |
| Check whether saved source chunks still match the local corpus | Read unchanged/changed/missing findings in frozen rank order, without regenerating or rewriting evidence | [Corpus drift and measured demo](docs/guides/CORPUS_DRIFT_GUIDE.md) |
| Demonstrate your engineering work | Use synthetic notes, review warnings, save artifacts, and explain limitations | [Portfolio walkthrough](docs/guides/RESEARCH_WORKFLOW_GUIDE.md#6-present-a-portfolio-demonstration) |
| Catch retrieval regressions before a release | Compare real BM25/hybrid rankings on labeled passages and enforce per-retriever quality gates | [Offline benchmarks](docs/guides/RETRIEVAL_BENCHMARK_GUIDE.md) |
| Extend ingestion or retrieval | Explicitly wire Python connectors, ranking helpers, or screening utilities | [Categorized catalog](docs/README.md) |

## Inspect near-duplicate evidence before generating

![Measured synthetic offline evidence collapse](docs/assets/near-duplicate-evidence.gif)

`near_duplicate_threshold` reuses the existing lexical collapser after reranking,
before paper quotas and minimum-document assessment. The measured fixture drops
five passages to three and changes context from 435 to 264 UTF-8 bytes, with zero
model/network calls. Threshold `1` means equal meaningful-term sets, not identical
text. Similar passages may contain important differences: inspect a baseline and
the [complete guide](docs/guides/NEAR_DUPLICATE_COLLAPSE_GUIDE.md), not a scientific
equivalence claim. Omission leaves existing behavior unchanged.

## Find exact wording without ranking

![Measured synthetic offline literal passage search](docs/assets/literal-search.gif)

`POST /research/search` finds literal phrases in current persisted chunks,
including text beyond the normal chunk prefix. Inspect exact Unicode match
offsets and bounded excerpts, narrow scope by paper or collection, and page in
stable ID order without retrieval, generation, or run writes. The
[complete API/Python guide](docs/guides/LITERAL_SEARCH_GUIDE.md) covers limits,
privacy, current-corpus semantics, and reproduction of this measured illustration.

## Compare selected papers before generating

![Measured synthetic offline research worksheet](docs/assets/research-worksheet.gif)

`POST /research/worksheet` inspects each question against each selected paper,
so one paper's global ranking does not crowd another out of the worksheet.
It returns retrieved passages, not generated answers or scientific judgments.
The [complete guide](docs/guides/RESEARCH_WORKSHEET_GUIDE.md) includes API/Python
examples, fixed bounds, privacy, and reproduction of this measured illustration.

## Screen papers before retrieving evidence

![Measured synthetic human paper-screening workflow](docs/assets/paper-screening.gif)

Collection screening saves human labels and reasons without generation or agent
events. Resume after restart, inspect stale decisions after a collection edit,
and explicitly pass current included IDs to retrieval. It is not automated
screening or scientific validation. The [complete guide](docs/guides/PAPER_SCREENING_GUIDE.md)
includes API/Python usage, revision conflicts, privacy limits, and GIF reproduction.

## Export complete human screening results

![Measured synthetic offline screening-result exports](docs/assets/screening-exports.gif)

`GET /collections/{collection_id}/screening/export?collection_revision=N&format=json|csv`
downloads all current members, labels, reasons, timestamps, revisions and counts
in one read transaction. Stale includes never become current retrieval IDs.
CSV text cells use a reversible apostrophe prefix; oversized full downloads fail
instead of silently dropping rows. This measured synthetic/offline illustration
is not a live UI, active learning, a PRISMA audit or frozen paper contents.
The [complete API/Python guide](docs/guides/SCREENING_EXPORT_GUIDE.md) covers
escaping, limits, errors, privacy, peer workflow attribution and reproduction.

## Export cited sources to a reference manager

![Measured synthetic offline saved bibliography export](docs/assets/saved-bibliography.gif)

`GET /runs/{run_id}/bibliography` downloads BibTeX from only the completed saved
answer's cited chunks, deduplicated by exact document ID in frozen rank order.
Add `?format=json` for captured metadata, cited-chunk mappings, and explicit
conflict/empty-citation warnings. No current corpus, model, or DOI service is
consulted. Metadata is unverified; review before import, not by compiling LaTeX.
The [complete guide](docs/guides/SAVED_BIBLIOGRAPHY_GUIDE.md) includes offline
API/Python usage and reproduction of this actual-output illustration.

## Keep human notes on exact frozen quotes

![Measured synthetic offline exact-quote annotations](docs/assets/evidence-annotations.gif)

`POST /runs/{run_id}/annotations` saves a human note at an exact Unicode
character span in completed frozen evidence. Repeated phrases stay tied to
their selected occurrence; UUID retries are idempotent and changed payloads
conflict instead of overwriting. GET recovers bounded history after restart
even when the current corpus is removed. No model, retrieval, or agent-event
write is involved. A note is an opinion and a quote is provenance, not proof.
The [complete API/Python guide](docs/guides/EVIDENCE_ANNOTATIONS_GUIDE.md) covers
selectors, errors, privacy, and reproduction from measured synthetic results.

## Inspect a recorded run

To inspect retrieval **without creating a run**, use `POST /retrieve` or
`await container.runner.preview(...)`. The [retrieval preview guide](docs/guides/RETRIEVAL_PREVIEW_GUIDE.md)
includes a reproducible offline GIF, Python/API examples, exact chunk/rank/path
and context-digest contracts, scope, and errors. This is the shared `/query`
context preparation, not semantic entailment or a promise about a changed corpus.

To block answer generation below a chosen distinct-document count, use
[`min_evidence_documents`](docs/guides/MINIMUM_EVIDENCE_DOCUMENTS_GUIDE.md).
Previews with an unmet minimum keep their passages; queries with an unmet
minimum preserve a diagnostic and snapshot without generating. Operational
preview failures return errors, not partial evidence. Passing a count is not
proof of scientific support.

![Synthetic offline evidence-export walkthrough](docs/assets/evidence-export.gif)

This generated animation illustrates the synthetic offline evidence-export demo,
not a live research UI or a real model's scientific findings. Follow the
[demo reproduction instructions](docs/DEMO.md) to inspect the actual output.

Forgot the run ID? `GET /runs?limit=20` discovers persisted runs with bounded
query previews and creation-ordered pagination. Its recorded state is not a
liveness claim; see [run history and restart recovery](docs/guides/RUN_HISTORY_GUIDE.md).

`GET /runs/{run_id}/export?format=json|markdown` reconstructs a completed run from
stored evidence, without another retrieval or generation call. Keep the query,
plan, answer, claims, exact source chunks, trace, and nonsecret model provenance
together for review. The saved context survives corpus changes and restart; this
is not a guarantee of identical output from a new LLM run or a signed audit record.

`POST /runs/{run_id}/reviews` records a bounded human judgment and comment in a
separate SQLite table. `GET /runs/{run_id}/reviews` reads retry-safe, paginated
history after restart without changing the saved answer or events. Acceptance is
an opinion, not factual verification or an authenticated approval. See the
[review guide and actual-output offline GIF](docs/guides/ANSWER_REVIEWS_GUIDE.md).

![Measured synthetic offline corpus-drift report](docs/assets/corpus-drift.gif)

`GET /runs/{run_id}/corpus-drift` compares frozen chunk/document identities and
exact field digests with current persisted chunks in one read-only transaction.
It reports changed or missing evidence without retrieval, generation, or event
writes. “Unchanged” means matching chunk fields, not whole-paper equality or
scientific validity. The [guide](docs/guides/CORPUS_DRIFT_GUIDE.md) includes bounded
API/Python usage and reproduction of this actual-output offline illustration.

## What actually runs

The API uses a hand-written **Observe -> Decide -> Act** state machine, not
LangGraph. Pydantic validates settings and schemas; HTTPX connects optional live
providers; SQLite stores documents, graph data, and durable run events.

| Stage | Default behavior |
| --- | --- |
| Ingest | `/ingest/text` normalizes text into fixed-size overlapping chunks and indexes entity co-mentions |
| Plan | Keyword-based intent analysis creates bounded retrieval tasks and a rationale trace |
| Retrieve | Deterministic HyDE template expansion, hash-vector cosine retrieval, BM25, and reciprocal rank fusion |
| Expand | Bounded traversal of an entity co-mention graph; paths are retrieval aids, not reasoning proofs |
| Rerank | Lexical overlap via `AdaptiveReranker`, not a learned cross-encoder |
| Generate and check | A routed provider or the fake adapter; claim/source-ID mapping with token-overlap checks |
| Record | SQLite events and frozen evidence for completed-run exports |

`DenseRetriever` uses `HashEmbeddingModel`: deterministic lexical hash vectors,
**not learned semantic embeddings**. Installing optional ML dependencies does not
switch the API to semantic embeddings or cross-encoder reranking. MMR, multi-HyDE,
query rewriting, screening checklists, metadata boosts, paper chat memory, and
most other cataloged helpers require explicit library integration.

For the exact wiring and storage lifecycle, see [Architecture](ARCHITECTURE.md).
For current model IDs, per-provider overrides, routing, and dated official
sources, use the [provider model guide](docs/guides/PROVIDER_MODELS_GUIDE.md).
`/query` requests the reasoning route; the default-provider setting is not a
universal override. Provider availability depends on your account and credentials.

## Boundaries to understand

- **A citation is not verification.** `CitationGrounder` checks token overlap and
  source IDs, not factual correctness or entailment. Review original passages,
  opposing evidence, and warnings even when `grounded` is true.
- **This is not a complete review platform.** Supporting/counter-evidence tasks
  do not constitute a systematic review, novelty proof, or medical decision.
  The offline demo is a plumbing demonstration, not a quality evaluation.
- **Local-first is not production-hardened.** There is no built-in authentication,
  tenant isolation, or PDF-upload endpoint/UI. Keep the API on loopback.
- **Evidence exports contain source text.** Queries, documents, answers, and
  local metadata may be sensitive. Live providers receive retrieved context;
  review permissions and content before transmitting or sharing anything.

Read [Safety](SAFETY.md) before using non-synthetic material.

## Offline evidence extractors

Clinical and statistical cue extractors are **opt-in Python helpers**, not
additional `/query` stages or clinical assessments. Browse the
[categorized cue guides](docs/README.md#clinical-and-statistical-evidence-cues)
for study-design, outcome, statistical, and reporting cues. Each guide retains
its usage examples and illustration; these heuristics do not validate findings.

Existing cue illustrations remain available without interrupting the introduction:

| Opt-in helper | Illustration |
| --- | --- |
| PerProtocolAnalysisCueExtractor | [GIF](docs/assets/per-protocol-analysis-cue-extractor.gif) |
| BayesianInterimPriorCueExtractor | [GIF](docs/assets/bayesian-interim-prior-cue-extractor.gif) |
| DifferenceInDifferencesCueExtractor | [GIF](docs/assets/difference-in-differences-cue-extractor.gif) |
| NegativeControlExposureCueExtractor | [GIF](docs/assets/negative-control-exposure-cue-extractor.gif) |
| EValueSensitivityCueExtractor | [GIF](docs/assets/evalue-sensitivity-cue-extractor.gif) |
| DoseResponseCueExtractor | [GIF](docs/assets/dose-response-cue-extractor.gif) |
| SpilloverInterferenceCueExtractor | [GIF](docs/assets/spillover-interference-cue-extractor.gif) |
| PlaceboTestCueExtractor | [GIF](docs/assets/placebo-test-cue-extractor.gif) |
| HeterogeneousTreatmentEffectCueExtractor | [GIF](docs/assets/heterogeneous-treatment-effect-cue-extractor.gif) |
| RegressionDiscontinuityCueExtractor | [GIF](docs/assets/regression-discontinuity-cue-extractor.gif) |
| InterruptedTimeSeriesCueExtractor | [GIF](docs/assets/interrupted-time-series-cue-extractor.gif) |
| PropensityScoreMatchingCueExtractor | [GIF](docs/assets/propensity-score-matching-cue-extractor.gif) |
| SyntheticControlCueExtractor | [GIF](docs/assets/synthetic-control-cue-extractor.gif) |
| NegativeControlOutcomeCueExtractor | [GIF](docs/assets/negative-control-outcome-cue-extractor.gif) |
| MendelianRandomizationCueExtractor | [GIF](docs/assets/mendelian-randomization-cue-extractor.gif) |
| InstrumentalVariableStrengthCueExtractor | [GIF](docs/assets/instrumental-variable-strength-cue-extractor.gif) |
| TimeVaryingConfoundingCueExtractor | [GIF](docs/assets/time-varying-confounding-cue-extractor.gif) |
| MediationAnalysisCueExtractor | [GIF](docs/assets/mediation-analysis-cue-extractor.gif) |
| CompetingRiskCueExtractor | [GIF](docs/assets/competing-risk-cue-extractor.gif) |
| TransportabilityCueExtractor | [GIF](docs/assets/transportability-cue-extractor.gif) |
| ConfoundingAdjustmentCueExtractor | [GIF](docs/assets/confounding-adjustment-cue-extractor.gif) |
| MissingDataMechanismCueExtractor | [GIF](docs/assets/missing-data-mechanism-cue-extractor.gif) |
| EstimandIchE9CueExtractor | [GIF](docs/assets/estimand-ich-e9-cue-extractor.gif) |

## Documentation

| Guide | Purpose |
| --- | --- |
| [Quickstart](QUICKSTART.md) | Working offline setup and first API request |
| [Research workflow and portfolio](docs/guides/RESEARCH_WORKFLOW_GUIDE.md) | Corpus, questions, evidence review, export, and acceptance checklist |
| [API and library examples](docs/EXAMPLES.md) | Copyable requests and clearly separated network adapters |
| [Configuration](CONFIGURATION.md) | Runtime settings and provider setup |
| [Architecture](ARCHITECTURE.md) | Integrated pipeline and persistence |
| [Documentation catalog](docs/README.md) | All existing extension/source guides, operations, and historical records |
| [Contributing](CONTRIBUTING.md) / [Security](SECURITY.md) | Contribution workflow and vulnerability reporting |

## Quality gates

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src/
uv run pytest tests/ -v --cov=src --cov-fail-under=70
```

[CI](.github/workflows/ci.yml) runs on Python 3.11 and 3.12.
[Security Scan](.github/workflows/security.yml) also audits dependencies and runs
Bandit. Passing these checks is not a scientific-accuracy benchmark.

## License

[Apache-2.0](LICENSE). Source papers and third-party service content retain their
own licenses and access terms.
