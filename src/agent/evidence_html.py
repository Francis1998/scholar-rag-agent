"""A bounded, script-free reader for an EvidenceExporter-validated frozen bundle."""

import base64
import hashlib
import json
from collections.abc import Iterator
from html import escape

from pydantic import BaseModel, JsonValue

from agent.evidence import EvidenceBundle
from storage.evidence_export import EvidenceExportError

MAX_HTML_BYTES = 4_194_304

_STYLE = """
:root { color-scheme: light; font-family: system-ui, sans-serif; line-height: 1.55; }
body { margin: 0 auto; max-width: 76rem; padding: 1.5rem; color: #172337; background: #fff; }
h1, h2, h3, h4, summary, dt { line-height: 1.3; }
h2 { border-bottom: 2px solid #d5dee9; padding-bottom: .4rem; margin-top: 2rem; }
nav { display: flex; flex-wrap: wrap; gap: .6rem 1.2rem; margin: 1rem 0; }
a { color: #164f91; text-decoration: underline; text-underline-offset: .15em; }
a:focus-visible, summary:focus-visible { outline: 3px solid #164f91; outline-offset: 3px; }
.skip-link { position: absolute; top: -4rem; left: 1rem; padding: .5rem; background: #fff; }
.skip-link:focus { top: .5rem; z-index: 1; }
section, article, details { scroll-margin-top: 1rem; }
article { border: 1px solid #c5d2e1; border-radius: .5rem; padding: 1rem; margin: 1rem 0; }
dl { display: grid; grid-template-columns: minmax(9rem, 1fr) minmax(0, 4fr); gap: .4rem 1rem; }
dt { font-weight: 650; padding-top: .35rem; }
dd { margin: 0; min-width: 0; }
pre { margin: .35rem 0 1rem; padding: .7rem; background: #f3f6fa; border-radius: .3rem; }
.literal { white-space: pre-wrap; overflow-wrap: anywhere; unicode-bidi: plaintext; }
code { font: inherit; }
.passage { border-left: 4px solid #376890; }
.unresolved { color: #8b251c; font-weight: 650; }
details { border: 1px solid #c5d2e1; padding: .8rem; margin: 1rem 0; }
summary { cursor: pointer; font-weight: 650; }
footer { border-top: 1px solid #c5d2e1; margin-top: 2rem; padding-top: 1rem; }
@media (max-width: 40rem) { body { padding: .8rem; } dl { display: block; } }
@media print {
  body { max-width: none; padding: 0; color: #000; }
  nav, .skip-link { display: none; }
  a { color: #000; }
  pre { background: none; }
  h2, h3, h4, summary { break-after: avoid; }
  article, details { border-color: #777; }
  details::details-content { content-visibility: visible; }
}
"""
_STYLE_HASH = base64.b64encode(hashlib.sha256(_STYLE.encode("utf-8")).digest()).decode("ascii")
HTML_CSP = (
    f"default-src 'none'; style-src 'sha256-{_STYLE_HASH}'; base-uri 'none'; form-action 'none'"
)


def _unrepresentable_text() -> EvidenceExportError:
    return EvidenceExportError(
        "html_text_not_representable",
        "Saved text cannot be represented faithfully as UTF-8 HTML. Inspect the saved record.",
    )


def _literal(text: str, css_class: str = "") -> str:
    if "\0" in text:
        raise _unrepresentable_text()
    # HTML normalizes raw CRs and drops an initial newline directly inside <pre>.
    literal = escape(text).replace("\r", "&#13;")
    return f'<pre class="literal {css_class}" dir="auto"><code>{literal}</code></pre>'


def _json(value: BaseModel | JsonValue, css_class: str = "") -> str:
    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return _literal(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2), css_class)


def _field(label: str, value: str) -> str:
    return f"<dt>{label}</dt><dd>{_literal(value)}</dd>"


def _references(chunk_ids: list[str], ranks: dict[str, int], css_class: str) -> Iterator[str]:
    yield f'<ul class="{css_class}">'
    for chunk_id in chunk_ids:
        rank = ranks.get(chunk_id)
        yield "<li>"
        if rank is None:
            yield '<strong class="unresolved">Unresolved reference</strong>'
        else:
            yield f'<a href="#source-{rank}">Evidence {rank}</a>'
        yield _literal(chunk_id)
        yield "</li>"
    yield "</ul>"
    if not chunk_ids:
        yield "<p>No references recorded in this category.</p>"


def _warnings(warnings: list[str], empty_message: str) -> Iterator[str]:
    if warnings:
        yield "<ul>"
        for warning in warnings:
            yield "<li>" + _literal(warning) + "</li>"
        yield "</ul>"
    else:
        yield f"<p>{empty_message}</p>"


def _fragments(bundle: EvidenceBundle) -> Iterator[str]:
    yield (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<meta http-equiv="Content-Security-Policy" content="{escape(HTML_CSP)}">'
        "<title>Offline evidence reader</title>"
        f"<style>{_STYLE}</style></head><body>"
        '<a class="skip-link" href="#main">Skip to main content</a>'
        "<header><h1>Offline evidence reader</h1>"
        "<p>A completed saved run, not scientific validation or a live research application.</p>"
        '<nav aria-label="Reader sections">'
        '<a href="#summary">Summary</a><a href="#question">Question</a>'
        '<a href="#answer">Answer</a><a href="#claims">Claims</a>'
        '<a href="#citations">Citations</a><a href="#sources">Frozen passages</a>'
        '<a href="#configuration">Configuration</a><a href="#events">Recorded trace</a>'
        '</nav></header><main id="main" tabindex="-1">'
    )
    missing_ids = {
        chunk_id for link in bundle.claim_evidence for chunk_id in link.missing_chunk_ids
    } | {link.chunk_id for link in bundle.citation_evidence if link.evidence_rank is None}
    yield '<section id="summary"><h2>Saved run summary</h2><dl>'
    for label, value in (
        ("Run ID", bundle.run_id),
        ("Agent ID", bundle.agent_id),
        ("Recorded status", bundle.status),
        ("Completed (UTC)", bundle.completed_at),
        ("Bundle schema", bundle.schema_version),
        ("Frozen passages", str(len(bundle.snapshot.sources))),
        (
            "Claims / final citations",
            f"{len(bundle.answer.claims)} / {len(bundle.answer.citations)}",
        ),
        ("Unresolved distinct reference IDs", str(len(missing_ids))),
        ("Recorded provider", bundle.generation.provider),
        (
            "Recorded model",
            "Not recorded"
            if bundle.generation.model_name is None
            else bundle.generation.model_name,
        ),
        ("Recorded task type", bundle.generation.task_type.value),
        ("Context SHA-256", bundle.snapshot.context_sha256),
    ):
        yield _field(label, value)
    yield "</dl><h3>Review before sharing</h3>"
    yield from _warnings(bundle.warnings, "No bundle warnings were recorded.")
    yield (
        "<p>Text is literal, not rendered Markdown. Unicode and bidirectional controls are "
        "preserved inside isolated text blocks; visual order alone is not an identity check. "
        "Hashes are content fingerprints, not signatures.</p></section>"
        '<section id="question"><h2>Original question</h2>'
    )
    yield _literal(bundle.query)
    yield '</section><section id="answer"><h2>Recorded answer</h2>'
    yield _literal(bundle.answer.answer, "answer-text")
    yield _literal(f"Recorded ungrounded flag: {str(bundle.answer.ungrounded).lower()}")
    yield "<h3>Answer warnings</h3>"
    yield from _warnings(bundle.answer.warnings, "No answer warnings were recorded.")
    yield (
        '</section><section id="claims"><h2>Claims and evidence associations</h2>'
        "<p>Proposed references were suggested during generation. Accepted references passed "
        "the recorded grounding step, which uses token overlap, not semantic entailment. "
        "A link only locates a passage; it does not establish support for a claim.</p>"
    )
    ranks = {source.chunk.chunk_id: source.rank for source in bundle.snapshot.sources}
    for number, (claim, claim_link) in enumerate(
        zip(bundle.answer.claims, bundle.claim_evidence, strict=True), start=1
    ):
        yield f'<article id="claim-{number}"><h3>Claim {number}</h3>'
        yield _literal(claim.text, "claim-text")
        yield _literal(f"Recorded grounded flag: {str(claim.grounded).lower()}", "grounded-flag")
        yield "<h4>Proposed references</h4>"
        yield from _references(claim_link.proposed_chunk_ids, ranks, "proposed")
        yield "<h4>Accepted by recorded grounding</h4>"
        yield from _references(claim_link.grounded_chunk_ids, ranks, "accepted")
        yield "<details><summary>Recorded claim association</summary>"
        yield _json(claim_link)
        yield "</details></article>"
    if not bundle.answer.claims:
        yield "<p>No claims were recorded.</p>"
    yield '</section><section id="citations"><h2>Final recorded citations</h2>'
    for number, (citation, citation_link) in enumerate(
        zip(bundle.answer.citations, bundle.citation_evidence, strict=True), start=1
    ):
        yield f'<article id="citation-{number}"><h3>Citation {number}</h3>'
        yield from _references([citation_link.chunk_id], ranks, "citation-reference")
        yield _json(citation, "citation-record")
        yield "<details><summary>Recorded citation association</summary>"
        yield _json(citation_link)
        yield "</details></article>"
    if not bundle.answer.citations:
        yield "<p>No final citations were recorded.</p>"
    yield (
        '</section><section id="sources"><h2>Exact frozen source passages</h2>'
        "<p>Final-context rank is retrieval order, not scientific strength. Titles, paths, IDs, "
        "URLs and metadata below are recorded text, never active links. These are captured "
        "chunks, not necessarily complete papers.</p>"
    )
    for source in bundle.snapshot.sources:
        yield f'<article id="source-{source.rank}"><h3>Evidence {source.rank}</h3>'
        yield _literal(source.chunk.text, "passage")
        metadata = source.model_dump(mode="json")
        metadata["chunk"].pop("text")
        yield "<h4>Recorded identity, rank, score, path, digest and metadata</h4>"
        yield _json(metadata, "source-metadata")
        yield "</article>"
    if not bundle.snapshot.sources:
        yield "<p>The saved model context contained no source passages.</p>"
    yield (
        '</section><section id="configuration"><h2>Recorded configuration and policy</h2>'
        "<p>These are generation-time records, not today's settings. A null scope means "
        "unscoped retrieval; a null policy means none was requested. Counts, quotas and "
        "lexical similarity do not establish evidence independence or scientific validity.</p>"
        '<section id="document-scope"><h3>Recorded document scope</h3>'
    )
    scope = bundle.plan.observation.document_ids
    yield _json(None if scope is None else list(scope))
    yield '</section><section id="evidence-policy"><h3>Recorded evidence policy</h3>'
    yield _json(bundle.plan.observation.evidence_policy)
    yield '</section><section id="runtime-configuration"><h3>Effective runtime limits</h3>'
    yield _json(bundle.configuration)
    yield "</section><h3>Capture format and limits</h3>"
    yield _literal(bundle.snapshot.context_format)
    yield _json(bundle.snapshot.limits)
    yield "</section>"
    for identifier, title, record in (
        ("plan", "Recorded plan and operational rationale", bundle.plan),
        ("request", "Exact recorded generation request", bundle.snapshot.request),
        (
            "generation",
            "Recorded provider/model identity and proposed references",
            bundle.generation,
        ),
    ):
        yield f'<details id="{identifier}"><summary>{title}</summary>'
        yield _json(record)
        yield "</details>"
    yield (
        '<details id="events"><summary>Ordered operational events</summary>'
        '<p>Snapshot payload references point to <a href="#sources">frozen passages</a> and '
        'the <a href="#request">exact request</a>; generation references point to '
        '<a href="#generation">recorded generation provenance</a>. Unknown event payloads '
        "remain explicitly omitted by the exporter. Later events are not part of this "
        "completed bundle.</p>"
    )
    for event in bundle.events:
        yield _json(event, "event-record")
    yield (
        "</details></main><footer><p>Use your browser's Find and Print commands. Expand details "
        "before printing if your browser keeps them closed in print preview.</p>"
        "<p>This file makes no network requests and has no scripts, forms, editable annotations "
        "or retrieval/generation controls. It does not provide authentication, signed evidence, "
        "automatic redaction or scientific validation. Review sensitive content before sharing."
        "</p></footer></body></html>\n"
    )


def render_html(bundle: EvidenceBundle) -> str:
    """Render a validated saved bundle, checking the full escaped UTF-8 size before delivery."""
    fragments: list[str] = []
    size = 0
    for fragment in _fragments(bundle):
        try:
            size += len(fragment.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise _unrepresentable_text() from exc
        if size > MAX_HTML_BYTES:
            raise EvidenceExportError(
                "html_export_too_large",
                f"HTML export exceeds {MAX_HTML_BYTES} UTF-8 bytes; no content was truncated. "
                "Use JSON or Markdown for this saved run.",
                413,
            )
        fragments.append(fragment)
    return "".join(fragments)
