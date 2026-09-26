"""Bounded per-question, per-paper retrieval inspection without generation or persistence."""

import asyncio
import json
from typing import Annotated, Literal, Self
from urllib.parse import quote

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from agent.evidence import SHA256, EvidenceSource, RunConfiguration, text_digest
from agent.markdown import literal_block
from agent.retrieval_preview import RetrievalPreview, RetrievalPreviewError
from agent.runner import AgentRunner
from retrieval.scope import DocumentIds
from storage.document_catalog import MAX_SOURCE_CHARACTERS, MAX_TITLE_CHARACTERS, Identity
from storage.document_chunks import ChunkIdentity
from storage.paper_collections import CollectionError, CollectionId, SQLitePaperCollections

MAX_QUESTIONS = 5
MAX_QUESTION_CHARACTERS = 500
MAX_WORKSHEET_DOCUMENTS = 10
MAX_CELLS = 50
MAX_PASSAGES_PER_CELL = 3
MAX_EXCERPT_CHARACTERS = 800
MAX_RESPONSE_BYTES = 262144
WORKSHEET_TIMEOUT_SECONDS = 30

_WARNINGS = (
    "Passages are retrieval results, not answers, support/conflict judgments, "
    "or scientific findings. "
    "Scores (including zero scores and hash-vector matches) do not establish support.",
    "no_passages means this preview returned no passages, not an absence of scientific evidence "
    "or a judgment of answerability.",
    "Only selection membership is frozen. Corpus contents and in-memory indexes can change; "
    "this worksheet is not a corpus snapshot. Inspection URLs read current stored chunks.",
    "Review before sharing: questions, document identifiers, source labels, and excerpts may be "
    "sensitive. The local API has no authentication or tenant isolation.",
)
Question = Annotated[
    str, StringConstraints(strict=True, min_length=1, max_length=MAX_QUESTION_CHARACTERS)
]
_PathPart = Annotated[str, StringConstraints(strict=True, max_length=128)]


class WorksheetRequest(BaseModel):
    """Small, immutable selection; omission never means the whole corpus."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    questions: tuple[Question, ...] = Field(min_length=1, max_length=MAX_QUESTIONS)
    document_ids: DocumentIds | SkipJsonSchema[None] = Field(
        default=None,
        description="1-10 supplied IDs; trim outer whitespace and keep first-seen order.",
        json_schema_extra={"maxItems": MAX_WORKSHEET_DOCUMENTS},
    )
    collection_id: CollectionId | SkipJsonSchema[None] = Field(
        default=None, description="Resolve this saved collection once, before asynchronous work."
    )
    passages_per_cell: int = Field(default=2, strict=True, ge=1, le=MAX_PASSAGES_PER_CELL)

    @field_validator("questions", mode="before")
    @classmethod
    def ordered_questions(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            raise ValueError("questions must be a list or tuple of strings.")
        return value

    @field_validator("questions")
    @classmethod
    def readable_questions(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if not value.strip():
                raise ValueError("Questions cannot be blank.")
            value.encode("utf-8")
        return values

    @field_validator("document_ids", "collection_id", mode="before")
    @classmethod
    def explicit_selection(cls, value: object) -> object:
        if value is None:
            raise ValueError("Selection cannot be null.")
        if isinstance(value, (list, tuple)) and len(value) > MAX_WORKSHEET_DOCUMENTS:
            raise ValueError("A worksheet accepts at most 10 supplied document IDs.")
        return value

    @model_validator(mode="after")
    def one_selection(self) -> Self:
        if (self.document_ids is None) == (self.collection_id is None):
            raise ValueError("Supply exactly one of document_ids or collection_id.")
        for identifier in self.document_ids or ():
            identifier.encode("utf-8")
        return self


class WorksheetError(RuntimeError):
    """Safe all-or-nothing failure; the cause retains diagnostics for trusted Python callers."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 500,
        *,
        question_index: int | None = None,
        document_index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.question_index = question_index
        self.document_index = document_index


class _WorksheetModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class WorksheetLimits(_WorksheetModel):
    """Fixed service bounds, independent of provider settings."""

    max_questions: Literal[5] = 5
    max_question_characters: Literal[500] = 500
    max_documents: Literal[10] = 10
    max_cells: Literal[50] = 50
    max_passages_per_cell: Literal[3] = 3
    max_excerpt_characters: Literal[800] = 800
    max_serialized_bytes: Literal[262144] = 262144
    overall_timeout_seconds: Literal[30] = 30


class WorksheetDocument(_WorksheetModel):
    document_id: Identity
    inspection_url: str | None = Field(max_length=1600)


class WorksheetPassage(_WorksheetModel):
    """Exact preview provenance with explicitly bounded display-only text prefixes."""

    document_id: Identity
    chunk_id: ChunkIdentity
    rank: int = Field(ge=1, le=50)
    score: float
    retriever: str = Field(max_length=128)
    path: tuple[_PathPart, ...] = Field(max_length=16)
    title: str = Field(max_length=MAX_TITLE_CHARACTERS)
    title_truncated: bool
    source: str = Field(max_length=MAX_SOURCE_CHARACTERS)
    source_truncated: bool
    excerpt: str = Field(max_length=MAX_EXCERPT_CHARACTERS)
    excerpt_truncated: bool
    text_characters: int = Field(ge=0)
    text_sha256: SHA256

    @classmethod
    def from_source(cls, source: EvidenceSource) -> Self:
        chunk = source.chunk
        return cls(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            rank=source.rank,
            score=source.score,
            retriever=source.retriever,
            path=tuple(source.path),
            title=chunk.title[:MAX_TITLE_CHARACTERS],
            title_truncated=len(chunk.title) > MAX_TITLE_CHARACTERS,
            source=chunk.source[:MAX_SOURCE_CHARACTERS],
            source_truncated=len(chunk.source) > MAX_SOURCE_CHARACTERS,
            excerpt=chunk.text[:MAX_EXCERPT_CHARACTERS],
            excerpt_truncated=len(chunk.text) > MAX_EXCERPT_CHARACTERS,
            text_characters=len(chunk.text),
            text_sha256=source.text_sha256,
        )


class WorksheetCell(_WorksheetModel):
    document_id: Identity
    status: Literal["passages_returned", "no_passages"]
    preview_source_count: int = Field(ge=0, le=50)
    passages_omitted: int = Field(ge=0, le=50)
    configuration: RunConfiguration
    passages: tuple[WorksheetPassage, ...] = Field(max_length=MAX_PASSAGES_PER_CELL)


class WorksheetRow(_WorksheetModel):
    question: Question
    cells: tuple[WorksheetCell, ...] = Field(min_length=1, max_length=MAX_WORKSHEET_DOCUMENTS)


def _bounded(content: str) -> str:
    if len(content.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise WorksheetError(
            "worksheet_too_large",
            "Worksheet exceeds the serialized download limit; reduce questions, papers, "
            "or passages_per_cell. No partial worksheet was returned.",
            413,
        )
    return content


class ResearchWorksheet(_WorksheetModel):
    """Portable inspection output, not a saved run, answer, or frozen corpus."""

    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["retrieval_passages"] = "retrieval_passages"
    collection_id: CollectionId | None
    passages_per_cell: int = Field(ge=1, le=MAX_PASSAGES_PER_CELL)
    documents: tuple[WorksheetDocument, ...] = Field(
        min_length=1, max_length=MAX_WORKSHEET_DOCUMENTS
    )
    rows: tuple[WorksheetRow, ...] = Field(min_length=1, max_length=MAX_QUESTIONS)
    limits: WorksheetLimits = Field(default_factory=WorksheetLimits)
    warnings: tuple[str, ...] = _WARNINGS

    def to_json(self) -> str:
        """Serialize within the same UTF-8 byte bound as the HTTP download."""
        return _bounded(
            json.dumps(
                self.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )

    def to_markdown(self) -> str:
        """Use static table labels and literal blocks for every untrusted string."""
        headings = ["Question", *(f"Paper {i}" for i in range(1, len(self.documents) + 1))]
        sections = [
            "# Research evidence worksheet\n",
            "Retrieval passages for human inspection; not generated answers or findings.\n",
            "## Interpretation and privacy\n",
            literal_block("\n".join(self.warnings)),
            "## Side-by-side passage counts\n",
            "| " + " | ".join(headings) + " |",
            "| " + " | ".join("---" for _ in headings) + " |",
        ]
        for number, row in enumerate(self.rows, 1):
            counts = [f"{cell.status}: {len(cell.passages)}" for cell in row.cells]
            sections.append("| " + " | ".join([f"Question {number}", *counts]) + " |")
        sections.append("\n## Selected papers\n")
        for number, document in enumerate(self.documents, 1):
            sections.extend(
                [f"### Paper {number}\n", literal_block(document.model_dump_json(indent=2), "json")]
            )
        sections.extend(
            [
                "## Selection and limits\n",
                literal_block(
                    f"collection_id: {self.collection_id}\n"
                    f"passages_per_cell: {self.passages_per_cell}"
                ),
                literal_block(self.limits.model_dump_json(indent=2), "json"),
            ]
        )
        for question_number, row in enumerate(self.rows, 1):
            sections.extend([f"## Question {question_number}\n", literal_block(row.question)])
            for paper_number, cell in enumerate(row.cells, 1):
                sections.extend(
                    [
                        f"### Paper {paper_number}\n",
                        literal_block(cell.model_dump_json(exclude={"passages"}, indent=2), "json"),
                    ]
                )
                for passage in cell.passages:
                    sections.extend(
                        [
                            f"#### Passage {passage.rank}\n",
                            literal_block(
                                passage.model_dump_json(exclude={"excerpt"}, indent=2), "json"
                            ),
                            literal_block(passage.excerpt),
                        ]
                    )
        return _bounded("\n".join(sections))


def _cell(
    raw_preview: RetrievalPreview, question: str, document_id: str, passages_per_cell: int
) -> WorksheetCell:
    preview = RetrievalPreview.model_validate(raw_preview.model_dump())
    sources = preview.sources
    if (
        preview.plan.observation.document_ids != (document_id,)
        or preview.plan.observation.original_query != question.strip()
        or len(sources) > preview.configuration.max_source_docs
        or [source.rank for source in sources] != list(range(1, len(sources) + 1))
        or len({source.chunk.chunk_id for source in sources}) != len(sources)
        or any(
            source.chunk.document_id != document_id
            or text_digest(source.chunk.text) != source.text_sha256
            for source in sources
        )
        or preview.context
        != "\n".join(
            f"[{source.chunk.chunk_id}] {source.chunk.title}: {source.chunk.text}"
            for source in sources
        )
        or text_digest(preview.context) != preview.context_sha256
    ):
        raise WorksheetError(
            "invalid_worksheet_evidence",
            "Preview provenance is inconsistent; no partial worksheet was returned.",
        )
    passages = tuple(WorksheetPassage.from_source(source) for source in sources[:passages_per_cell])
    return WorksheetCell(
        document_id=document_id,
        status="passages_returned" if passages else "no_passages",
        preview_source_count=len(sources),
        passages_omitted=len(sources) - len(passages),
        configuration=preview.configuration,
        passages=passages,
    )


class ResearchWorksheetService:
    """Compose the existing preview path; never alter a shared runner's configuration."""

    def __init__(self, runner: AgentRunner, collections: SQLitePaperCollections) -> None:
        self._runner = runner
        self._collections = collections

    async def build(self, request: WorksheetRequest) -> ResearchWorksheet:
        """Freeze the complete selection before awaiting; abort on any cell failure."""
        request = WorksheetRequest.model_validate(request.model_dump(exclude_none=True))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + WORKSHEET_TIMEOUT_SECONDS
        question_index: int | None = None
        document_index: int | None = None

        def check_deadline() -> None:
            if loop.time() >= deadline:
                raise TimeoutError("Worksheet deadline exceeded.")

        try:
            async with asyncio.timeout_at(deadline):
                if request.collection_id is not None:
                    document_ids = self._collections.resolve(request.collection_id)
                else:
                    assert request.document_ids is not None
                    document_ids = self._collections.validate_document_ids(request.document_ids)
                if (
                    len(document_ids) > MAX_WORKSHEET_DOCUMENTS
                    or len(document_ids) * len(request.questions) > MAX_CELLS
                ):
                    raise WorksheetError(
                        "worksheet_selection_too_large",
                        "A worksheet allows at most 10 papers and 50 question/paper cells; "
                        "choose a smaller selection.",
                        422,
                    )
                check_deadline()
                documents = tuple(
                    WorksheetDocument(
                        document_id=identifier,
                        inspection_url=(
                            None
                            if any(part in {".", ".."} for part in identifier.split("/"))
                            else f"/documents/{quote(identifier, safe='')}/chunks"
                        ),
                    )
                    for identifier in document_ids
                )
                rows = []
                for row_index, question in enumerate(request.questions):
                    question_index = row_index
                    cells = []
                    for column_index, document_id in enumerate(document_ids):
                        document_index = column_index
                        check_deadline()
                        preview = await self._runner.preview(question, document_ids=(document_id,))
                        cells.append(
                            _cell(preview, question, document_id, request.passages_per_cell)
                        )
                    rows.append(WorksheetRow(question=question, cells=tuple(cells)))
                # Recheck the frozen IDs, not mutable collection membership or contents.
                try:
                    self._collections.validate_document_ids(document_ids)
                except CollectionError as exc:
                    if exc.code != "unknown_documents":
                        raise
                    raise WorksheetError(
                        "worksheet_documents_changed",
                        "Selected documents disappeared during inspection; retry with an "
                        "existing selection. No partial worksheet was returned.",
                        409,
                    ) from exc
                result = ResearchWorksheet(
                    collection_id=request.collection_id,
                    passages_per_cell=request.passages_per_cell,
                    documents=documents,
                    rows=tuple(rows),
                )
                result.to_json()
                result.to_markdown()
                check_deadline()
                return result
        except (WorksheetError, CollectionError):
            raise
        except RetrievalPreviewError as exc:
            raise WorksheetError(
                exc.code,
                str(exc),
                exc.status_code,
                question_index=question_index,
                document_index=document_index,
            ) from exc
        except TimeoutError as exc:
            raise WorksheetError(
                "worksheet_timeout",
                "Worksheet exceeded its overall deadline; no partial worksheet was returned.",
                504,
                question_index=question_index,
                document_index=document_index,
            ) from exc
        except Exception as exc:
            raise WorksheetError(
                "worksheet_failed",
                "Worksheet inspection failed; no partial worksheet was returned.",
                question_index=question_index,
                document_index=document_index,
            ) from exc
