"""Reference-library import API: multipart or JSON preview, explicit commit, history."""

import logging
import re

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.concurrency import run_in_threadpool

from api.collections import COLLECTION_HEADERS, _CollectionRoute, collection_errors
from api.dependencies import AppContainer
from api.native_forms import FormTooLargeError, read_body
from ingestion.reference_files import MAX_FILE_BYTES, MAX_FILE_NAME_CHARACTERS, FormatChoice
from ingestion.reference_import import (
    MAX_FILES,
    DraftId,
    ImportTarget,
    ReferenceImportHistory,
    ReferenceImportPreview,
    ReferenceImportResult,
    UploadedFile,
)
from storage.paper_collections import CollectionError, CollectionId, CollectionName, Revision

logger = logging.getLogger(__name__)
MAX_MULTIPART_BYTES = MAX_FILES * MAX_FILE_BYTES + 65_536
MAX_JSON_BYTES = 16_777_216
MAX_TEXT_FIELD_BYTES = 4_096
_TEXT_FIELDS = frozenset({"collection_id", "collection_revision", "new_collection_name", "format"})
_FORMATS: dict[str, FormatChoice] = {
    "auto": "auto",
    "ris": "ris",
    "medline": "medline",
    "csl_json": "csl_json",
    "csv": "csv",
    "tsv": "tsv",
}
_JSON = re.compile(r"application/json(?:\s*;\s*charset=(?:utf-8|\"utf-8\"))?")


class ImportFileContent(BaseModel):
    """One file sent as JSON text; its SHA-256 is computed over the UTF-8 encoding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_name: str = Field(min_length=1, max_length=MAX_FILE_NAME_CHARACTERS)
    content: str = Field(max_length=MAX_FILE_BYTES)
    format: FormatChoice = "auto"


class ImportPreviewRequest(BaseModel):
    """Files plus exactly one target: an existing collection revision or a new name."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    files: list[ImportFileContent] = Field(min_length=1, max_length=MAX_FILES)
    collection_id: CollectionId | None = None
    collection_revision: Revision | None = None
    new_collection_name: CollectionName | None = None


class ImportCommitRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    draft_id: DraftId


def _invalid(message: str) -> CollectionError:
    return CollectionError("invalid_reference_import", message, 422)


def _target(fields: dict[str, object]) -> ImportTarget:
    try:
        return ImportTarget.model_validate(
            {key: value for key, value in fields.items() if value not in (None, "")}
        )
    except ValidationError as exc:
        raise _invalid(
            "Choose collection_id with collection_revision (an existing collection), or "
            "new_collection_name (1-120 characters, no control characters)."
        ) from exc


def _json_preview(body: bytes) -> tuple[list[UploadedFile], ImportTarget]:
    try:
        payload = ImportPreviewRequest.model_validate_json(body)
        files = [
            UploadedFile(item.file_name, item.content.encode("utf-8"), item.format)
            for item in payload.files
        ]
    except (ValidationError, UnicodeEncodeError) as exc:
        raise _invalid(
            f"Send JSON with 1-{MAX_FILES} files (file_name, content text, optional format "
            "auto/ris/medline/csl_json/csv/tsv) and one target. Unknown fields are rejected."
        ) from exc
    fields = payload.model_dump(exclude={"files"})
    return files, _target(fields)


def _parts(body: bytes, boundary: bytes) -> list[tuple[dict[bytes, bytes], bytes]]:
    """Parse an already bounded body in memory; nothing is spooled to disk."""
    parts: list[tuple[dict[bytes, bytes], bytearray]] = []
    name, value = bytearray(), bytearray()

    def header_end() -> None:
        parts[-1][0][bytes(name).lower()] = bytes(value)
        name.clear()
        value.clear()

    parser = MultipartParser(
        boundary,
        {
            "on_part_begin": lambda: parts.append(({}, bytearray())),
            "on_part_data": lambda data, start, end: parts[-1][1].extend(data[start:end]),
            "on_header_field": lambda data, start, end: name.extend(data[start:end]),
            "on_header_value": lambda data, start, end: value.extend(data[start:end]),
            "on_header_end": header_end,
        },
    )
    parser.write(body)
    parser.finalize()
    return [(headers, bytes(content)) for headers, content in parts]


def _multipart_preview(body: bytes, boundary: bytes) -> tuple[list[UploadedFile], ImportTarget]:
    try:
        parts = _parts(body, boundary)
    except ValueError as exc:
        raise _invalid("The multipart body is malformed.") from exc
    uploads: list[tuple[str, bytes]] = []
    fields: dict[str, str] = {}
    for headers, content in parts:
        disposition, options = parse_options_header(headers.get(b"content-disposition", b""))
        name = options.get(b"name", b"").decode("utf-8", "replace")
        if disposition != b"form-data":
            raise _invalid("Every part needs form-data disposition.")
        if name == "files" and b"filename" in options:
            try:
                file_name = options[b"filename"].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise _invalid("File names must be UTF-8.") from exc
            file_name = file_name.replace("\\", "/").rsplit("/", 1)[-1]
            if not file_name and not content:
                continue
            uploads.append((file_name, content))
        elif name in _TEXT_FIELDS and name not in fields and len(content) <= MAX_TEXT_FIELD_BYTES:
            try:
                fields[name] = content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise _invalid("Form fields must be UTF-8.") from exc
        else:
            raise _invalid(
                "Use files plus optional format, collection_id, collection_revision and "
                "new_collection_name fields, each at most once."
            )
    choice = _FORMATS.get(fields.pop("format", "") or "auto")
    if choice is None:
        raise _invalid("format must be auto, ris, medline, csl_json, csv or tsv.")
    if not 1 <= len(uploads) <= MAX_FILES:
        raise _invalid(f"Choose 1-{MAX_FILES} reference files.")
    revision = fields.get("collection_revision", "")
    target_fields: dict[str, object] = dict(fields)
    if revision:
        if not (revision.isascii() and revision.isdecimal() and len(revision) <= 19):
            raise _invalid("collection_revision must be a positive integer.")
        target_fields["collection_revision"] = int(revision)
    files = [UploadedFile(name, content, choice) for name, content in uploads]
    return files, _target(target_fields)


async def read_preview_request(request: Request) -> tuple[list[UploadedFile], ImportTarget]:
    """Read a bounded JSON or multipart body; raise CollectionError on any invalid input."""
    content_types = request.headers.getlist("content-type")
    if len(content_types) != 1 or "content-encoding" in request.headers:
        raise CollectionError(
            "reference_import_media_type",
            "Send multipart/form-data or application/json without content encoding.",
            415,
        )
    media, options = parse_options_header(content_types[0])
    if media == b"multipart/form-data" and options.get(b"boundary"):
        limit, multipart = MAX_MULTIPART_BYTES, True
    elif _JSON.fullmatch(content_types[0].strip().lower()):
        limit, multipart = MAX_JSON_BYTES, False
    else:
        raise CollectionError(
            "reference_import_media_type",
            "Send multipart/form-data or application/json without content encoding.",
            415,
        )
    try:
        body = await read_body(request, max_bytes=limit)
    except FormTooLargeError as exc:
        raise CollectionError(
            "reference_import_too_large",
            f"The request exceeds {limit} bytes; nothing was read or stored.",
            413,
        ) from exc
    except ValueError as exc:
        raise _invalid("The request body length is invalid.") from exc
    if multipart:
        return _multipart_preview(body, options[b"boundary"])
    return _json_preview(body)


router = APIRouter(prefix="/collections", tags=["reference import"], route_class=_CollectionRoute)
_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"description": "Unknown collection or unknown/expired preview draft"},
    409: {"description": "Stale revision, name conflict, nothing new, or already committed"},
    413: {"description": "Request, file or record count exceeds a fixed bound"},
    415: {"description": "Body is neither multipart/form-data nor application/json"},
    422: {"description": "Invalid target, file, format, encoding or collection capacity"},
    503: {"description": "Import, collection or corpus storage unavailable"},
}


@router.post(
    "/imports/preview",
    response_model=ReferenceImportPreview,
    responses=_RESPONSES,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": ImportPreviewRequest.model_json_schema()},
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                            },
                            "format": {"type": "string"},
                            "collection_id": {"type": "string"},
                            "collection_revision": {"type": "integer"},
                            "new_collection_name": {"type": "string"},
                        },
                        "required": ["files"],
                    }
                },
            },
        }
    },
)
async def preview_import(request: Request, response: Response) -> ReferenceImportPreview:
    """Dry run: parse, deduplicate and report; stores only a private 24-hour draft."""
    container: AppContainer = request.app.state.container
    with collection_errors():
        files, target = await read_preview_request(request)
        preview = await run_in_threadpool(container.reference_imports.preview, files, target)
    response.headers.update(COLLECTION_HEADERS)
    return preview


@router.post(
    "/imports", response_model=ReferenceImportResult, status_code=201, responses=_RESPONSES
)
def commit_import(
    request: Request, payload: ImportCommitRequest, response: Response
) -> ReferenceImportResult:
    """Add a previewed draft's new papers; never deletes or overwrites papers or members."""
    container: AppContainer = request.app.state.container
    with collection_errors():
        result = container.reference_imports.commit(payload.draft_id)
    response.headers.update(COLLECTION_HEADERS)
    root_path = request.scope.get("root_path", "").rstrip("/")
    response.headers["Location"] = f"{root_path}/collections/{result.collection.collection_id}"
    return result


@router.get(
    "/{collection_id}/imports",
    response_model=ReferenceImportHistory,
    responses={404: _RESPONSES[404], 409: _RESPONSES[409], 503: _RESPONSES[503]},
)
def import_history(
    request: Request, collection_id: CollectionId, response: Response
) -> ReferenceImportHistory:
    """Saved import records for a collection, newest first: per-file counts and provenance."""
    container: AppContainer = request.app.state.container
    with collection_errors():
        history = container.reference_imports.history(collection_id)
    response.headers.update(COLLECTION_HEADERS)
    return history
