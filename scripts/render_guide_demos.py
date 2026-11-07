"""Render guide demo GIFs from real, offline runs of each guide's own example.

Every demo executes the guide's ```python block verbatim in a fresh subprocess
(``PYTHONPATH=src``, scrubbed environment, sockets disabled, timeout). A small,
visible harness supplies sample inputs before the example and prints results
after it; the GIF shows the guide code in white, the harness in grey, and the
captured stdout. Nothing in a transcript is typed by hand.

Source connectors never reach a live API: ``httpx.AsyncClient`` is bound to an
``httpx.MockTransport`` that serves the recorded payloads already used by
``tests/test_connectors.py``. Any request without a recorded fixture fails the
render, so a demo cannot silently show an empty result.

Usage::

    python scripts/render_guide_demos.py                       # re-render all
    python scripts/render_guide_demos.py --only crossref_journals_source.gif
    python scripts/render_guide_demos.py --print --only bibtex-export.gif
    python scripts/render_guide_demos.py --check               # committed GIFs current?
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
WORK_ROOT = ROOT / "build" / "guide-demos"
TIMEOUT_SECONDS = 120
UNRECORDED = "!! no recorded fixture for"
PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)

# Harness helpers defined (not shown) before every demo; the GIF footer names them.
OFFLINE_PRELUDE = '''
from retrieval.models import Chunk, Document, SearchResult


def hit(chunk_id, title, text, score, **metadata):
    """Build one scored SearchResult; metadata values are stored as strings."""
    document_id = str(metadata.pop("document_id", "paper-" + chunk_id))
    source = str(metadata.pop("source", "demo"))
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title=title,
        text=text,
        source=source,
        metadata={key: str(value) for key, value in metadata.items()},
    )
    return SearchResult(chunk=chunk, score=score, retriever="hybrid")


def show(results, *keys):
    """Print a ranked list: rank, chunk id, score, title, and chosen fields."""
    if not results:
        print("   (no results)")
    for rank, result in enumerate(results, 1):
        chunk = result.chunk
        extra = "".join(
            f"  {key}={chunk.metadata.get(key, getattr(chunk, key, '-'))}" for key in keys
        )
        print(
            f"{rank}. {chunk.chunk_id}  {result.score:.3f}  {chunk.title}{extra}"
        )
'''

CONNECTOR_PRELUDE = '''
import importlib


def trace(target, *keys, titles=True):
    """Print every search() result of ``module.Class`` as it returns."""
    module_name, _, class_name = target.rpartition(".")
    connector = getattr(importlib.import_module(module_name), class_name)
    original = connector.search

    async def search(self, query, *args, **kwargs):
        documents = await original(self, query, *args, **kwargs)
        options = "".join(f", {key}={value!r}" for key, value in kwargs.items())
        print(f"-> search({query!r}{options}): {len(documents)} document(s)")
        for document in documents if titles else []:
            print(f"   - {document.title}")
            for key in keys:
                print(f"       {key}: {document.metadata.get(key, '-')}")
        return documents

    connector.search = search


def show_docs(documents, *keys):
    """Print normalized documents: title plus chosen metadata fields."""
    print(f"{len(documents)} document(s)")
    for document in documents:
        print(f"- {document.title}")
        for key in keys:
            print(f"    {key}: {document.metadata.get(key, '-')}")
'''


@dataclass(frozen=True)
class Route:
    """One recorded HTTP response served through ``httpx.MockTransport``.

    Attributes:
        url: Prefix (scheme, host, path) that the request URL must start with.
        fixture: Expression returning the recorded payload (a JSON-able object
            or a text body), evaluated with ``fx`` = ``tests.test_connectors``
            and ``inline(test, variable)``, which reads a literal assigned
            inside one of that module's test functions.
        params: Query parameters that must match exactly for this route.
        content_type: Content type used for text bodies.
    """

    url: str
    fixture: str
    params: tuple[tuple[str, str], ...] = ()
    content_type: str = "application/json"


@dataclass(frozen=True)
class GuideDemo:
    """A guide example plus the visible harness that feeds and prints it.

    Attributes:
        gif: File name under ``docs/assets`` that the guide embeds.
        guide: Guide path relative to the repository root.
        setup: Harness code run before the example (sample inputs, stubs).
        show: Harness code run after the example (prints the results).
        blocks: Indexes of the guide's ```python blocks to run, in order.
        routes: Recorded HTTP fixtures; non-empty marks a connector demo.
    """

    gif: str
    guide: str
    setup: str = ""
    show: str = ""
    blocks: tuple[int, ...] = (0,)
    routes: tuple[Route, ...] = ()

    @property
    def connector(self) -> bool:
        """Return whether the demo serves recorded HTTP fixtures."""
        return bool(self.routes)

    @property
    def title(self) -> str:
        """Return the guide's H1 without a trailing ``Guide``."""
        heading = (ROOT / self.guide).read_text(encoding="utf-8").splitlines()[0]
        return re.sub(r"\s+Guide$", "", heading.lstrip("# ").strip())

    def example(self) -> str:
        """Return the selected ```python blocks exactly as the guide shows them."""
        blocks = PYTHON_BLOCK.findall((ROOT / self.guide).read_text(encoding="utf-8"))
        return "\n".join(blocks[index].rstrip() + "\n" for index in self.blocks)


@dataclass(frozen=True)
class Transcript:
    """Real output of one demo run."""

    demo: GuideDemo
    stdout: str


class DemoError(RuntimeError):
    """Raised when a guide example fails or reaches an unrecorded endpoint."""


def _dedent(source: str) -> str:
    return re.sub(r"^\n", "", source).rstrip() + "\n" if source.strip() else ""


def _child_environment() -> dict[str, str]:
    """Return a scrubbed environment: no API keys, proxies, or mail settings."""
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TZ": "UTC",
        "LC_ALL": "C.UTF-8",
        "HOME": str(WORK_ROOT),
    }
    if "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return environment


def run_demo(demo: GuideDemo) -> Transcript:
    """Execute one demo in a subprocess and return its real stdout.

    Raises:
        DemoError: If the example raises, times out, writes to stderr, or
            requests a URL without a recorded fixture.
    """
    workdir = WORK_ROOT / Path(demo.gif).stem
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    payload = {
        "sections": [
            ["prelude", CONNECTOR_PRELUDE if demo.connector else OFFLINE_PRELUDE],
            ["setup", _dedent(demo.setup)],
            ["guide", demo.example()],
            ["show", _dedent(demo.show)],
        ],
        "routes": [
            {
                "url": route.url,
                "fixture": route.fixture,
                "params": dict(route.params),
                "content_type": route.content_type,
            }
            for route in demo.routes
        ],
    }
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=workdir,
            env=_child_environment(),
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise DemoError(f"{demo.guide}: example timed out") from error
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if completed.returncode != 0 or completed.stderr.strip():
        raise DemoError(f"{demo.guide}: example failed\n{completed.stderr.strip()}")
    if UNRECORDED in completed.stdout:
        raise DemoError(f"{demo.guide}: unrecorded request\n{completed.stdout}")
    if not completed.stdout.strip():
        raise DemoError(f"{demo.guide}: example printed nothing")
    return Transcript(demo=demo, stdout=completed.stdout)


# ---------------------------------------------------------------------------
# Child process: no network, recorded transport, verbatim guide execution.
# ---------------------------------------------------------------------------


def _disable_network() -> None:
    """Make any socket connection or DNS lookup fail loudly."""
    import socket

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("network access is disabled while rendering guide demos")

    socket.socket.connect = refuse  # type: ignore[method-assign,assignment]
    socket.socket.connect_ex = refuse  # type: ignore[method-assign,assignment]
    socket.create_connection = refuse  # type: ignore[assignment]
    socket.getaddrinfo = refuse  # type: ignore[assignment]


def _install_recorded_transport(routes: list[dict[str, object]]) -> None:
    """Bind every ``httpx.AsyncClient`` to recorded fixtures from the tests."""
    import ast

    import httpx
    import tests.test_connectors as fx

    def inline(test_name: str, variable: str = "payload") -> object:
        """Return the literal assigned to ``variable`` inside one connector test."""
        tree = ast.parse(Path(str(fx.__file__)).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == test_name:
                for statement in node.body:
                    if isinstance(statement, ast.AnnAssign | ast.Assign):
                        targets = (
                            statement.targets
                            if isinstance(statement, ast.Assign)
                            else [statement.target]
                        )
                        if any(isinstance(t, ast.Name) and t.id == variable for t in targets):
                            assert statement.value is not None
                            return ast.literal_eval(statement.value)
        raise LookupError(f"{test_name}: no literal {variable!r} in tests/test_connectors.py")

    def handle(request: httpx.Request) -> httpx.Response:
        bare = str(request.url.copy_with(query=None))
        for route in routes:
            params = route["params"]
            assert isinstance(params, dict)
            if bare.startswith(str(route["url"])) and all(
                request.url.params.get(key) == value for key, value in params.items()
            ):
                expression = str(route["fixture"])
                print(f"{request.method} {request.url}")
                if request.content:
                    print(f"  body {request.content.decode('utf-8')}")
                print(f"  <- recorded fixture {expression.replace('fx.', '')}")
                namespace = {"fx": fx, "inline": inline}
                body = eval(expression, namespace)  # noqa: S307 - repo fixture expression
                if isinstance(body, str):
                    headers = {"content-type": str(route["content_type"])}
                    return httpx.Response(200, text=body, headers=headers)
                return httpx.Response(200, json=body)
        print(f"{UNRECORDED} {request.method} {request.url}")
        return httpx.Response(404, json={})

    transport = httpx.MockTransport(handle)
    original_init = httpx.AsyncClient.__init__

    def init(self: httpx.AsyncClient, *args: object, **kwargs: object) -> None:
        kwargs["transport"] = transport
        kwargs["trust_env"] = False
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    httpx.AsyncClient.__init__ = init  # type: ignore[method-assign]


def _child_main() -> int:
    """Run the sections passed on stdin, sharing one namespace like a script."""
    import ast
    import asyncio
    import inspect

    payload = json.loads(sys.stdin.read())
    _disable_network()
    if payload["routes"]:
        _install_recorded_transport(payload["routes"])
    namespace: dict[str, object] = {"__name__": "__main__"}
    for label, source in payload["sections"]:
        if not source.strip():
            continue
        code = compile(source, f"<{label}>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        result = eval(code, namespace)  # noqa: S307 - executes the guide example on purpose
        if inspect.iscoroutine(result):
            asyncio.run(result)
        sys.stdout.flush()
    return 0


# ---------------------------------------------------------------------------
# Rendering: house palette from scripts/create_evidence_gif.py, crisp text.
# ---------------------------------------------------------------------------

WIDTH = 880
MIN_HEIGHT = 300
ROWS_PER_FRAME = 23
LINE_HEIGHT = 19
TEXT_LEFT = 40
TEXT_RIGHT = WIDTH - 40
TERMINAL_TOP = 90
TEXT_TOP = TERMINAL_TOP + 12
FOOTER_SPACE = 48
MAX_CODE_FRAMES = 2
MAX_OUTPUT_FRAMES = 3
INDENT_SCALE = 0.8
BACKGROUND = "#0f172a"
CARD = "#1e293b"
TERMINAL = "#0b1120"
BANNER = "#38bdf8"
TITLE = "#e2e8f0"
GUIDE_CODE = "#f1f5f9"
HARNESS_CODE = "#94a3b8"
OUTPUT = "#a7f3d0"
REQUEST = "#fcd34d"
COUNTER = "#34d399"
PALETTE = (BACKGROUND, CARD, TERMINAL, BANNER, TITLE, GUIDE_CODE, HARNESS_CODE, OUTPUT)
PALETTE += (REQUEST, COUNTER)

Line = tuple[str, str]


@dataclass(frozen=True)
class Fonts:
    """Pillow's bundled font at three sizes, so renders match on every machine."""

    small: ImageFont.FreeTypeFont
    title: ImageFont.FreeTypeFont
    body: ImageFont.FreeTypeFont

    @classmethod
    def load(cls) -> Fonts:
        """Load the bundled FreeType font (Pillow >= 10.1 with FreeType)."""
        fonts = [ImageFont.load_default(size=size) for size in (13, 20, 15)]
        if not all(isinstance(font, ImageFont.FreeTypeFont) for font in fonts):
            raise RuntimeError("Pillow needs FreeType support to render guide demos.")
        small, title, body = (font for font in fonts if isinstance(font, ImageFont.FreeTypeFont))
        return cls(small=small, title=title, body=body)

    def measure(self, text: str) -> float:
        """Return the drawn width of ``text`` in the non-anti-aliased body font."""
        return float(self.body.getlength(text, mode="1"))


def _wrap(text: str, color: str, measure: Callable[[str], float]) -> list[Line]:
    """Wrap one logical line to the terminal width, keeping its indentation."""
    indent = len(text) - len(text.lstrip(" "))
    cell = measure("0") * INDENT_SCALE
    wrapped: list[Line] = []
    rest = text.lstrip(" ")
    current = indent
    while True:
        limit = TEXT_RIGHT - TEXT_LEFT - current * cell
        if measure(rest) <= limit:
            wrapped.append((" " * current + rest, color))
            return wrapped
        cut = len(rest)
        while cut > 1 and measure(rest[:cut]) > limit:
            cut -= 1
        space = rest.rfind(" ", 0, cut)
        cut = space if space > cut // 2 else cut
        wrapped.append((" " * current + rest[:cut], color))
        rest = rest[cut:].lstrip(" ")
        current = indent + 4


def _code_lines(demo: GuideDemo, measure: Callable[[str], float]) -> list[Line]:
    lines: list[Line] = [(f"$ python demo.py   # example from {demo.guide}", BANNER)]
    for source, color in (
        (_dedent(demo.setup), HARNESS_CODE),
        (demo.example(), GUIDE_CODE),
        (_dedent(demo.show), HARNESS_CODE),
    ):
        for raw in source.rstrip("\n").splitlines():
            lines.extend(_wrap(raw, color, measure))
    return lines


def _output_lines(transcript: Transcript, measure: Callable[[str], float]) -> list[Line]:
    note = "real stdout, exit 0"
    if transcript.demo.connector:
        note += ", recorded fixtures, no network"
    lines: list[Line] = [(f"$ python demo.py   # {note}", BANNER)]
    for raw in transcript.stdout.rstrip("\n").splitlines():
        color = OUTPUT
        if raw.startswith(("GET ", "POST ", "  body ")):
            color = REQUEST
        elif raw.startswith("  <- recorded fixture"):
            color = HARNESS_CODE
        lines.extend(_wrap(raw, color, measure))
    return lines


def _pages(lines: list[Line], *, max_pages: int, rerun: str) -> list[list[Line]]:
    """Split lines into frames, repeating the prompt line on each frame."""
    header, body = lines[0], lines[1:]
    per_page = ROWS_PER_FRAME - 1
    pages = [body[start : start + per_page] for start in range(0, len(body), per_page)] or [[]]
    if len(pages) > max_pages:
        pages = pages[:max_pages]
        hidden = len(body) - sum(len(page) for page in pages) + 1
        pages[-1] = [*pages[-1][:-1], (f"... {hidden} more lines: {rerun}", HARNESS_CODE)]
    return [[header, *page] for page in pages]


def _frame(
    demo: GuideDemo,
    lines: Sequence[Line],
    *,
    caption: str,
    counter: str,
    height: int,
    fonts: Fonts,
) -> Image.Image:
    image = Image.new("RGB", (WIDTH, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.fontmode = "1"
    draw.rounded_rectangle((12, 12, WIDTH - 12, height - 12), radius=14, fill=CARD)
    kind = "RECORDED FIXTURE, NO NETWORK" if demo.connector else "OFFLINE / REAL EXECUTION"
    draw.text((30, 24), f"SCHOLAR RAG / GUIDE DEMO / {kind}", font=fonts.small, fill=BANNER)
    draw.text((30, 44), demo.title, font=fonts.title, fill=TITLE)
    draw.text((30, 70), caption, font=fonts.small, fill=HARNESS_CODE)
    bottom = height - FOOTER_SPACE
    draw.rounded_rectangle((26, TERMINAL_TOP, WIDTH - 26, bottom), radius=10, fill=TERMINAL)
    cell = fonts.measure("0") * INDENT_SCALE
    for row, (text, color) in enumerate(lines):
        stripped = text.lstrip(" ")
        x = TEXT_LEFT + (len(text) - len(stripped)) * cell
        draw.text((x, TEXT_TOP + row * LINE_HEIGHT), stripped, font=fonts.body, fill=color)
    if demo.connector:
        footer = "tests/test_connectors.py payloads via httpx.MockTransport; trace()/show_docs()"
    else:
        footer = "Real stdout of the guide example; harness helpers hit()/show()"
    footer += " in scripts/render_guide_demos.py"
    draw.text((30, height - 38), footer, font=fonts.small, fill=HARNESS_CODE)
    draw.text((WIDTH - 66, height - 38), counter, font=fonts.small, fill=COUNTER)
    return image


def _palette() -> Image.Image:
    """Return a palette image holding exactly the colours the renderer draws."""
    palette = Image.new("P", (1, 1))
    channels = [value for color in PALETTE for value in ImageColor.getrgb(color)[:3]]
    palette.putpalette(channels + channels[:3] * (256 - len(PALETTE)))
    return palette


def render(transcript: Transcript) -> bytes:
    """Return the GIF bytes for one transcript (deterministic for a Pillow build)."""
    fonts = Fonts.load()
    demo = transcript.demo
    rerun = f"python scripts/render_guide_demos.py --print --only {demo.gif}"
    code = _pages(_code_lines(demo, fonts.measure), max_pages=MAX_CODE_FRAMES, rerun=rerun)
    output = _pages(
        _output_lines(transcript, fonts.measure), max_pages=MAX_OUTPUT_FRAMES, rerun=rerun
    )
    rows = max(len(page) for page in (*code, *output))
    height = max(MIN_HEIGHT, TEXT_TOP + rows * LINE_HEIGHT + 10 + FOOTER_SPACE)
    legend = "white: verbatim guide example   grey: demo harness (sample input / print)"
    captioned = [(page, legend, 4500) for page in code]
    captioned += [(page, "captured stdout of the run above (not edited)", 6000) for page in output]
    palette = _palette()
    frames = [
        _frame(
            demo,
            page,
            caption=caption,
            counter=f"{index}/{len(captioned)}",
            height=height,
            fonts=fonts,
        ).quantize(palette=palette, dither=Image.Dither.NONE)
        for index, (page, caption, _duration) in enumerate(captioned, start=1)
    ]
    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=[duration for _page, _caption, duration in captioned],
        loop=0,
        optimize=True,
        disposal=1,
    )
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ARXIV_API = "https://export.arxiv.org/api/query"
CROSSREF = "https://api.crossref.org/"
DATACITE = "https://api.datacite.org/"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
OPENALEX = "https://api.openalex.org/"
ORCID = "https://pub.orcid.org/v3.0/"
WIKIDATA = "https://www.wikidata.org/w/api.php"


def _traced(
    gif: str,
    target: str,
    keys: tuple[str, ...],
    *routes: Route,
    titles: bool = True,
    blocks: tuple[int, ...] = (0,),
    setup: str = "",
    show: str = "",
) -> GuideDemo:
    """Build a connector demo whose harness prints every ``search()`` result."""
    arguments = ", ".join(f'"{value}"' for value in (target, *keys))
    trace = f"trace({arguments}{'' if titles else ', titles=False'})"
    guide = f"docs/guides/{Path(gif).stem.upper()}_GUIDE.md"
    return GuideDemo(
        gif=gif,
        guide=guide,
        setup=f"{_dedent(setup)}{trace}\n",
        show=show,
        blocks=blocks,
        routes=routes,
    )


DEMOS: tuple[GuideDemo, ...] = (
    GuideDemo(
        gif="abstract-keyword-boost.gif",
        guide="docs/guides/ABSTRACT_KEYWORD_BOOST_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Sparse retrieval baselines", "BM25 stays strong.", 0.91,
        abstract="BM25 and SPLADE baselines for passage ranking."),
    hit("c2", "Message passing for molecules", "Atom graphs as input.", 0.74,
        abstract="Graph neural networks predict molecular properties."),
    hit("c3", "Protein language models", "Sequence embeddings.", 0.70,
        abstract="Language models over protein sequences."),
]
print("input ranking:")
show(retrieved_results)
""",
        show="""
booster = AbstractKeywordBoost(alpha=0.5)
boosted = booster.boost(retrieved_results, query="graph neural molecular properties")
print("after AbstractKeywordBoost(alpha=0.5):")
show(boosted)
""",
    ),
    GuideDemo(
        gif="abstract_overlap_boost.gif",
        guide="docs/guides/ABSTRACT_OVERLAP_BOOST_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Sparse retrieval baselines", "BM25 stays strong.", 0.88,
        abstract="BM25 baselines for passage ranking"),
    hit("c2", "Message passing for molecules", "Atom graphs as input.", 0.80,
        abstract="graph neural networks for molecules"),
]
print("input ranking:")
show(retrieved_results)
""",
        show="""
boosted = processor.boost(retrieved_results, query="graph neural networks for molecules")
print("after AbstractOverlapBooster() (alpha=0.3):")
show(boosted)
""",
    ),
    GuideDemo(
        gif="adaptive_retrieval_gate.gif",
        guide="docs/guides/ADAPTIVE_RETRIEVAL_GATE_GUIDE.md",
        setup="""
from retrieval.sparse import BM25Retriever

user_query = "What evidence links sleep loss to memory consolidation?"
retriever = BM25Retriever()
retriever.add_chunks([
    hit("c1", "Sleep and memory", "Sleep loss impairs memory consolidation.", 0).chunk,
    hit("c2", "Coffee markets", "Arabica prices rose in 2024.", 0).chunk,
])
answer_without_retrieval = lambda reason: print("answer_without_retrieval:", reason)
synthesize = lambda results: print(
    "synthesize:", [(r.chunk.chunk_id, round(r.score, 3)) for r in results]
)
""",
        show="""
print(decision.action, decision.matched_cues, "-", decision.reason)
for query in ["hello", "What can you do?", "Compare BM25 versus dense methods"]:
    other = gate.decide(query)
    print(f"{query!r}: {other.action} {other.matched_cues} - {other.reason}")
""",
    ),
    GuideDemo(
        gif="agentic_chunk_boundary.gif",
        guide="docs/guides/AGENTIC_CHUNK_BOUNDARY_GUIDE.md",
        blocks=(0, 1),
        setup="""
markdown_text = (
    "# Methods\\n\\n" + "Graph networks pass messages between atoms. " * 12 + "\\n\\n"
    + "## Data\\n\\n" + "QM9 and ZINC provide property labels. " * 14 + "\\n\\n"
    + "## Results\\n\\n" + "Error on QM9 drops by twelve percent. " * 9
)
print(f"markdown_text: {len(markdown_text)} chars")
""",
        show="""
for number, text in enumerate(chunks, 1):
    print(f"chunk {number}: {len(text):4d} chars  starts {text[:34]!r}")
for chunk in retrieval_chunks:
    print(chunk.chunk_id, "chunk_index", chunk.metadata["chunk_index"], chunk.source)
""",
    ),
    GuideDemo(
        gif="answerability_gate.gif",
        guide="docs/guides/ANSWERABILITY_GATE_GUIDE.md",
        setup="""
query = "Does sleep deprivation impair memory consolidation?"
retrieved_results = [
    hit("c1", "Sleep and memory", "Sleep deprivation impairs memory consolidation.", 0.82),
    hit("c2", "Hippocampal replay", "Replay during sleep supports consolidation.", 0.77),
    hit("c3", "Coffee markets", "Arabica prices rose in 2024.", 0.41),
]
refuse_or_rewrite = lambda: print("refuse_or_rewrite()")
synthesize = lambda results: print("synthesize:", [r.chunk.chunk_id for r in results])
""",
        show="""
print(f"answerability={report.answerability:.2f}  result_scores={report.result_scores}")
print(f"kept_count={report.kept_count}  dropped_count={report.dropped_count}")
""",
    ),
    GuideDemo(
        gif="author_count_boost.gif",
        guide="docs/guides/AUTHOR_COUNT_BOOST_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Single-author essay", "Opinion on GNNs.", 0.90, authors="Lee"),
    hit("c2", "Consortium report", "Large benchmark.", 0.86, author_count=40),
    hit("c3", "Lab study", "Controlled experiment.", 0.80, authors="Kim; Patel; Ruiz"),
]
print("input ranking:")
show(retrieved_results, "authors", "author_count")
""",
        show="""
boosted = processor.boost(retrieved_results)
print("after AuthorCountBooster() (2..12 authors preferred, alpha=0.25):")
show(boosted, "authors", "author_count")
""",
    ),
    GuideDemo(
        gif="authority_boost.gif",
        guide="docs/guides/AUTHORITY_BOOST_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Workshop note", "Early idea.", 0.84, is_peer_reviewed="false"),
    hit("c2", "Flagship journal", "Full study.", 0.80, venue_rank=1, impact_factor=12.5),
    hit("c3", "Unlabelled hit", "No authority metadata.", 0.79),
    hit("c4", "Curated source", "Reviewed dataset.", 0.78, source_authority=0.9),
]
print("input ranking:")
show(retrieved_results)
""",
        show="""
print("after AuthorityBooster(alpha=0.3):")
show(boosted)
""",
    ),
    GuideDemo(
        gif="bibtex-export.gif",
        guide="docs/guides/BIBTEX_EXPORT_GUIDE.md",
        setup="""
ranked_hits = [
    hit("c1", "Attention Is All You Need", "Transformers.", 0.93, document_id="p1",
        doi="10.48550/arXiv.1706.03762", authors="Vaswani, Shazeer, Parmar",
        year=2017, venue="NeurIPS", type="conference"),
    hit("c2", "Attention Is All You Need", "Lower-scored chunk of p1.", 0.61,
        document_id="p1", doi="10.48550/arXiv.1706.03762", year=2017),
    hit("c3", "BERT", "Bidirectional encoders.", 0.88, document_id="p2",
        authors="Devlin and Chang", year=2019, journal="NAACL"),
]
""",
        show="print(bib)",
    ),
    GuideDemo(
        gif="citation-graph.gif",
        guide="docs/guides/CITATION_GRAPH_GUIDE.md",
        setup="""
corpus = [
    hit("a1", "GraphRAG survey", "Survey.", 0.90, doi="10.1/a", references="10.1/b; 10.1/c"),
    hit("b1", "KG question answering", "KGQA.", 0.0, doi="10.1/b"),
    hit("c1", "Entity linking", "Linking.", 0.0, doi="10.1/c", references="10.1/d"),
    hit("d1", "Open knowledge bases", "Two hops away.", 0.0, doi="10.1/d"),
    hit("e1", "Follow-up study", "Cites the survey.", 0.0, doi="10.1/e", cites="10.1/a"),
]
corpus_chunks = [result.chunk for result in corpus]
seed_results = corpus[:1]
""",
        show="""
print("one hop from seed a1 (10.1/a), hop_decay=0.5:")
show(expanded, "doi")
""",
    ),
    GuideDemo(
        gif="citation_count_boost.gif",
        guide="docs/guides/CITATION_COUNT_BOOST_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "New preprint", "Fresh result.", 0.86, citation_count=0),
    hit("c2", "Solid follow-up", "Replication.", 0.82, cited_by_count=150),
    hit("c3", "Classic paper", "Widely used.", 0.80, citation_count=12000),
]
print("input ranking:")
show(retrieved_results, "citation_count", "cited_by_count")
""",
        show="""
print("after CitationCountBooster(alpha=0.3):")
show(boosted, "citation_count", "cited_by_count")
""",
    ),
    GuideDemo(
        gif="citation_groundedness_score.gif",
        guide="docs/guides/CITATION_GROUNDEDNESS_SCORE_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Sleep and memory", "Sleep deprivation impairs memory consolidation.", 0.9,
        authors="Walker, Stickgold", year=2006),
    hit("c2", "Caffeine study", "Caffeine increases alertness in shift workers.", 0.8,
        authors="Smith", year=2019),
]
draft_answer = (
    "Sleep deprivation impairs memory consolidation [1]. "
    "Caffeine cures chronic insomnia (Walker, 2006)."
)
flag_misattributed_citation = lambda marker, sentence: print("flag:", marker, "|", sentence)
""",
        show="""
print(f"groundedness={report.groundedness:.2f}  grounded={report.grounded_count}"
      f"  ungrounded={report.ungrounded_count}")
for citation in report.citations:
    print(citation.marker, citation.candidate_chunk_ids, f"overlap={citation.overlap_score:.2f}",
          "grounded" if citation.grounded else "NOT grounded")
""",
    ),
    GuideDemo(
        gif="claim_density_boost.gif",
        guide="docs/guides/CLAIM_DENSITY_BOOST_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Background", "Sleep has been studied for decades. Many labs exist.", 0.84),
    hit("c2", "Findings", "We show that sleep loss impairs recall. "
        "Our results suggest a dose effect.", 0.80),
]
print("input ranking:")
show(retrieved_results)
""",
        show="""
boosted = processor.boost(retrieved_results, query="sleep loss recall")
print("after ClaimDensityBooster() (alpha=0.3):")
show(boosted)
""",
    ),
    GuideDemo(
        gif="claim_verification_gate.gif",
        guide="docs/guides/CLAIM_VERIFICATION_GATE_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Sleep and memory", "Sleep deprivation impairs memory consolidation.", 0.9),
]
draft_answer = (
    "Sleep deprivation impairs memory consolidation. "
    "Coffee fully reverses every sleep loss effect."
)
flag_ungrounded = lambda claim: print("flag_ungrounded:", claim)
""",
        show="""
print(f"groundedness={report.groundedness:.2f}  supported={report.supported_count}"
      f"  unsupported={report.unsupported_count}")
for verdict in report.claims:
    print(f"{verdict.support_score:.2f}", verdict.supporting_chunk_ids, verdict.claim)
""",
    ),
    GuideDemo(
        gif="coherence_boost.gif",
        guide="docs/guides/COHERENCE_BOOST_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Mixed notes", "Transformer attention is popular. Bananas are yellow. "
        "Tax law changed.", 0.85),
    hit("c2", "Focused passage", "Transformer attention weighs tokens. "
        "Attention heads let the transformer focus.", 0.80),
]
print("input ranking:")
show(retrieved_results)
""",
        show="""
print("after CoherenceBooster(alpha=0.3):")
show(boosted)
""",
    ),
    GuideDemo(
        gif="contextual_compression.gif",
        guide="docs/guides/CONTEXTUAL_COMPRESSION_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "GNNs for chemistry", "Our lab moved buildings in 2021. Graph neural "
        "networks predict molecular properties from atom graphs. Funding came from "
        "two agencies. Message passing aggregates neighbouring atoms.", 0.91),
    hit("c2", "Unrelated", "The cafeteria menu changed. Parking is limited.", 0.42),
]
""",
        show="""
for result in context_results:
    print(f"{result.chunk.chunk_id} ({len(result.chunk.text)} chars): {result.chunk.text}")
""",
    ),
    GuideDemo(
        gif="cross_encoder_gate.gif",
        guide="docs/guides/CROSS_ENCODER_GATE_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "GNNs for molecules", "Graph neural networks predict molecules.", 0.88),
    hit("c2", "Weather report", "Rain is expected tomorrow.", 0.86),
    hit("c3", "Molecular graphs", "Neural message passing on molecular graphs.", 0.70),
]
print("input ranking:")
show(retrieved_results)
""",
        show="""
kept = stage.gate(retrieved_results, query="graph neural networks for molecules")
print("after CrossEncoderGate() (min_score=0.2):")
show(kept)
""",
    ),
    GuideDemo(
        gif="diversity-cap-gate.gif",
        guide="docs/guides/DIVERSITY_CAP_GATE_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Survey, section 1", "Intro.", 0.95, document_id="survey"),
    hit("c2", "Survey, section 2", "Methods.", 0.93, document_id="survey"),
    hit("c3", "Survey, section 3", "Results.", 0.91, document_id="survey"),
    hit("c4", "Benchmark paper", "Numbers.", 0.80, document_id="bench"),
    hit("c5", "Survey, section 4", "Outlook.", 0.78, document_id="survey"),
]
""",
        show="""
gate = DiversityCapGate(max_per_source=2)
print("DiversityCapGate(max_per_source=2):")
show(gate.gate(retrieved_results), "document_id")
""",
    ),
    GuideDemo(
        gif="entity_overlap_boost.gif",
        guide="docs/guides/ENTITY_OVERLAP_BOOST_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Generic tuning", "Fine-tuning language models on QA data.", 0.83),
    hit("c2", "BERT on SQuAD", "We fine-tune BERT on SQuAD and report EM.", 0.78),
]
print("input ranking:")
show(retrieved_results)
""",
        show="""
boosted = processor.boost(retrieved_results, query="BERT fine-tuning on SQuAD")
print("after EntityOverlapBooster() (alpha=0.3):")
show(boosted)
""",
    ),
    GuideDemo(
        gif="evaluation-harness.gif",
        guide="docs/guides/EVALUATION_HARNESS_GUIDE.md",
    ),
    GuideDemo(
        gif="freshness_boost.gif",
        guide="docs/guides/FRESHNESS_BOOST_GUIDE.md",
        setup="""
results = [
    hit("c1", "Classic method", "Older but relevant.", 0.90, published_at="2016-03-01"),
    hit("c2", "Recent method", "New benchmark.", 0.88, published_at="2026-05-10"),
    hit("c3", "Undated note", "No date metadata.", 0.70),
]
print("input ranking:")
show(results, "published_at")
""",
        show="""
print("after FreshnessBooster(half_life_days=730, relevance_weight=0.75):")
show(fresh_results, "published_at")
""",
    ),
    GuideDemo(
        gif="keyword-match-gate.gif",
        guide="docs/guides/KEYWORD_MATCH_GATE_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "GNN molecules", "Graph neural network for molecules.", 0.90),
    hit("c2", "Graph theory", "Graph coloring bounds.", 0.85),
    hit("c3", "Neural molecules", "Neural models of molecules.", 0.70),
]
""",
        show="""
gate = KeywordMatchGate(min_coverage=0.5)
kept = gate.gate(retrieved_results, query="graph neural network molecules")
print("KeywordMatchGate(min_coverage=0.5) keeps (score = query coverage):")
show(kept)
""",
    ),
    GuideDemo(
        gif="language_prefer.gif",
        guide="docs/guides/LANGUAGE_PREFER_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Graphennetze", "Deutsch.", 0.90, language="de"),
    hit("c2", "Graph networks", "English.", 0.80, language="en"),
    hit("c3", "Reseaux de graphes", "Francais.", 0.75, lang="fr"),
]
print("input ranking:")
show(retrieved_results, "language", "lang")
""",
        show="""
print("boost mode (alpha=0.3):")
show(preferred, "language", "lang")
print("filter mode:")
show(filtered, "language", "lang")
""",
    ),
    GuideDemo(
        gif="lexical_overlap_boost.gif",
        guide="docs/guides/LEXICAL_OVERLAP_BOOST_GUIDE.md",
        setup="""
query = "sparse lexical retrieval"
retrieved_results = [
    hit("c1", "Dense encoders", "Dense vectors for semantic search.", 0.88),
    hit("c2", "BM25 revisited", "Sparse lexical retrieval with BM25.", 0.80),
]
print("input ranking:")
show(retrieved_results)
""",
        show="""
print("after LexicalOverlapBooster(alpha=0.3):")
show(boosted)
""",
    ),
    GuideDemo(
        gif="literature-review-outline.gif",
        guide="docs/guides/LITERATURE_REVIEW_OUTLINE_GUIDE.md",
        setup="""
ranked_results = [
    hit("c1", "GraphRAG: from local to global", "Survey.", 0.94, document_id="p1",
        section="background"),
    hit("c2", "Community summaries", "Method.", 0.81, document_id="p2",
        section="methods"),
    hit("c3", "KG-augmented QA benchmark", "Benchmark.", 0.77, document_id="p3"),
]
""",
    ),
    GuideDemo(
        gif="metadata-equals-gate.gif",
        guide="docs/guides/METADATA_EQUALS_GATE_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Journal study", "Peer-reviewed.", 0.90, publication_type="journal-article",
        language="en"),
    hit("c2", "Preprint", "Not yet reviewed.", 0.88, publication_type="preprint",
        language="en"),
    hit("c3", "German article", "Deutsch.", 0.80, publication_type="journal-article",
        language="de"),
]
""",
        show="""
gate = MetadataEqualsGate(required={"publication_type": "journal-article", "language": "en"})
print("MetadataEqualsGate(required=journal-article + en) keeps:")
show(gate.gate(retrieved_results), "publication_type", "language")
""",
    ),
    GuideDemo(
        gif="min-abstract-length-gate.gif",
        guide="docs/guides/MIN_ABSTRACT_LENGTH_GATE_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Stub record", "Abstract not available.", 0.95),
    hit("c2", "Full abstract", "We benchmark graph neural networks on twelve molecular "
        "property datasets and report consistent gains.", 0.82),
]
for result in retrieved_results:
    print(result.chunk.chunk_id, len(result.chunk.text), "chars")
""",
        show="""
gate = MinAbstractLengthGate(min_chars=80)
print("MinAbstractLengthGate(min_chars=80) keeps:")
show(gate.gate(retrieved_results))
""",
    ),
    GuideDemo(
        gif="min-unique-sources-gate.gif",
        guide="docs/guides/MIN_UNIQUE_SOURCES_GATE_GUIDE.md",
        setup="""
one_paper = [
    hit("c1", "Survey, part 1", "Intro.", 0.91, document_id="survey"),
    hit("c2", "Survey, part 2", "Methods.", 0.88, document_id="survey"),
]
two_papers = [*one_paper, hit("c3", "Benchmark", "Numbers.", 0.75, document_id="bench")]
""",
        show="""
gate = MinUniqueSourcesGate(min_sources=2)
print("one document  ->")
show(gate.gate(one_paper), "document_id")
print("two documents ->")
show(gate.gate(two_papers), "document_id")
""",
    ),
    GuideDemo(
        gif="novelty_diversify.gif",
        guide="docs/guides/NOVELTY_DIVERSIFY_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "GNN benchmark", "graph neural networks predict molecular properties", 0.92),
    hit("c2", "GNN benchmark (copy)", "graph neural networks predict molecular "
        "properties well", 0.90),
    hit("c3", "Protein folding", "language models fold protein sequences", 0.80),
]
print("input ranking:")
show(retrieved_results)
""",
        show="""
print("after NoveltyDiversifier(alpha=0.5):")
show(diversified)
""",
    ),
    GuideDemo(
        gif="open_access_prefer.gif",
        guide="docs/guides/OPEN_ACCESS_PREFER_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Paywalled study", "Closed.", 0.90, open_access="false"),
    hit("c2", "Open study", "Free to read.", 0.84, is_oa="true"),
    hit("c3", "Unknown access", "No flag.", 0.80),
]
print("input ranking:")
show(retrieved_results, "open_access", "is_oa")
""",
        show="""
print("boost mode (alpha=0.3):")
show(preferred, "open_access", "is_oa")
print("filter mode:")
show(filtered, "open_access", "is_oa")
""",
    ),
    GuideDemo(
        gif="paper-chat-memory.gif",
        guide="docs/guides/PAPER_CHAT_MEMORY_GUIDE.md",
        show="""
print(prompt_block)
for turn in memory.get_turns("s1"):
    print(turn.role, turn.document_ids, turn.chunk_ids)
""",
    ),
    GuideDemo(
        gif="paraphrase_collapse.gif",
        guide="docs/guides/PARAPHRASE_COLLAPSE_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Original", "Sleep deprivation impairs memory consolidation in healthy "
        "adults.", 0.91),
    hit("c2", "Paraphrase", "In healthy adults, sleep deprivation impairs memory "
        "consolidation.", 0.89),
    hit("c3", "Different claim", "Caffeine improves alertness in night shifts.", 0.70),
]
print("input ranking:")
show(retrieved_results)
""",
        show="""
print("after ParaphraseCollapser() (threshold=0.85, 3-gram Jaccard):")
show(stage.collapse(retrieved_results))
""",
    ),
    GuideDemo(
        gif="parent_document.gif",
        guide="docs/guides/PARENT_DOCUMENT_GUIDE.md",
        setup="""
full_document = Document(
    document_id="paper-1", title="Sleep and memory", source="pdf",
    text="Abstract. Methods. Results: sleep loss impairs consolidation. Discussion.",
)
child_chunk_results = [
    hit("p1-c3", "Sleep and memory", "Results: sleep loss impairs...", 0.82,
        document_id="paper-1"),
    hit("p2-c1", "Second paper", "A child span.", 0.77, document_id="paper-2"),
    hit("p1-c1", "Sleep and memory", "Abstract.", 0.64, document_id="paper-1"),
    hit("p9-c1", "No parent stored", "Orphan child.", 0.60, document_id="paper-9"),
]
""",
        show="""
for result in parent_results:
    print(f"{result.chunk.document_id}  {result.score:.2f}  {result.retriever} {result.path}")
    print(f"    {result.chunk.text}")
""",
    ),
    GuideDemo(
        gif="peer-reviewed-gate.gif",
        guide="docs/guides/PEER_REVIEWED_GATE_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "arXiv preprint", "Not reviewed.", 0.93, peer_reviewed="false"),
    hit("c2", "Journal article", "Reviewed.", 0.85, peer_reviewed="true"),
    hit("c3", "Unlabelled", "No flag.", 0.80),
]
""",
        show="""
print("PeerReviewedGate() keeps:")
show(PeerReviewedGate().gate(retrieved_results), "peer_reviewed")
print("PeerReviewedGate(require_peer_reviewed=False) keeps:")
show(PeerReviewedGate(require_peer_reviewed=False).gate(retrieved_results), "peer_reviewed")
""",
    ),
    GuideDemo(
        gif="preprint_demote.gif",
        guide="docs/guides/PREPRINT_DEMOTE_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "arXiv version", "Preprint.", 0.88, venue="arXiv"),
    hit("c2", "Journal version", "Published.", 0.84, type="journal-article"),
    hit("c3", "bioRxiv study", "Preprint.", 0.80, publication_type="preprint"),
]
print("input ranking:")
show(retrieved_results)
""",
        show="""
print("after PreprintDemoter(alpha=0.25):")
show(demoted, "venue", "type", "publication_type")
""",
    ),
    GuideDemo(
        gif="query_decomposition.gif",
        guide="docs/guides/QUERY_DECOMPOSITION_GUIDE.md",
        setup="""
from retrieval.sparse import BM25Retriever

retriever = BM25Retriever()
retriever.add_chunks([
    hit("m1", "GNNs for molecules", "graph neural networks for molecules", 0).chunk,
    hit("s1", "Sparse ranking", "sparse retrieval for ranking with BM25", 0).chunk,
    hit("x1", "Unrelated", "coffee prices and markets", 0).chunk,
])
""",
        show="""
for part in parts:
    print("part:", part)
print("fused (BM25 run per part, then RRF):")
for result in fused:
    print(f"{result.chunk.chunk_id}  {result.score:.4f}  {result.retriever} {result.path}")
""",
    ),
    GuideDemo(
        gif="recency_half_life.gif",
        guide="docs/guides/RECENCY_HALF_LIFE_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Older classic", "Seminal method.", 0.90, year=2006),
    hit("c2", "Recent study", "New results.", 0.82, year=2025),
    hit("c3", "Undated note", "No year.", 0.80),
]
print("input ranking:")
show(retrieved_results, "year")
""",
        show="""
boosted = processor.boost(retrieved_results, query="memory consolidation")
print("after RecencyHalfLifeBooster() (half_life=5y, ref_year=2026, alpha=0.3):")
show(boosted, "year")
""",
    ),
    GuideDemo(
        gif="reciprocal-rank-fusion-gate.gif",
        guide="docs/guides/RECIPROCAL_RANK_FUSION_GATE_GUIDE.md",
        setup="""
dense_results = [
    hit("a", "Dense favourite", "Semantic match.", 0.91),
    hit("b", "Shared hit", "Found by both.", 0.85),
]
bm25_results = [
    hit("b", "Shared hit", "Found by both.", 12.4),
    hit("c", "Keyword favourite", "Exact terms.", 11.0),
]
""",
        show="""
print("fused with k=60 (score = sum of 1 / (60 + rank)):")
show(fused)
""",
    ),
    GuideDemo(
        gif="required_metadata_gate.gif",
        guide="docs/guides/REQUIRED_METADATA_GATE_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Complete record", "Has both.", 0.90, doi="10.1/x", year=2024),
    hit("c2", "No year", "Missing year.", 0.88, doi="10.1/y"),
    hit("c3", "Blank DOI", "Empty value.", 0.80, doi="", year=2023),
]
""",
        show="""
print("RequiredMetadataGate(required_keys=['doi', 'year']) keeps:")
show(filtered, "doi", "year")
""",
    ),
    GuideDemo(
        gif="retracted_filter.gif",
        guide="docs/guides/RETRACTED_FILTER_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Retracted trial", "Withdrawn.", 0.92, retraction_status="retracted"),
    hit("c2", "Replication", "Holds up.", 0.85),
    hit("c3", "Flagged study", "Bool flag.", 0.80, is_retracted="true"),
]
""",
        show="""
print("RetractedFilter() (mode='filter'):")
show(processor.filter(retrieved_results))
print("RetractedFilter(mode='demote') (alpha=0.5):")
show(RetractedFilter(mode="demote").filter(retrieved_results))
""",
    ),
    GuideDemo(
        gif="score-threshold-gate.gif",
        guide="docs/guides/SCORE_THRESHOLD_GATE_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Strong hit", "Relevant.", 0.82),
    hit("c2", "Borderline", "Partly relevant.", 0.50),
    hit("c3", "Weak hit", "Noise.", 0.21),
]
""",
        show="""
print("ScoreThresholdGate(min_score=0.5) keeps:")
show(ScoreThresholdGate(min_score=0.5).gate(retrieved_results))
""",
    ),
    GuideDemo(
        gif="section_type_boost.gif",
        guide="docs/guides/SECTION_TYPE_BOOST_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Acknowledgements", "Thanks.", 0.90, section="acknowledgements"),
    hit("c2", "Results", "Effect sizes.", 0.84, section="results"),
    hit("c3", "Discussion", "Interpretation.", 0.82, section_type="discussion"),
]
print("input ranking:")
show(retrieved_results, "section", "section_type")
""",
        show="""
print("after SectionTypeBooster(alpha=0.3, discussion=0.8):")
show(boosted, "section", "section_type")
""",
    ),
    GuideDemo(
        gif="sentence_window_expand.gif",
        guide="docs/guides/SENTENCE_WINDOW_EXPAND_GUIDE.md",
        setup="""
full_paper_text = (
    "We recruited 40 adults. Participants slept 4 hours. "
    "Recall dropped by 20 percent. The effect persisted for two days."
)
retrieved_results = [
    hit("c1", "Sleep study", "Recall dropped by 20 percent.", 0.88,
        document_text=full_paper_text),
]
""",
        show="""
print("hit text:     ", retrieved_results[0].chunk.text)
print("window (+/-1):", expanded[0].chunk.text)
""",
    ),
    GuideDemo(
        gif="source-authority-gate.gif",
        guide="docs/guides/SOURCE_AUTHORITY_GATE_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Blog post", "Opinion.", 0.92, source_authority="low"),
    hit("c2", "Nature paper", "Study.", 0.85, venue="Nature"),
    hit("c3", "Curated dataset", "Data.", 0.80, source_authority=0.7),
]
""",
        show="""
gate = SourceAuthorityGate(alpha=0.3, min_authority=0.5, venue_tiers={"Nature": "high"})
print("SourceAuthorityGate(alpha=0.3, min_authority=0.5, Nature=high):")
show(gate.gate(retrieved_results), "source_authority", "venue")
""",
    ),
    GuideDemo(
        gif="temporal_freshness_cutoff.gif",
        guide="docs/guides/TEMPORAL_FRESHNESS_CUTOFF_GUIDE.md",
        blocks=(0, 1),
        setup="""
retrieved_results = [
    hit("c1", "Last year", "Recent.", 0.90, published_at="2025-06-01"),
    hit("c2", "Old survey", "Dated.", 0.88, year=2019),
    hit("c3", "Undated", "No date.", 0.70),
]
""",
        show="""
print("pinned cutoff (max_age_days=365, as_of=2026-01-01) keeps:")
show(cutoff.filter(retrieved_results), "published_at", "year")
""",
    ),
    GuideDemo(
        gif="term_coverage_boost.gif",
        guide="docs/guides/TERM_COVERAGE_BOOST_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Partial match", "graph methods overview", 0.86),
    hit("c2", "Full match", "graph neural networks for molecules", 0.80),
]
print("input ranking:")
show(retrieved_results)
""",
        show="""
boosted = stage.boost(retrieved_results, query="graph neural networks molecules")
print("after TermCoverageBooster() (alpha=0.4):")
show(boosted)
""",
    ),
    GuideDemo(
        gif="time-decay-gate.gif",
        guide="docs/guides/TIME_DECAY_GATE_GUIDE.md",
        setup="""
from datetime import date

retrieved_results = [
    hit("c1", "Older", "2023.", 0.90, published_at="2023-01-01"),
    hit("c2", "Newer", "2025.", 0.70, published_at="2025-07-01"),
    hit("c3", "Undated", "No date.", 0.95),
]
""",
        show="""
gate = TimeDecayGate(half_life_days=365, as_of=date(2026, 1, 1))
print("TimeDecayGate(half_life_days=365, as_of=2026-01-01):")
show(gate.gate(retrieved_results), "published_at")
""",
    ),
    GuideDemo(
        gif="title_match_boost.gif",
        guide="docs/guides/TITLE_MATCH_BOOST_GUIDE.md",
        setup="""
query = "graph neural networks"
retrieved_results = [
    hit("c1", "A survey of retrieval", "Mentions graph neural networks once.", 0.86),
    hit("c2", "Graph neural networks", "Core paper.", 0.80),
]
print("input ranking:")
show(retrieved_results)
""",
        show="""
print("after TitleMatchBooster(alpha=0.3):")
show(boosted)
""",
    ),
    GuideDemo(
        gif="venue_tier_boost.gif",
        guide="docs/guides/VENUE_TIER_BOOST_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "Workshop paper", "Unknown venue.", 0.88, venue="Local Workshop"),
    hit("c2", "Society paper", "Mapped to tier2.", 0.84, venue="My Society Journal"),
    hit("c3", "Nature paper", "Built-in tier1.", 0.80, journal="Nature"),
]
print("input ranking:")
show(retrieved_results, "venue", "journal")
""",
        show="""
print("after VenueTierBooster(alpha=0.3, My Society Journal=tier2):")
show(boosted, "venue", "journal")
""",
    ),
    GuideDemo(
        gif="year-range-gate.gif",
        guide="docs/guides/YEAR_RANGE_GATE_GUIDE.md",
        setup="""
retrieved_results = [
    hit("c1", "2018 paper", "Too old.", 0.90, year=2018),
    hit("c2", "2022 paper", "In range.", 0.85, year=2022),
    hit("c3", "No year", "Unparseable.", 0.80, year="n.d."),
]
""",
        show="""
gate = YearRangeGate(min_year=2020, max_year=2024)
print("YearRangeGate(min_year=2020, max_year=2024) keeps:")
show(gate.gate(retrieved_results), "year")
""",
    ),
    *(
        _traced(
            "arxiv_html_abstract_source.gif",
            "ingestion.arxiv_html_abstract.ArxivHtmlAbstractConnector",
            ("arxiv_id", "abstract_source"),
            Route(ARXIV_API, "fx.ARXIV_HTML_API_FIXTURE", content_type="application/atom+xml"),
            Route("https://arxiv.org/abs/", "fx.ARXIV_HTML_ABS_FIXTURE", content_type="text/html"),
        ),
        _traced(
            "biorxiv_collections_source.gif",
            "ingestion.biorxiv_collections.BioRxivCollectionsConnector",
            (),
            Route("https://api.biorxiv.org/details/biorxiv/", "fx._biorxiv_collections_payload()"),
            titles=False,
        ),
        _traced(
            "crossref_journals_source.gif",
            "ingestion.crossref_journals.CrossrefJournalsConnector",
            ("issn", "issns", "publisher", "subjects", "total_dois"),
            Route(CROSSREF + "journals", "fx._crossref_journals_search_payload()"),
        ),
        _traced(
            "crossref_members_source.gif",
            "ingestion.crossref_members.CrossrefMembersConnector",
            (),
            Route(CROSSREF + "members", "fx._crossref_members_list_payload()"),
            titles=False,
        ),
        _traced(
            "crossref_relations_source.gif",
            "ingestion.crossref_relations.CrossrefRelationsConnector",
            (),
            Route(
                CROSSREF + "works/",
                "inline('test_crossref_relations_connector_resolves_doi_queries')",
            ),
            Route(CROSSREF + "works", "fx._crossref_relations_list_payload()"),
            titles=False,
            show='show_docs(work, "doi", "relation_types", "relation_count")',
        ),
        _traced(
            "crossref_types_source.gif",
            "ingestion.crossref_types.CrossrefTypesConnector",
            ("crossref_type", "doi", "year"),
            Route(CROSSREF + "works", "inline('test_crossref_types_connector_filters_by_type')"),
        ),
        _traced(
            "crossref_works_funder_source.gif",
            "ingestion.crossref_works_funder.CrossrefWorksFunderConnector",
            ("funders", "funder_id"),
            Route(CROSSREF + "works", "fx._crossref_works_funder_payload()"),
        ),
        _traced(
            "crossref_works_issn_type_source.gif",
            "ingestion.crossref_works_issn_type.CrossrefWorksIssnTypeConnector",
            ("crossref_type", "issn"),
            Route(CROSSREF + "works", "fx._crossref_works_issn_type_payload()"),
        ),
        _traced(
            "crossref_works_license_source.gif",
            "ingestion.crossref_works_license.CrossrefWorksLicenseConnector",
            ("doi", "license_url"),
            Route(CROSSREF + "works", "fx._crossref_works_license_payload()"),
        ),
        _traced(
            "crossref_works_type_license_source.gif",
            "ingestion.crossref_works_type_license.CrossrefWorksTypeLicenseConnector",
            ("license_url",),
            Route(CROSSREF + "works", "fx._crossref_works_type_license_payload()"),
        ),
        _traced(
            "datacite_dois_prefix_source.gif",
            "ingestion.datacite_dois_prefix.DataCiteDoisPrefixConnector",
            ("doi", "doi_prefix"),
            Route(DATACITE + "dois", "fx._datacite_dois_prefix_payload()"),
        ),
        _traced(
            "datacite_events_source.gif",
            "ingestion.datacite_events.DataCiteEventsConnector",
            ("relation_type", "subject_doi", "object_doi", "event_source"),
            Route(DATACITE + "events", "fx._datacite_events_payload()"),
        ),
        _traced(
            "datacite_related_source.gif",
            "ingestion.datacite_related.DataciteRelatedConnector",
            (),
            Route(
                DATACITE + "dois",
                "inline('test_datacite_related_connector_searches_and_enriches_related_identifiers')",
            ),
            titles=False,
        ),
        _traced(
            "datacite_reports_source.gif",
            "ingestion.datacite_reports.DataCiteReportsConnector",
            ("doi", "resource_type", "publisher", "year"),
            Route(DATACITE + "dois", "fx._datacite_reports_payload()"),
        ),
        _traced(
            "europepmc_grants_source.gif",
            "ingestion.europepmc_grants.EuropePmcGrantsConnector",
            ("grant_id", "funder", "pi", "institution", "amount"),
            Route(
                "https://www.ebi.ac.uk/europepmc/GristAPI/rest/get/",
                "fx._europepmc_grants_payload()",
            ),
        ),
        _traced(
            "europepmc_preprints_source.gif",
            "ingestion.europepmc_preprints.EuropePmcPreprintsConnector",
            ("preprint_server", "doi", "year", "cited_by_count"),
            Route(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                "fx._europepmc_preprints_payload()",
            ),
        ),
        _traced(
            "openaire_projects_source.gif",
            "ingestion.openaire_projects.OpenaireProjectsConnector",
            (),
            Route(
                "https://api.openaire.eu/search/projects", "fx._openaire_projects_search_payload()"
            ),
            titles=False,
            blocks=(0, 1),
            show='show_docs(documents, "code", "funder", "start_date")',
        ),
        _traced(
            "openalex_author_works_source.gif",
            "ingestion.openalex_author_works.OpenAlexAuthorWorksConnector",
            ("year",),
            Route(
                OPENALEX + "authors",
                "inline('test_openalex_author_works_connector_searches_author_then_works', "
                "'authors_payload')",
            ),
            Route(OPENALEX + "works", "fx._openalex_author_works_payload()"),
        ),
        _traced(
            "openalex_authors_source.gif",
            "ingestion.openalex_authors.OpenAlexAuthorsConnector",
            (),
            Route(OPENALEX + "authors/", "fx._openalex_authors_search_payload()['results'][0]"),
            Route(OPENALEX + "authors", "fx._openalex_authors_search_payload()"),
            titles=False,
            blocks=(0, 1),
            show='show_docs(documents, "author_id", "orcid", "h_index")',
        ),
        _traced(
            "openalex_concepts_ancestors_source.gif",
            "ingestion.openalex_concepts_ancestors.OpenAlexConceptsAncestorsConnector",
            ("concept_id", "ancestor_path"),
            Route(
                OPENALEX + "concepts/",
                "fx._openalex_concepts_ancestors_payload()['results'][0]",
            ),
            Route(OPENALEX + "concepts", "fx._openalex_concepts_ancestors_payload()"),
        ),
        _traced(
            "openalex_concepts_source.gif",
            "ingestion.openalex_concepts.OpenAlexConceptsConnector",
            (),
            Route(
                OPENALEX + "concepts/",
                "inline('test_openalex_concepts_connector_resolves_concept_id_with_works_filter',"
                " 'concept_payload')",
            ),
            Route(
                OPENALEX + "works",
                "inline('test_openalex_concepts_connector_resolves_concept_id_with_works_filter',"
                " 'works_payload')",
            ),
            Route(OPENALEX + "concepts", "fx._openalex_concepts_search_payload()"),
            titles=False,
            blocks=(0, 1),
            show='show_docs(documents, "concept_id", "level", "works_count")',
        ),
        _traced(
            "openalex_funders_source.gif",
            "ingestion.openalex_funders.OpenAlexFundersConnector",
            ("openalex_funder_id", "country", "grants_count"),
            Route(OPENALEX + "funders", "fx._openalex_funders_search_payload()"),
        ),
        _traced(
            "openalex_institutions_source.gif",
            "ingestion.openalex_institutions.OpenAlexInstitutionsConnector",
            (),
            Route(
                OPENALEX + "institutions/",
                "fx._openalex_institutions_search_payload()['results'][0]",
            ),
            Route(OPENALEX + "institutions", "fx._openalex_institutions_search_payload()"),
            titles=False,
            blocks=(0, 1),
            show='show_docs(documents, "institution_id", "country_code", "ror")',
        ),
        _traced(
            "openalex_keywords_source.gif",
            "ingestion.openalex_keywords.OpenAlexKeywordsConnector",
            ("openalex_keyword_id", "works_count", "cited_by_count"),
            Route(OPENALEX + "keywords", "fx._openalex_keywords_search_payload()"),
        ),
        _traced(
            "openalex_publishers_source.gif",
            "ingestion.openalex_publishers.OpenAlexPublishersConnector",
            ("openalex_publisher_id", "parent_publisher", "country_codes"),
            Route(OPENALEX + "publishers", "fx._openalex_publishers_search_payload()"),
        ),
        _traced(
            "openalex_sources_hierarchy_source.gif",
            "ingestion.openalex_sources_hierarchy.OpenAlexSourcesHierarchyConnector",
            ("openalex_source_id", "hierarchy_path"),
            Route(OPENALEX + "sources/", "fx._openalex_sources_hierarchy_source()"),
            Route(OPENALEX + "sources", "{'results': [fx._openalex_sources_hierarchy_source()]}"),
        ),
        _traced(
            "openalex_sources_host_org_source.gif",
            "ingestion.openalex_sources_host_org.OpenAlexSourcesHostOrgConnector",
            ("host_organization_name", "host_org_works_count"),
            Route(
                OPENALEX + "publishers",
                "inline('test_openalex_sources_host_org_connector_searches_publishers_then_sources',"
                " 'publisher_payload')",
            ),
            Route(OPENALEX + "sources", "{'results': [fx._openalex_sources_host_org_source()]}"),
        ),
        _traced(
            "openalex_sources_source.gif",
            "ingestion.openalex_sources.OpenAlexSourcesConnector",
            ("openalex_source_id", "issn_l"),
            Route(OPENALEX + "sources", "fx._openalex_sources_search_payload()"),
        ),
        _traced(
            "openalex_topics_hierarchy_source.gif",
            "ingestion.openalex_topics_hierarchy.OpenAlexTopicsHierarchyConnector",
            ("openalex_topic_id", "hierarchy_path"),
            Route(OPENALEX + "topics", "fx._openalex_topics_hierarchy_payload()"),
        ),
        _traced(
            "orcid_education_source.gif",
            "ingestion.orcid_education.OrcidEducationConnector",
            ("organization", "degree_title", "start_date", "end_date"),
            Route(ORCID + "0000-0002-1825-0097/educations", "fx._orcid_education_payload()"),
        ),
        _traced(
            "orcid_employments_source.gif",
            "ingestion.orcid_employments.OrcidEmploymentsConnector",
            ("organization", "role_title", "start_date", "end_date"),
            Route(ORCID + "0000-0002-1825-0097/employments", "fx._orcid_employments_payload()"),
        ),
        _traced(
            "orcid_works_summaries_source.gif",
            "ingestion.orcid_works_summaries.OrcidWorksSummariesConnector",
            ("doi", "year", "work_type"),
            Route(ORCID + "0000-0002-1825-0097/works", "fx._orcid_works_summaries_payload()"),
        ),
        _traced(
            "pubmed_mesh_source.gif",
            "ingestion.pubmed_mesh.PubmedMeshConnector",
            ("mesh_ui", "tree_numbers"),
            Route(EUTILS + "esearch.fcgi", "fx._pubmed_mesh_esearch_payload()"),
            Route(EUTILS + "esummary.fcgi", "fx._pubmed_mesh_esummary_payload()"),
        ),
        _traced(
            "semantic_scholar_bulk_source.gif",
            "ingestion.semantic_scholar_bulk.SemanticScholarBulkConnector",
            ("semantic_scholar_id",),
            Route(
                "https://api.semanticscholar.org/graph/v1/paper/batch",
                "inline('test_semantic_scholar_bulk_connector_batches_ids')",
            ),
        ),
        _traced(
            "ssrn_source.gif",
            "ingestion.ssrn.SsrnConnector",
            (),
            Route(CROSSREF + "works/", "inline('test_ssrn_connector_resolves_ssrn_doi')"),
            Route(CROSSREF + "works", "fx._ssrn_work_list_payload()"),
            titles=False,
            blocks=(0, 1),
            show='show_docs(documents, "doi", "ssrn_url", "container")',
        ),
        _traced(
            "wikidata_scholarly_source.gif",
            "ingestion.wikidata_scholarly.WikidataScholarlyConnector",
            (),
            Route(WIKIDATA, "fx._wikidata_search_payload()", (("action", "wbsearchentities"),)),
            Route(WIKIDATA, "fx._wikidata_entities_payload()", (("action", "wbgetentities"),)),
            titles=False,
            blocks=(0, 1),
            show='show_docs(documents, "wikidata_id", "doi", "year")',
        ),
    ),
)


def _select(names: Iterable[str]) -> list[GuideDemo]:
    wanted = list(names)
    if not wanted:
        return list(DEMOS)
    by_name = {demo.gif: demo for demo in DEMOS}
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        raise SystemExit(f"Unknown demo(s): {', '.join(unknown)}")
    return [by_name[name] for name in wanted]


def main(argv: Sequence[str] | None = None) -> int:
    """Run, print, check, or render the selected guide demos."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="+", default=[], help="GIF file names to process")
    parser.add_argument("--print", action="store_true", help="print transcripts only")
    parser.add_argument("--check", action="store_true", help="fail if a committed GIF is stale")
    parser.add_argument("--list", action="store_true", help="list registered demos")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.child:
        return _child_main()
    if arguments.list:
        for demo in DEMOS:
            kind = "connector" if demo.connector else "offline"
            print(f"{demo.gif:48} {kind:9} {demo.guide}")
        return 0
    demos = _select(arguments.only)
    with ThreadPoolExecutor(max_workers=max(1, arguments.jobs)) as pool:
        futures = [pool.submit(run_demo, demo) for demo in demos]
    failures: list[str] = []
    for demo, future in zip(demos, futures, strict=True):
        try:
            transcript = future.result()
        except DemoError as error:
            failures.append(str(error))
            continue
        if arguments.print:
            print(f"===== {demo.gif} ({demo.guide})\n{transcript.stdout}")
            continue
        target = ASSETS / demo.gif
        data = render(transcript)
        if arguments.check:
            if not target.exists() or target.read_bytes() != data:
                failures.append(f"{target.relative_to(ROOT)} is stale; re-render it")
            continue
        target.write_bytes(data)
        print(f"rendered {target.relative_to(ROOT)} ({len(data):,} bytes)")
    shutil.rmtree(WORK_ROOT, ignore_errors=True)
    if WORK_ROOT.parent.is_dir() and not any(WORK_ROOT.parent.iterdir()):
        WORK_ROOT.parent.rmdir()
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
