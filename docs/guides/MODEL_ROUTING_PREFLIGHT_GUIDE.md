# Model routing preflight: verify without providers

Inspect **which adapter would be selected**, not whether a provider will accept
a request. This playbook adds executable route inspection and a synthetic
saved-evidence exercise to the [model catalog and migration guide](PROVIDER_MODELS_GUIDE.md)
and [routing reference](../CONFIGURATION.md#routing-and-fallbacks).
It changes no routing policy, model default, endpoint, or budget.

Use the existing Python 3.11+ development environment from the
[quickstart](../../QUICKSTART.md). Each Python block below is independent:
paste a whole block into a Python script or run it with the environment's
Python interpreter reading standard input. No server, provider account, real
paper, optional embedding model, or additional dependency is needed.
The [guide tests](../../tests/test_model_routing_preflight_guide.py) execute the
published blocks, rather than a second implementation.

**Do not substitute real credentials.** The examples use the existing
`offline_settings` helper, which accepts only explicit constructor inputs and
defaults, not environment, dotenv, or file-secret values. The one ordinary
`Settings` source-precedence exercise replaces the environment mapping with
synthetic values and uses a temporary dotenv file. Nothing prints a settings
dump, credential, adapter headers, or real request context.

## 1. Separate provider families from API model IDs

`default_model` is an exact lowercase family: `openai`, `anthropic`, `gemini`,
`kimi`, or `fake`. It is not a catalog ID. The four `*_model` fields are
nonblank, whitespace-trimmed API IDs; validation does not discover models or
check an account. A model ID without a nonempty key builds no live adapter.

```python
from pathlib import Path

from pydantic import ValidationError
from scripts.demo_evidence_export import offline_settings

from config import Settings
from llm.providers import OpenAIAdapter
from llm.router import build_model_router
from llm.schemas import LLMRequest, TaskType

baseline = offline_settings(Path("unused-preflight.sqlite3"))
# Reconstruct the helper's init-only Settings subclass, retaining validation.
settings = type(baseline)(
    database_path=baseline.database_path,
    default_model="openai",
    openai_model="  gpt-6.1-terra  ",
)
assert isinstance(settings, Settings)
assert settings.openai_model == "gpt-6.1-terra"
assert build_model_router(settings).route(TaskType.DEFAULT).provider_name == "fake"

try:
    type(baseline)(default_model="gpt-6.1-terra")
except ValidationError as error:
    assert error.errors()[0]["loc"] == ("default_model",)
    print("API ID rejected as a provider family")
else:
    raise AssertionError("Expected family validation to reject an API ID")

request = LLMRequest(task_type=TaskType.DEFAULT, prompt="Synthetic question", context="")
direct = OpenAIAdapter(api_key="", model="explicit-constructor-model")
assert direct.payload(request)["model"] == "explicit-constructor-model"
print("family=openai; requested_model=gpt-6.1-terra; selected_adapter=fake")
print("direct constructor keeps its own model; no generation attempted")
```

The unused database path is not opened by settings or router construction.
Direct adapter constructors do not load application settings; their explicit
`model=` is independent of `SCHOLAR_RAG_OPENAI_MODEL`. Do not call `generate()`
on this illustrative HTTP adapter. Prefer validated settings when constructing
the application; `model_copy(update=...)` is not a validation step.

## 2. Inspect every task and fallback before generation

`ModelRouter.route()` returns an adapter without calling it. The order is
**preferred family, configured default, OpenAI, fake**. Duplicate choices in
that chain are harmless; it does not search all other configured providers.

| `TaskType` | First choice | If that adapter is absent |
| --- | --- | --- |
| `REASONING` | `anthropic` | default, then `openai`, then `fake` |
| `SPEED` | `gemini` | default, then `openai`, then `fake` |
| `COST` | `kimi` | default, then `openai`, then `fake` |
| `DEFAULT` | configured default | `openai`, then `fake` |

This block builds real adapters with an explicitly fake nonempty key to
simulate presence. It **never calls generation**. The report reads only
`provider_name` and the public payload's model field (Gemini encodes its model
in the endpoint path). All prompts are synthetic; no private adapter fields
or credential headers are inspected.

```python
from pathlib import Path

import httpx
from scripts.demo_evidence_export import offline_settings

from llm.fake import FakeLLMAdapter
from llm.providers import GeminiAdapter, HTTPProviderAdapter
from llm.router import ModelRouter, build_model_router
from llm.schemas import LLMRequest, TaskType

key_names = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
}
cases = [
    ("all keyed", "openai", tuple(key_names)),
    ("fake default, all keyed", "fake", tuple(key_names)),
    ("fake default, only OpenAI keyed", "fake", ("openai",)),
    ("only default Gemini", "gemini", ("gemini",)),
    ("missing default, OpenAI keyed", "anthropic", ("openai",)),
    ("only unrelated Kimi keyed", "anthropic", ("kimi",)),
    ("no keys", "openai", ()),
]
baseline = offline_settings(Path("unused-preflight.sqlite3"))
request = LLMRequest(task_type=TaskType.DEFAULT, prompt="Synthetic inspection", context="")
routes = {}
for label, default, enabled in cases:
    settings = type(baseline)(
        database_path=baseline.database_path,
        default_model=default,
        **{
            alias: "synthetic-not-a-credential" if provider in enabled else ""
            for provider, alias in key_names.items()
        },
    )
    router = build_model_router(settings)
    routes[label] = {}
    print(label)
    for task in TaskType:
        adapter = router.route(task)
        model = None
        if isinstance(adapter, GeminiAdapter):
            model = (
                httpx.URL(adapter.endpoint).path.rsplit("/", 1)[-1].removesuffix(":generateContent")
            )
        elif isinstance(adapter, HTTPProviderAdapter):
            model = adapter.payload(request)["model"]
        routes[label][task.value] = adapter.provider_name
        print(f"  {task.value}: {adapter.provider_name}; requested_model={model}")

custom = FakeLLMAdapter()
direct_router = ModelRouter(adapters={"local-demo": custom}, default_provider="local-demo")
assert all(direct_router.route(task) is custom for task in TaskType)
assert routes["fake default, all keyed"]["reasoning"] == "anthropic"
assert routes["only unrelated Kimi keyed"]["reasoning"] == "fake"
assert routes["only unrelated Kimi keyed"]["cost"] == "kimi"
print("custom direct-router adapter: fake; generation calls: 0")
```

Expected provider matrix, in enum order:

| Synthetic configuration | REASONING | SPEED | COST | DEFAULT |
| --- | --- | --- | --- | --- |
| all keyed; default OpenAI | anthropic | gemini | kimi | openai |
| all keyed; default fake | anthropic | gemini | kimi | fake |
| only OpenAI keyed; default fake | fake | fake | fake | fake |
| only Gemini keyed; default Gemini | gemini | gemini | gemini | gemini |
| only OpenAI keyed; default Anthropic missing | openai | openai | openai | openai |
| only Kimi keyed; default Anthropic missing | fake | fake | kimi | fake |
| no keys; default OpenAI | fake | fake | fake | fake |

The requested IDs in the all-keyed case remain `claude-sonnet-5-5`,
`gemini-3.8-flash`, `kimi-k3.1-pro`, and `gpt-6-astra`.
An empty or `None` key is absent; the builder checks truthiness, not credential
validity. A nonempty placeholder (even whitespace) is not authentication.
Custom direct-router names are allowed by `ModelRouter`; they are not valid
application `Settings.default_model` values.

### Why `default_model="fake"` alone is not offline

Preferred adapters win **before** the default. If an Anthropic key is present,
`REASONING` still selects Anthropic with a fake default; similarly Gemini and
Kimi remain preferred for `SPEED` and `COST`. The current `/query` synthesis
always uses `REASONING`, not `DEFAULT`. Overriding only an OpenAI API ID
therefore does not force `/query` through OpenAI.

Use `offline_settings` for an isolated demonstration, not ordinary
`Settings(default_model="fake")`. Passing `_env_file=None` to ordinary settings
disables dotenv loading **only**, not process environment. For a real deployment,
deliberately control all sources, rebuild the application, and inspect its
routes before enabling generation. These examples do not inspect or modify an
already-running server.

Missing configuration is different from a failed provider call. An HTTP/auth
error, deadline, or unusable provider response does not trigger another
provider or fake answer. Existing transient retries remain within the selected
adapter. This is a selection preflight, not a live health check or failover test.

## 3. Prove configuration precedence and the rebuild boundary

For ordinary application settings, precedence is explicit `Settings(...)`
arguments, then process environment, then dotenv, then defaults. No file-secret
directory is configured by this application. If a caller supplies one, Pydantic
places that source below dotenv and above defaults. Credential aliases are the
unprefixed names in the previous example; model fields use `SCHOLAR_RAG_`.

The following scoped mock exposes only synthetic variables and a temporary dotenv file.
It demonstrates that `_env_file=None` still permits environment overrides,
explicit settings beat both, and an existing router keeps its original adapter.
It replaces the mapping without enumerating or printing the original environment,
and restores the original mapping on exit. Run this standalone exercise, not
inside a concurrently serving application.

```python
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from config import Settings
from llm.providers import HTTPProviderAdapter
from llm.router import build_model_router
from llm.schemas import LLMRequest, TaskType

with (
    TemporaryDirectory(prefix="routing-precedence-") as temporary,
    patch("os.environ", {"SCHOLAR_RAG_OPENAI_MODEL": "environment-model"}),
):
    dotenv = Path(temporary) / ".env"
    dotenv.write_text(
        "SCHOLAR_RAG_OPENAI_MODEL=dotenv-model\nOPENAI_API_KEY=synthetic-not-a-credential\n",
        encoding="utf-8",
    )
    request = LLMRequest(task_type=TaskType.DEFAULT, prompt="Synthetic", context="")
    initial = Settings(_env_file=dotenv)
    router = build_model_router(initial)
    assert initial.openai_model == "environment-model"
    assert Settings(_env_file=None).openai_model == "environment-model"
    explicit = Settings(_env_file=dotenv, openai_model="explicit-model")
    assert explicit.openai_model == "explicit-model"
    os.environ.pop("SCHOLAR_RAG_OPENAI_MODEL")
    assert Settings(_env_file=dotenv).openai_model == "dotenv-model"
    assert Settings(_env_file=None).openai_model == "gpt-6-astra"
    assert initial.openai_model == "environment-model"
    rebuilt = build_model_router(explicit)
    original_adapter = router.route(TaskType.DEFAULT)
    rebuilt_adapter = rebuilt.route(TaskType.DEFAULT)
    assert isinstance(original_adapter, HTTPProviderAdapter)
    assert isinstance(rebuilt_adapter, HTTPProviderAdapter)
    assert original_adapter.payload(request)["model"] == "environment-model"
    assert rebuilt_adapter.payload(request)["model"] == "explicit-model"
print("explicit > environment > dotenv > defaults")
print("_env_file=None does not disable environment")
print("old router=environment-model; rebuilt router=explicit-model")
```

The API constructs its container/router when `create_app` runs. Changing a
shell variable, dotenv file, or an unrelated `Settings` object does not mutate
those adapters. Restart the application with the intended environment (all
workers), or explicitly construct a new settings/container/app in Python.
Changing a parent shell cannot update an already-running process environment.
Removing only an environment override may expose an older dotenv value.
Saved run provenance remains historical; do not rewrite it to match a new
configuration.

## 4. Run fake generation and retain honest portfolio evidence

This exercise refuses to overwrite an existing output directory. It ingests
one explicitly synthetic note into a temporary SQLite database, runs the
real in-process API with the fake adapter, exports the completed frozen
evidence, and removes only its temporary database on exit.
HTTPX network transports and live-adapter generation are blocked during the
exercise; `TestClient` uses an in-process transport, not a listening server.
These guards are an executable check, not an operating-system firewall.

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from agent.evidence import EvidenceBundle
from api.application import create_app
from llm.router import build_model_router
from llm.schemas import TaskType

output = Path("routing-preflight-artifact")
output.mkdir(exist_ok=False)
text = (
    "Synthetic routing exercise, not a real paper or scientific finding. "
    "GraphRAG connects entities across these synthetic research passages."
)
with (
    TemporaryDirectory(prefix="routing-fake-") as temporary,
    patch(
        "httpx.HTTPTransport.handle_request", side_effect=AssertionError("Network forbidden")
    ) as sync_network,
    patch(
        "httpx.AsyncHTTPTransport.handle_async_request",
        side_effect=AssertionError("Network forbidden"),
    ) as async_network,
    patch(
        "llm.providers.HTTPProviderAdapter.generate",
        side_effect=AssertionError("Live generation forbidden"),
    ) as live_generate,
):
    settings = offline_settings(Path(temporary) / "synthetic.sqlite3")
    router = build_model_router(settings)
    routes = {task.value: router.route(task).provider_name for task in TaskType}
    assert set(routes.values()) == {"fake"}
    with TestClient(create_app(settings)) as client:
        ingested = client.post(
            "/ingest/text",
            json={"title": "Synthetic routing note", "source": "synthetic:preflight", "text": text},
        )
        ingested.raise_for_status()
        response = client.post(
            "/query", json={"query": "How does GraphRAG connect synthetic research passages?"}
        )
        response.raise_for_status()
        result = response.json()["result"]
        assert result["state"] == "DONE", result["error"]
        exported = client.get(f"/runs/{result['run_id']}/export", params={"format": "json"})
        exported.raise_for_status()
        bundle = EvidenceBundle.model_validate_json(exported.content)
        assert bundle.generation.provider == "fake"
        assert bundle.generation.model_name is None
        assert len(bundle.snapshot.sources) == 1
        assert bundle.snapshot.sources[0].chunk.text == text
        markdown = client.get(f"/runs/{result['run_id']}/export", params={"format": "markdown"})
        markdown.raise_for_status()
    live_generate.assert_not_called()
    sync_network.assert_not_called()
    async_network.assert_not_called()
    (output / "bundle.json").write_bytes(exported.content)
    (output / "bundle.md").write_bytes(markdown.content)
    checks = {
        "data": "synthetic, not scientific evidence",
        "routes": routes,
        "state": bundle.status,
        "provider": bundle.generation.provider,
        "requested_model": bundle.generation.model_name,
        "captured_sources": len(bundle.snapshot.sources),
        "live_generation_calls": live_generate.call_count,
        "http_network_calls": sync_network.call_count + async_network.call_count,
        "scientific_validation": False,
    }
    (output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
print(json.dumps(checks, indent=2))
print(f"Saved synthetic evidence to {output.resolve()}")
```

Expected: all four routes `fake`, state `DONE`, provider `fake`, requested model
`null`, one captured source, and zero live-generation/HTTP-network calls. Run IDs and
timestamps vary; this is deterministic fake behavior, not a byte-identical
full-run replay. Keep the three output files together and record the checkout
commit used when presenting them. Existing artifacts are never replaced;
choose a fresh output directory for another run.

A truthful portfolio caption:

> Exercised task-aware adapter selection and an in-process, local-first research
> workflow with synthetic data. Captured the actual completed fake answer and
> frozen source context. No provider inference, account entitlement, model
> quality, retrieval benchmark, or scientific conclusion was validated.

The fake adapter formats the question and up to three supplied citation IDs;
it does not reason over the paper. A `DONE` state or lexical grounding flag is
not proof of factual support, entailment, study quality, or research completeness.
The application uses FastAPI/Pydantic, HTTPX, SQLite, deterministic hash
embeddings, BM25, a co-mention graph, and a hand-written observe/decide/act
state machine, not LangGraph or a hosted vector database.

## What this preflight cannot establish

All live adapters are stateless, single-turn text adapters. Selecting a model
ID adds no tools, streaming, multimodal input, or reasoning-state replay.
The selected defaults remain unchanged. Official documentation reviewed
**2026-10-30 America/Los_Angeles** provides catalog context, not live inference:

| Official source | Relevant distinction |
| --- | --- |
| [OpenAI model catalog](https://developers.openai.com/api/docs/models.md) | Astra is the flagship; Sol, Terra, and Luna have different documented positioning, not measured tradeoffs here. |
| [GPT-6.1 Terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra.md) | Text Chat Completions is supported; tools require Responses. Its reasoning-effort options do not create an effort setting in this repository. |
| [Anthropic model overview](https://platform.claude.com/docs/en/models/overview) | Generally recommends Opus 5.5 and lists Fable/Sonnet/Haiku; this repository still selects Sonnet 5.5. |
| [Gemini 4 Argon](https://ai.google.dev/gemini-api/docs/models/gemini-4-argon) | Latest flagship; the existing SPEED choice remains Gemini 3.8 Flash, not an automatic upgrade. |
| [Kimi quickstart](https://platform.kimi.ai/docs/get-your-api-key.md) | Recommends K3.1 Pro and describes a high-speed variant; the variant is not an automatic fallback. |

This review does not redate earlier migration-contract checks in the
[provider guide](PROVIDER_MODELS_GUIDE.md). The existing
[mocked HTTPX model-contract tests](../../tests/test_provider_models.py) check
payloads and normalized provenance without live calls. Neither mocks nor this
preflight prove account access, current availability, latency, cost, or answer
quality. Any live evaluation is a separate, user-authorized activity using
permitted data and the existing [bounded migration checklist](PROVIDER_MODELS_GUIDE.md#selection-and-migration-checklist).

To re-execute these blocks and the existing router/configuration/mock contracts
from the repository root with its installed development environment:

```bash
env -i PATH="$PATH" HOME="$HOME" TMPDIR="${TMPDIR:-/tmp}" \
  .venv/bin/python -m pytest tests/test_model_routing_preflight_guide.py \
  tests/test_llm_router.py tests/test_provider_models.py \
  tests/test_default_provider_configuration.py -q
```

The guide tests use temporary directories, synthetic ambient-variable/dotenv
sentinels, and failing network/live-generation guards. They also verify every
matrix cell, the exact printed model IDs, the exported evidence, and refusal
to overwrite an existing output file, directory, or symlink.
