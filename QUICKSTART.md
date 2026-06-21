# Quickstart

## 1. Install

```bash
uv sync --extra dev
```

## 2. Run The Deterministic Demo

```bash
uv run python scripts/demo_local.py
```

## 3. Start The API

```bash
uv run uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs`, ingest a fixture or paper source, then submit a query to `/query`.
