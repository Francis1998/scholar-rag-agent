# Demo

The demo gallery is generated from `scripts/create_demo_gif.py` and committed under
`docs/assets/`.

`docs/assets/demo.gif` shows the deterministic local flow:

1. Ingest a fixture paper.
2. Submit a research query.
3. Inspect the Observe -> Decide -> Act planning trace.
4. Return a citation-backed answer.

Additional assets:

- `docs/assets/use_cases.gif`: starts from common research workflow issues and shows how Scholar RAG Agent addresses them.
- `docs/assets/planning_trace.gif`: shows Observe -> Decide -> Act planning and persisted audit events.
- `docs/assets/grounded_answer.gif`: shows validated claims, source chunk IDs, and `[UNGROUNDED]` protection.

Recreate the assets with:

```bash
uv run python scripts/create_demo_gif.py
```

Run the same end-to-end flow in the terminal with:

```bash
uv run python scripts/demo_local.py
```
