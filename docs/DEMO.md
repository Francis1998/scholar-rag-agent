# Demo

`docs/assets/demo.gif` shows the deterministic local flow:

1. Ingest a fixture paper.
2. Submit a research query.
3. Inspect the Observe -> Decide -> Act planning trace.
4. Return a citation-backed answer.

Recreate the asset with:

```bash
uv run python scripts/create_demo_gif.py
```

Run the same end-to-end flow in the terminal with:

```bash
uv run python scripts/demo_local.py
```
