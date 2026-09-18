"""FastAPI entrypoint for Scholar RAG Agent."""

from api.application import create_app

app = create_app()


def main() -> None:
    """Run the API with uvicorn."""
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
