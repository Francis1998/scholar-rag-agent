"""CLI helper for ingesting local text fixtures."""

import argparse
from pathlib import Path

from ingestion.chunking import stable_id
from retrieval.models import Document


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Ingest a local text file as a paper fixture.")
    parser.add_argument("path", type=Path)
    return parser.parse_args()


def main() -> None:
    """Print normalized document metadata for a local text file."""
    args = parse_args()
    text = args.path.read_text(encoding="utf-8")
    document = Document(
        document_id=stable_id(str(args.path.resolve()), "doc"),
        title=args.path.stem,
        text=text,
        source=str(args.path),
        metadata={"source_type": "text"},
    )
    print(document.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
