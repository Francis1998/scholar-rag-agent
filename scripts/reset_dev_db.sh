#!/usr/bin/env bash
# Reset development database for scholar-rag-agent
# Usage: ./scripts/reset_dev_db.sh
set -euo pipefail

echo "Resetting dev environment for scholar-rag-agent..."

if command -v docker &>/dev/null; then
    docker compose down -v 2>/dev/null || true
fi

echo "Clearing local caches..."
find . -name "*.db" -not -path "./.git/*" -delete 2>/dev/null || true
find . -name "*.sqlite" -not -path "./.git/*" -delete 2>/dev/null || true
find . -name "*.sqlite3" -not -path "./.git/*" -delete 2>/dev/null || true

echo "Dev environment reset. Run 'make dev-install' to reinstall dependencies."
