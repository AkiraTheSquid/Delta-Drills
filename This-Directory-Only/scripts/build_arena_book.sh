#!/usr/bin/env bash
# Build the ARENA Jupyter Book and stage its output under
# Local_Deployed_Shared/arena-book/ so the deploy step rsyncs it to the
# deploy worktree.
#
# Run on its own to rebuild after editing _toc.yml / _config.yml / intro.md,
# or let deploy_delta_drills.sh call it.

set -euo pipefail

REPO_DIR="/home/stellar-thread/Applications/Delta-Drills-Local"
BOOK_SRC="$REPO_DIR/arena-book"
SHARED_DEST="$REPO_DIR/Local_Deployed_Shared/arena-book"
VENV="$BOOK_SRC/.venv"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[arena-book]${NC} $*"; }
warn() { echo -e "${YELLOW}[arena-book]${NC} $*"; }

if [ ! -d "$BOOK_SRC" ]; then
  echo "[arena-book] source not found at $BOOK_SRC" >&2
  exit 1
fi

if [ ! -d "$VENV" ]; then
  info "Creating venv at $VENV"
  python3 -m venv "$VENV"
fi

info "Installing/refreshing requirements"
"$VENV/bin/pip" install -q -r "$BOOK_SRC/requirements.txt"

info "Building Jupyter Book"
(cd "$BOOK_SRC" && "$VENV/bin/jupyter-book" build .)

if [ ! -f "$BOOK_SRC/_build/html/index.html" ]; then
  echo "[arena-book] build did not produce _build/html/index.html" >&2
  exit 1
fi

info "Staging build output to $SHARED_DEST"
rm -rf "$SHARED_DEST"
cp -r "$BOOK_SRC/_build/html" "$SHARED_DEST"

info "Done. $(du -sh "$SHARED_DEST" | cut -f1) staged."
