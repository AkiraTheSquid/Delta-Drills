#!/usr/bin/env bash
# sync_arena_fork.sh — keep a checkout of Seth's ARENA_3.0 fork on disk.
#
# WHY
#   The in-app ARENA notebooks (scripts/compile_arena_notebooks.py) read the
#   fork FIRST and Callum's ARENA_5.0 cut second, so a notebook pushed to
#   github.com/AkiraTheSquid/ARENA_3.0 is the notebook the app renders and the
#   Colab links open. Seth's study group pulls that same repo, so everyone —
#   the app, Colab, a local clone — is on one file. Seth, 2026-09-06: "make it
#   such that it still mirrors the github along with the colab ... they are
#   pulling the directory on their end on their own."
#
# WHAT
#   Clone (first run) or fast-forward (every run after) into
#   Local_Deployed_Shared/content/ARENA_3.0-fork/. That folder is gitignored
#   with the rest of content/ — it is upstream data, not this repo's.
#
#   The fetch names the URL every time, so ARENA_FORK_URL is honoured on an
#   existing checkout too — not only at clone time.
#
#   🔴 --ff-only, never merge: this checkout is read-only. If it cannot
#   fast-forward, something edited it by hand; fail loudly rather than build
#   notebooks from a state nobody pushed.
#
# EXIT CODES — the deploy reads them (This-Directory-Only/scripts/deploy_delta_drills.sh):
#   0  synced (or already current)
#   2  the remote could not be reached; the checkout on disk is untouched and
#      still usable — the deploy warns and compiles from it
#   3  the checkout is not a plain fast-forward of the remote (hand edits,
#      diverged history, wrong branch); the deploy must ABORT
#   1  anything else (bad clone, missing git, …)
#
# USAGE
#   scripts/sync_arena_fork.sh            # clone or pull, print the sha
#   ARENA_FORK_URL=... scripts/sync_arena_fork.sh   # override the remote
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK_DIR="$REPO_DIR/Local_Deployed_Shared/content/ARENA_3.0-fork"
FORK_URL="${ARENA_FORK_URL:-https://github.com/AkiraTheSquid/ARENA_3.0.git}"
BRANCH="${ARENA_FORK_BRANCH:-main}"

if [ ! -d "$FORK_DIR/.git" ]; then
  echo "[arena-fork] cloning $FORK_URL -> $FORK_DIR"
  if ! git clone --quiet --branch "$BRANCH" --single-branch "$FORK_URL" "$FORK_DIR"; then
    echo "[arena-fork] clone failed" >&2
    exit 2
  fi
else
  echo "[arena-fork] fetching $FORK_URL ($BRANCH)"
  if ! git -C "$FORK_DIR" fetch --quiet "$FORK_URL" "$BRANCH"; then
    echo "[arena-fork] fetch failed — remote unreachable; checkout left as it was" >&2
    exit 2
  fi
  if [ -n "$(git -C "$FORK_DIR" status --porcelain --untracked-files=no)" ]; then
    echo "[arena-fork] checkout has local modifications to TRACKED files — it is read-only; discard them first (untracked files such as Modulario README/watch.py templates are ignored)" >&2
    git -C "$FORK_DIR" status --short >&2
    exit 3
  fi
  if ! git -C "$FORK_DIR" merge --ff-only --quiet FETCH_HEAD; then
    echo "[arena-fork] checkout is not a fast-forward of $FORK_URL $BRANCH — refusing to build from it" >&2
    exit 3
  fi
fi

sha="$(git -C "$FORK_DIR" rev-parse --short HEAD)"
when="$(git -C "$FORK_DIR" log -1 --format=%cd --date=short)"
echo "[arena-fork] at $sha ($when)"
