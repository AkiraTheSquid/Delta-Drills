#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/stellar-thread/Applications/Delta-Drills-Local"
DEPLOY_DIR="/home/stellar-thread/Applications/Delta-Drills-Deployed"

if [ ! -d "$DEPLOY_DIR/.git" ]; then
  echo "Deploy worktree not found. Run:"
  echo "  git -C \"$REPO_DIR\" worktree add -b deploy \"$DEPLOY_DIR\""
  exit 1
fi

git -C "$REPO_DIR" fetch origin
git -C "$REPO_DIR" pull --ff-only

git -C "$DEPLOY_DIR" checkout deploy
git -C "$DEPLOY_DIR" merge main

echo "Deploy worktree updated from main."
echo "Review, commit any deploy-only changes, then push deploy:"
echo "  git -C \"$DEPLOY_DIR\" push origin deploy"
