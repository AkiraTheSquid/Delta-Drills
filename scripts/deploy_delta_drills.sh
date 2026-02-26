#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# deploy_delta_drills — one-command deploy for Delta Drills
#
# 1. Checks for uncommitted changes on main (auto-commits all)
# 2. Exports question bank to frontend/questions.json
# 3. Pushes main to origin
# 4. In the deploy worktree, merges main into deploy
# 5. Verifies no user data leaked into deploy tree
# 6. Pushes deploy to origin (triggers Vercel)
# ============================================================

REPO_DIR="/home/stellar-thread/Applications/Delta-Drills-Local"
DEPLOY_DIR="/home/stellar-thread/Applications/Delta-Drills-Deployed"
VERCEL_URL="https://delta-drills.vercel.app"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[deploy]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*"; }

# --- Pre-flight checks ---

if [ ! -d "$DEPLOY_DIR/.git" ] && [ ! -f "$DEPLOY_DIR/.git" ]; then
  error "Deploy worktree not found at $DEPLOY_DIR"
  echo "  Run: git -C \"$REPO_DIR\" worktree add -b deploy \"$DEPLOY_DIR\""
  exit 1
fi

# --- Step 1: Check for uncommitted changes on main ---

info "Checking for uncommitted changes on main..."
if ! git -C "$REPO_DIR" diff --quiet || ! git -C "$REPO_DIR" diff --cached --quiet; then
  warn "Uncommitted changes detected — auto-committing all files:"
  git -C "$REPO_DIR" status --short

  # Auto-commit everything, including untracked files.
  git -C "$REPO_DIR" add -A
  if ! git -C "$REPO_DIR" diff --cached --quiet; then
    git -C "$REPO_DIR" commit -m "chore: auto-commit before deploy"
  fi
fi

# --- Step 2: Export question bank ---

info "Exporting question bank to questions.json..."
python3 "$REPO_DIR/scripts/export_questions_json.py"

# If the export created/updated questions.json, stage and commit it
if ! git -C "$REPO_DIR" diff --quiet -- questions.json 2>/dev/null || \
   git -C "$REPO_DIR" ls-files --others --exclude-standard -- questions.json | grep -q .; then
  info "questions.json updated — auto-committing..."
  git -C "$REPO_DIR" add questions.json
  git -C "$REPO_DIR" commit -m "chore: update questions.json for deploy"
fi

# --- Step 3: Push main to origin ---

info "Pushing main to origin..."
git -C "$REPO_DIR" push origin main

# --- Step 3b: Deploy Supabase (best-effort, non-blocking) ---

if command -v supabase >/dev/null 2>&1 && [ -f "$REPO_DIR/supabase/config.toml" ]; then
  info "Deploying Supabase (best-effort)..."
  set +e
  (cd "$REPO_DIR" && supabase db push)
  if [ -d "$REPO_DIR/supabase/functions" ]; then
    (cd "$REPO_DIR" && supabase functions deploy --all)
  fi
  set -e
else
  warn "Supabase CLI/config not found — skipping Supabase deploy."
fi

# --- Step 4: Merge main into deploy worktree ---

info "Merging main into deploy branch..."
git -C "$DEPLOY_DIR" checkout deploy
git -C "$DEPLOY_DIR" merge main --no-edit

# Remove backend/ and Fly.io config from deploy branch — Vercel serves frontend only
DEPLOY_REMOVED=0
for item in backend/ Dockerfile fly.toml; do
  if git -C "$DEPLOY_DIR" ls-files --error-unmatch "$item" >/dev/null 2>&1; then
    git -C "$DEPLOY_DIR" rm -rf "$item"
    DEPLOY_REMOVED=1
  fi
done
if [ "$DEPLOY_REMOVED" -eq 1 ]; then
  info "Removing backend/Dockerfile/fly.toml from deploy branch (frontend only)..."
  git -C "$DEPLOY_DIR" commit -m "chore: remove backend and Fly.io config from deploy branch"
fi

# --- Step 5: Push deploy to origin (triggers Vercel) ---

info "Pushing deploy to origin..."
git -C "$DEPLOY_DIR" push origin deploy

# --- Step 6: Deploy backend to Fly.io ---

FLYCTL="${HOME}/.fly/bin/flyctl"
if [ -f "$FLYCTL" ] || command -v flyctl >/dev/null 2>&1; then
  FLYCTL="${FLYCTL:-flyctl}"
  info "Deploying backend to Fly.io..."
  (cd "$REPO_DIR" && "$FLYCTL" deploy --ha=false)
else
  warn "flyctl not found — skipping Fly.io deploy."
  warn "Install: curl -L https://fly.io/install.sh | sh"
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Deploy complete!${NC}"
echo -e "${GREEN}  Vercel:  ${VERCEL_URL}${NC}"
echo -e "${GREEN}  Backend: https://delta-drills-backend.fly.dev${NC}"
echo -e "${GREEN}======================================${NC}"
