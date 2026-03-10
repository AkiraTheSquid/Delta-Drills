#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# deploy_delta_drills — one-command deploy for Delta Drills
#
# Runtime environment split:
# - localhost / local worktree: backend auth + backend practice state only
# - deployed host / Vercel: Supabase auth + Supabase practice state
#
# 1. Checks for uncommitted changes on main (auto-commits all)
# 2. Exports question bank artifacts
# 3. Pushes main to origin
# 4. Syncs Local_Deployed_Shared/ into the deploy worktree
# 5. Pushes deploy to origin (triggers Vercel)
# 6. Deploys the backend to Fly.io
# ============================================================

REPO_DIR="/home/stellar-thread/Applications/Delta-Drills-Local"
DEPLOY_DIR="/home/stellar-thread/Applications/Delta-Drills-Deployed"
REPO_THIS_DIR="$REPO_DIR/This-Directory-Only"
REPO_SHARED_DIR="$REPO_DIR/Local_Deployed_Shared"
DEPLOY_SHARED_DIR="$DEPLOY_DIR/Local_Deployed_Shared"
REFRESH_SPLIT_SCRIPT="$REPO_THIS_DIR/scripts/refresh_split_layout.py"
VERCEL_URL="https://delta-drills.vercel.app"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[deploy]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*"; }

auto_commit_if_dirty() {
  local repo_dir="$1"
  local message="$2"

  if ! git -C "$repo_dir" diff --quiet || ! git -C "$repo_dir" diff --cached --quiet || \
     git -C "$repo_dir" ls-files --others --exclude-standard | grep -q .; then
    warn "Uncommitted changes detected in $repo_dir — auto-committing all files:"
    git -C "$repo_dir" status --short
    git -C "$repo_dir" add -A
    if ! git -C "$repo_dir" diff --cached --quiet; then
      git -C "$repo_dir" commit -m "$message"
    fi
  fi
}

# --- Pre-flight checks ---

if [ ! -d "$DEPLOY_DIR/.git" ] && [ ! -f "$DEPLOY_DIR/.git" ]; then
  error "Deploy worktree not found at $DEPLOY_DIR"
  echo "  Run: git -C \"$REPO_DIR\" worktree add -b deploy \"$DEPLOY_DIR\""
  exit 1
fi
if [ ! -d "$REPO_SHARED_DIR" ]; then
  error "Shared deploy source not found at $REPO_SHARED_DIR"
  exit 1
fi
if [ ! -f "$REFRESH_SPLIT_SCRIPT" ]; then
  error "Split refresh helper not found at $REFRESH_SPLIT_SCRIPT"
  exit 1
fi

# --- Step 1: Check for uncommitted changes on main ---

info "Checking for uncommitted changes on main..."
git -C "$REPO_DIR" checkout main
auto_commit_if_dirty "$REPO_DIR" "chore: auto-commit before deploy"

# --- Step 2: Export question bank ---

info "Exporting question bank artifacts..."
python3 "$REPO_THIS_DIR/scripts/export_questions_json.py"
python3 "$REPO_THIS_DIR/scripts/extract_arena_prereqs.py"
python3 "$REFRESH_SPLIT_SCRIPT" --root "$REPO_DIR"

auto_commit_if_dirty "$REPO_DIR" "chore: update deploy artifacts"

# --- Step 3: Push main to origin ---

info "Pushing main to origin..."
git -C "$REPO_DIR" push origin main

# --- Step 3b: Deploy Supabase (best-effort, non-blocking) ---

if command -v supabase >/dev/null 2>&1 && [ -f "$REPO_THIS_DIR/supabase/config.toml" ]; then
  info "Deploying Supabase (best-effort)..."
  set +e
  (cd "$REPO_THIS_DIR" && supabase db push)
  if [ -d "$REPO_THIS_DIR/supabase/functions" ]; then
    (cd "$REPO_THIS_DIR" && supabase functions deploy --all)
  fi
  set -e
else
  warn "Supabase CLI/config not found — skipping Supabase deploy."
fi

# --- Step 4: Sync the shared subtree into the deploy worktree ---

info "Syncing Local_Deployed_Shared into deploy branch..."
git -C "$DEPLOY_DIR" checkout deploy
mkdir -p "$DEPLOY_SHARED_DIR" "$DEPLOY_DIR/This-Directory-Only"
auto_commit_if_dirty "$DEPLOY_DIR" "chore: checkpoint deploy worktree before shared sync"
rsync -a --delete "$REPO_SHARED_DIR"/ "$DEPLOY_SHARED_DIR"/
python3 "$REFRESH_SPLIT_SCRIPT" --root "$DEPLOY_DIR"
auto_commit_if_dirty "$DEPLOY_DIR" "chore: sync shared deploy payload"

# --- Step 5: Push deploy to origin (triggers Vercel) ---

info "Pushing deploy to origin..."
git -C "$DEPLOY_DIR" push origin deploy

# --- Step 6: Deploy backend to Fly.io ---

FLYCTL="${HOME}/.fly/bin/flyctl"
if [ -f "$FLYCTL" ] || command -v flyctl >/dev/null 2>&1; then
  FLYCTL="${FLYCTL:-flyctl}"
  info "Deploying backend to Fly.io..."
  (
    cd "$REPO_DIR" && \
    "$FLYCTL" deploy . \
      --config "$REPO_THIS_DIR/fly.toml" \
      --dockerfile "$REPO_THIS_DIR/Dockerfile" \
      --ignorefile "$REPO_DIR/.dockerignore" \
      --ha=false
  )
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
