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
LOG_DIR="$REPO_THIS_DIR/logs"
TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
LOG_FILE="$LOG_DIR/deploy_delta_drills-$TIMESTAMP.txt"
VERCEL_URL="https://delta-drills.vercel.app"
VERCEL_PROJECT="delta-drills"
VERCEL_SCOPE="seth-gibsons-projects"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[deploy]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*"; }

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

info "Writing deploy log to $LOG_FILE"
info "Shared frontend source of truth: $REPO_SHARED_DIR"
info "Deploy sync target (must be identical after sync): $DEPLOY_SHARED_DIR"

verify_vercel_frontend() {
  local url="$1"
  local attempts="${2:-10}"
  local delay_seconds="${3:-3}"
  local attempt
  local status

  for attempt in $(seq 1 "$attempts"); do
    status="$(curl -s -o /dev/null -w '%{http_code}' "$url" || true)"
    if [ "$status" = "200" ]; then
      info "Verified Vercel frontend is serving at $url"
      return 0
    fi
    warn "Vercel frontend check attempt $attempt/$attempts returned HTTP $status for $url"
    sleep "$delay_seconds"
  done

  return 1
}

auto_commit_if_dirty() {
  local repo_dir="$1"
  local message="$2"
  local status_output
  local tracked_changes
  local untracked_changes

  status_output="$(git -C "$repo_dir" status --short)"
  tracked_changes="$(printf '%s\n' "$status_output" | grep -v '^?? ' | grep -v 'This-Directory-Only/logs/' || true)"
  untracked_changes="$(git -C "$repo_dir" ls-files --others --exclude-standard | grep -v '^This-Directory-Only/logs/' || true)"

  if [ -n "$tracked_changes" ] || [ -n "$untracked_changes" ]; then
    warn "Uncommitted changes detected in $repo_dir — auto-committing all files:"
    printf '%s\n' "$status_output"
    while IFS= read -r line; do
      [ -n "$line" ] || continue
      case "$line" in
        *"This-Directory-Only/logs/"*) continue ;;
      esac
      git -C "$repo_dir" add -A -- "${line:3}"
    done <<< "$status_output"
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
rsync -a --delete --exclude '.vercel/' "$REPO_SHARED_DIR"/ "$DEPLOY_SHARED_DIR"/
python3 "$REFRESH_SPLIT_SCRIPT" --root "$DEPLOY_DIR"
auto_commit_if_dirty "$DEPLOY_DIR" "chore: sync shared deploy payload"

# --- Step 5: Push deploy to origin (triggers Vercel) ---

info "Pushing deploy to origin..."
git -C "$DEPLOY_DIR" push origin deploy

# --- Step 5b: Deploy frontend to Vercel from Local_Deployed_Shared ---

if command -v vercel >/dev/null 2>&1; then
  info "Deploying frontend to Vercel from Local_Deployed_Shared..."
  if [ ! -f "$DEPLOY_SHARED_DIR/.vercel/project.json" ]; then
    (
      cd "$DEPLOY_SHARED_DIR" && \
      vercel link --yes --project "$VERCEL_PROJECT" --scope "$VERCEL_SCOPE"
    )
  fi
  (
    cd "$DEPLOY_SHARED_DIR" && \
    vercel deploy --prod --yes --scope "$VERCEL_SCOPE"
  )
  if ! verify_vercel_frontend "$VERCEL_URL" 10 3; then
    warn "Primary Vercel alias did not serve successfully after deploy; retrying one forced frontend deploy..."
    (
      cd "$DEPLOY_SHARED_DIR" && \
      vercel deploy --prod --yes --force --scope "$VERCEL_SCOPE"
    )
    verify_vercel_frontend "$VERCEL_URL" 10 3 || {
      error "Vercel deploy completed, but $VERCEL_URL is still not serving the app."
      exit 1
    }
  fi
else
  warn "Vercel CLI not found — relying on Git-connected Vercel deploy."
fi

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
