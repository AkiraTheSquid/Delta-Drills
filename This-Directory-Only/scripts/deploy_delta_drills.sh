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
# 7. Republishes the Colab edition from the same tree (non-fatal)
# ============================================================

REPO_DIR="/home/stellar-thread/Applications/Delta-Drills-Local"
DEPLOY_DIR="/home/stellar-thread/Applications/Delta-Drills-Deployed"
REPO_THIS_DIR="$REPO_DIR/This-Directory-Only"
REPO_SHARED_DIR="$REPO_DIR/Local_Deployed_Shared"
DEPLOY_SHARED_DIR="$DEPLOY_DIR/Local_Deployed_Shared"
REFRESH_SPLIT_SCRIPT="$REPO_THIS_DIR/scripts/refresh_split_layout.py"
# The only interpreter with torch. Anything that runs learner code -- the
# exporter's expected_output recompute, the audit gate -- must use it.
BACKEND_PY="$REPO_THIS_DIR/backend/.venv/bin/python"
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

  # Stage everything in one call. `git add -A` respects .gitignore, so the
  # logs/ dir is already excluded via .gitignore line 18 and we don't need
  # an explicit pathspec. core.quotePath=false avoids octal-escaping of
  # non-ASCII filenames (the previous per-line loop choked on emoji-bearing
  # ARENA notebook filenames and explicit-pathspec attempts hit
  # 'ignored-files' warnings that exited non-zero under set -e).
  git -C "$repo_dir" -c core.quotePath=false add -A

  if ! git -C "$repo_dir" diff --cached --quiet; then
    warn "Auto-committing dirty changes in $repo_dir"
    git -C "$repo_dir" -c core.quotePath=false status --short
    git -C "$repo_dir" commit -m "$message"
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
# Assert the property that actually matters, not just that the file exists: a
# venv without torch fails exactly the same silent way bare python3 did.
# This belongs in preflight, ahead of Step 1 -- Step 1 auto-commits the working
# tree, so a guard placed any later aborts the deploy only after it has already
# written a commit.
if ! "$BACKEND_PY" -c "import torch" >/dev/null 2>&1; then
  error "$BACKEND_PY cannot import torch."
  error "The expected_output recompute would silently fall back to the CSV's"
  error "values and ship ~40 stdout questions no correct answer can match."
  error "Fix the backend venv before deploying."
  exit 1
fi

# --- Step 1: Check for uncommitted changes on main ---

info "Checking for uncommitted changes on main..."
git -C "$REPO_DIR" checkout main
auto_commit_if_dirty "$REPO_DIR" "chore: auto-commit before deploy"

# --- Step 2: Export question bank ---

# A leftover non-empty function_mode_broken_ids.json makes the export below
# silently EXCLUDE those ids from questions.json (including torch/Colab
# questions the offline validator simply can't run). Refuse to deploy on it.
BROKEN_IDS_FILE="$REPO_THIS_DIR/chatgpt/function_mode_broken_ids.json"
if [ -s "$BROKEN_IDS_FILE" ] && ! grep -qx '\[\]' "$BROKEN_IDS_FILE"; then
  error "Stale $BROKEN_IDS_FILE present — export would silently drop those ids."
  error "Inspect it, fix or dismiss the failures, then: rm '$BROKEN_IDS_FILE' and re-deploy."
  exit 1
fi

info "Exporting question bank artifacts..."
# MUST be the backend venv, not bare python3. The exporter rebuilds every
# expected_output from the CSV and then overwrites it with what the canonical
# answer actually prints under the grading harness. That recompute imports
# torch; _load_harness() swallows a failed preload_torch(), so without it every
# torch question errors, yields empty stdout, and falls into the
# "answer-errors kept stored value" branch -- which keeps the CSV's value, not
# the correction. The result is 40 stdout_prediction questions shipping
# expected outputs no right answer can match, silently, on every deploy.
# Step 2b already uses this interpreter for the audit gate; the exporter that
# FEEDS that gate has to run on it too.
"$BACKEND_PY" "$REPO_THIS_DIR/scripts/export_questions_json.py"
python3 "$REPO_THIS_DIR/scripts/extract_arena_prereqs.py"
python3 "$REPO_THIS_DIR/scripts/extract_arena_exercises.py"
# split_arena_exercises.py + build_arena_colab_index.py are intentionally NOT
# run here — see arena/README.md "Recent Changes" 2026-05-16 for why we
# unwired the per-exercise split-notebook path. The scripts and the
# arena-book-colab/ tree are kept on disk for archival reference.
python3 "$REFRESH_SPLIT_SCRIPT" --root "$REPO_DIR"

# The in-app Notebooks tab reads compiled JSON, not the .ipynb files the Colab
# edition publishes. Both are emitted by the same compiler
# (generate_colab_notebooks.build_notebook), and both embed the question bank
# exported above -- so a deploy that skips this ships a notebook teaching the
# PREVIOUS bank. There is no visible symptom: the tab renders perfectly and is
# simply out of date. Cheap (~2s, no torch), and it runs before the rsync at
# step 4 that mirrors Local_Deployed_Shared/ into the Deployed worktree.
info "Compiling web notebooks for the in-app Notebooks tab..."
python3 "$REPO_DIR/scripts/compile_web_notebooks.py"

# --- Step 2b: Hardened bank audit gate ---
# Blocks the deploy on gameable grading (bare-fixture cheats passing), broken
# starters, and degenerate expected values. See pipeline/audit_question_bank.py.
info "Auditing question bank (gameability gate)..."
"$BACKEND_PY" \
  "$REPO_SHARED_DIR/pipeline/audit_question_bank.py" --gate

# --- Step 2c: Grading-harness regression tests ---
# Torch tensor equality, rng seeding across setup re-exec, mech-gate
# non-degeneracy on tensors. See pipeline/test_torch_grading.py (codex
# cross-review 2026-07-11). Blocks the deploy on any regression.
info "Running grading-harness regression tests..."
"$BACKEND_PY" \
  "$REPO_SHARED_DIR/pipeline/test_torch_grading.py"

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

# --- Step 3c: Build the ARENA Jupyter Book ---

if [ -d "$REPO_DIR/arena-book" ]; then
  info "Building ARENA Jupyter Book..."
  bash "$REPO_THIS_DIR/scripts/build_arena_book.sh"
  auto_commit_if_dirty "$REPO_DIR" "chore: refresh arena-book build output"
else
  warn "arena-book/ not found — skipping Book build."
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
  # flyctl on this machine doesn't reliably pick up the access_token stored
  # in ~/.fly/config.yml — extract it into FLY_API_TOKEN so the deploy step
  # authenticates without an interactive `flyctl auth login`. The env var
  # only persists for the duration of this subshell.
  if [ -z "${FLY_API_TOKEN:-}" ] && [ -f "$HOME/.fly/config.yml" ]; then
    _fly_token=$(grep -E "^access_token:" "$HOME/.fly/config.yml" | sed 's/^access_token: *//')
    if [ -n "$_fly_token" ]; then
      export FLY_API_TOKEN="$_fly_token"
    fi
  fi
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

# --- Step 7: Keep the Colab edition on the same frontend ---
#
# delta-drills-colab.vercel.app is a SECOND Vercel project serving this exact
# frontend — one codebase, two deploys, the host decides whether a drill opens
# the editor or its notebook. That invariant only holds if both projects are
# published from the same tree, and nothing enforced it: this script deployed
# the main project and stopped, so every deploy left the fork one build behind
# and it stayed behind until someone happened to look.
#
# It is not a cosmetic lag. The fork ran for a week on a build whose knowledge
# graph read a learner model the backend no longer writes, so it drew 63 flat
# bubbles for an account with real practice history — the same symptom as a
# broken backend, on a frontend nobody had redeployed. Syncing here makes the
# fork's staleness impossible rather than merely visible.
#
# Non-fatal on purpose. The main deploy is already live by this point, and the
# colab script runs its own preflights (extension checks, notebook index audit)
# that can legitimately fail without saying anything about the deploy that just
# succeeded. A failure here means "the fork is stale", not "the release is bad".
COLAB_DEPLOY_SCRIPT="$REPO_THIS_DIR/scripts/deploy_delta_drills_colab.sh"
COLAB_DIR="/home/stellar-thread/Applications/Delta-Drills-Colab"
if [ -x "$COLAB_DEPLOY_SCRIPT" ] && [ -d "$COLAB_DIR" ]; then
  info "Syncing the Colab edition to this same frontend..."
  set +e
  bash "$COLAB_DEPLOY_SCRIPT"
  colab_status=$?
  set -e
  if [ "$colab_status" -ne 0 ]; then
    warn "Colab edition deploy FAILED (exit $colab_status) — the main deploy is live,"
    warn "but https://delta-drills-colab.vercel.app is still on the previous build."
    warn "Re-run: $COLAB_DEPLOY_SCRIPT"
  fi
else
  warn "Colab worktree or deploy script missing — skipping the Colab edition."
  warn "https://delta-drills-colab.vercel.app will keep serving its previous build."
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Deploy complete!${NC}"
echo -e "${GREEN}  Vercel:  ${VERCEL_URL}${NC}"
echo -e "${GREEN}  Colab:   https://delta-drills-colab.vercel.app${NC}"
echo -e "${GREEN}  Backend: https://delta-drills-backend.fly.dev${NC}"
echo -e "${GREEN}======================================${NC}"
