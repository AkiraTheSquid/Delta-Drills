#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# deploy_delta_drills — one-command deploy for Delta Drills
#
# 1. Checks for uncommitted changes on main (warns + prompts)
# 2. Exports question bank to frontend/questions.json
# 3. Pushes main to origin
# 4. In the deploy worktree, merges main into deploy
# 5. Verifies no user data leaked into deploy tree
# 6. Pushes deploy to origin (triggers Vercel)
# ============================================================

REPO_DIR="/home/stellar-thread/Applications/pdf-split-tool"
DEPLOY_DIR="/home/stellar-thread/Applications/pdf-split-tool-deployed"
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
  warn "Uncommitted changes detected — auto-committing tracked files:"
  git -C "$REPO_DIR" status --short

  # Auto-commit tracked file changes; leave untracked files untouched.
  git -C "$REPO_DIR" add -u
  if ! git -C "$REPO_DIR" diff --cached --quiet; then
    git -C "$REPO_DIR" commit -m "chore: auto-commit before deploy"
  fi

  if git -C "$REPO_DIR" ls-files --others --exclude-standard | grep -q .; then
    warn "Untracked files were NOT committed:"
    git -C "$REPO_DIR" ls-files --others --exclude-standard
  fi
fi

# --- Step 2: Export question bank ---

info "Exporting question bank to frontend/questions.json..."
python3 "$REPO_DIR/scripts/export_questions_json.py"

# If the export created/updated questions.json, stage and commit it
if ! git -C "$REPO_DIR" diff --quiet frontend/questions.json 2>/dev/null || \
   git -C "$REPO_DIR" ls-files --others --exclude-standard | grep -q "frontend/questions.json"; then
  info "questions.json updated — auto-committing..."
  git -C "$REPO_DIR" add frontend/questions.json
  git -C "$REPO_DIR" commit -m "chore: update questions.json for deploy"
fi

# --- Step 3: Push main to origin ---

info "Pushing main to origin..."
git -C "$REPO_DIR" push origin main

# --- Step 4: Merge main into deploy worktree ---

info "Merging main into deploy branch..."
git -C "$DEPLOY_DIR" checkout deploy
git -C "$DEPLOY_DIR" merge main --no-edit

# --- Step 5: Verify no user data leaked ---

LEAKED=0
for pattern in "backend/user_data" "storage/jobs"; do
  if [ -d "$DEPLOY_DIR/$pattern" ] && [ "$(ls -A "$DEPLOY_DIR/$pattern" 2>/dev/null)" ]; then
    error "User data found in deploy tree: $pattern/"
    LEAKED=1
  fi
done

if [ -f "$DEPLOY_DIR/backend/.env" ]; then
  error "backend/.env found in deploy tree!"
  LEAKED=1
fi

if [ "$LEAKED" -eq 1 ]; then
  error "User data leaked into deploy tree. Aborting push."
  echo "  Fix .gitignore and run: git rm -r --cached <path>"
  exit 1
fi

info "No user data in deploy tree."

# --- Step 6: Push deploy to origin (triggers Vercel) ---

info "Pushing deploy to origin..."
git -C "$DEPLOY_DIR" push origin deploy

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Deploy complete!${NC}"
echo -e "${GREEN}  Vercel will auto-deploy from the${NC}"
echo -e "${GREEN}  deploy branch.${NC}"
echo -e "${GREEN}  ${VERCEL_URL}${NC}"
echo -e "${GREEN}======================================${NC}"
