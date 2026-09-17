#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# deploy_delta_drills_colab — publish the COLAB EDITION.
#
# A second Vercel project serving the same frontend as delta-drills.vercel.app.
# The code is byte-identical; `Local_Deployed_Shared/practice/colab_mode.js`
# asks `location.hostname` which deploy it is, and on this host a drill routes
# to its published lesson notebook instead of the in-page editor.
#
#   delta-drills.vercel.app        solve in the editor
#   delta-drills-colab.vercel.app  solve in the notebook  ← this script
#
# Same Fly backend, same Supabase, same question bank, so practice on either
# moves the same mastery record. There is deliberately NO separate content
# pipeline: this syncs whatever `Local_Deployed_Shared/` currently holds.
#
# WHAT THIS DOES NOT DO, on purpose:
#   - It does not re-export the question bank and does not run the audit gate.
#     Both belong to `deploy_delta_drills.sh`, which owns the bank. Change the
#     bank → run that first, then this. Running an export here would let the
#     two deploys disagree about what a question is.
#   - It does not touch Fly. The backend is shared and unchanged by this deploy.
#   - It does not push `main`. Only the `deploy-colab` branch moves.
# ============================================================

REPO_DIR="/home/stellar-thread/Applications/Delta-Drills-Local"
COLAB_DIR="/home/stellar-thread/Applications/Delta-Drills-Colab"
REPO_SHARED_DIR="$REPO_DIR/Local_Deployed_Shared"
COLAB_SHARED_DIR="$COLAB_DIR/Local_Deployed_Shared"
REFRESH_SPLIT_SCRIPT="$REPO_DIR/This-Directory-Only/scripts/refresh_split_layout.py"
LOG_DIR="$REPO_DIR/This-Directory-Only/logs"
TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
LOG_FILE="$LOG_DIR/deploy_delta_drills_colab-$TIMESTAMP.txt"
BRANCH="deploy-colab"
VERCEL_URL="https://delta-drills-colab.vercel.app"
VERCEL_PROJECT="delta-drills-colab"
if [ -z "${VERCEL_TOKEN:-}" ] && [ -f "$HOME/.config/vercel/token.env" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.config/vercel/token.env"
fi
VERCEL_SCOPE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["orgId"])' "$COLAB_SHARED_DIR/.vercel/project.json")"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[colab-deploy]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*"; }

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
info "Log: $LOG_FILE"

if [ ! -d "$COLAB_DIR" ]; then
  error "Missing worktree $COLAB_DIR. Create it once with:"
  error "  git -C $REPO_DIR worktree add $COLAB_DIR $BRANCH"
  exit 1
fi

# Step 2 rsyncs $REPO_SHARED_DIR — the WORKING TREE, not a commit — into the
# colab worktree and publishes it. Anything another session has in flight there
# would ship to the fork's public URL regardless of whether it was committed.
# Same gate the main deploy grew on 2026-08-22, for the same reason.
repo_dirty="$(git -C "$REPO_DIR" -c core.quotePath=false status --porcelain)"
if [ -n "$repo_dirty" ]; then
  error "Working tree at $REPO_DIR is not clean. Refusing to publish the Colab edition."
  error "The sync below copies the working tree, so these paths would ship:"
  echo "$repo_dirty" >&2
  exit 1
fi

# --- 1. Preflight: the extension must still point at this deploy ---
# The extension frames this URL and its toolbar button opens it. Shipping a
# frontend the extension cannot use is the failure this fork exists to avoid.
info "Preflight: extension checks..."
python3 "$REPO_DIR/extension/watch.py"

# Every question this deploy will route to Colab must exist in the notebook
# PUBLISHED on GitHub. Routing hides the editor and the submit bar, so a question
# whose cell was never published is a dead end wearing the face of a feature.
info "Preflight: notebook index vs published notebooks..."
python3 "$REPO_DIR/This-Directory-Only/scripts/audit_colab_notebook_index.py"

# --- 2. Sync the shared frontend into the colab worktree ---
info "Syncing Local_Deployed_Shared into $BRANCH..."
git -C "$COLAB_DIR" checkout "$BRANCH"
mkdir -p "$COLAB_SHARED_DIR"
# --exclude .vercel/: that directory is this worktree's link to its OWN Vercel
# project. Copying the main deploy's link over it would publish the colab build
# to delta-drills.vercel.app — the one mistake in this script that would be
# invisible until someone noticed the main site had changed.
rsync -a --delete --exclude '.vercel/' "$REPO_SHARED_DIR"/ "$COLAB_SHARED_DIR"/
python3 "$REFRESH_SPLIT_SCRIPT" --root "$COLAB_DIR"

changed_paths=()
mapfile -d '' -t changed_paths < <(
  git -C "$COLAB_DIR" ls-files --modified --deleted --others --exclude-standard -z
)
if [ "${#changed_paths[@]}" -gt 0 ]; then
  git -C "$COLAB_DIR" add -- "${changed_paths[@]}"
fi
if git -C "$COLAB_DIR" diff --cached --quiet; then
  info "No frontend changes to commit."
else
  git -C "$COLAB_DIR" commit -q -m "chore: sync colab-edition frontend payload"
  info "Committed the synced payload."
fi
git -C "$COLAB_DIR" push -q origin "$BRANCH"

# --- 3. Deploy to Vercel ---
if ! command -v vercel >/dev/null 2>&1; then
  error "Vercel CLI not found — this fork has no git-connected deploy to fall back on."
  exit 1
fi

if [ ! -f "$COLAB_SHARED_DIR/.vercel/project.json" ]; then
  info "Linking $COLAB_SHARED_DIR to project $VERCEL_PROJECT..."
  (cd "$COLAB_SHARED_DIR" && vercel link --yes --project "$VERCEL_PROJECT" --scope "$VERCEL_SCOPE")
fi

info "Deploying frontend to $VERCEL_PROJECT..."
(cd "$COLAB_SHARED_DIR" && vercel deploy --prod --yes --scope "$VERCEL_SCOPE")

# --- 4. Verify: serving, AND actually the Colab edition ---
# HTTP 200 alone would pass on a deploy that lost colab_mode.js and quietly
# served the ordinary app under the fork's name — which looks identical until a
# torch drill opens an editor that cannot import torch.
info "Verifying $VERCEL_URL ..."
for attempt in $(seq 1 10); do
  status="$(curl -s -o /tmp/dd-colab-index.html -w '%{http_code}' "$VERCEL_URL" || true)"
  [ "$status" = "200" ] && break
  warn "attempt $attempt/10: HTTP $status"
  sleep 3
done
if [ "$status" != "200" ]; then
  error "$VERCEL_URL is not serving (HTTP $status)."
  exit 1
fi
if ! grep -q 'practice/colab_mode.js' /tmp/dd-colab-index.html; then
  error "$VERCEL_URL served a build with no colab_mode.js — it would behave as the normal app."
  exit 1
fi

# Sign-in is keyed on the JavaScript ORIGIN, so this deploy's hostname has to be
# registered on the OAuth client before the Google button works — the failure is
# "Error 400: origin_mismatch" at the moment of pressing it, long after a deploy
# that verified green. Cheap to check here, so check it.
if ! grep -q "$VERCEL_URL" "$REPO_SHARED_DIR/auth-config.js"; then
  warn "auth-config.js does not mention $VERCEL_URL."
  warn "Google sign-in will fail with 'Error 400: origin_mismatch' until that"
  warn "origin is added to the OAuth client's Authorized JavaScript origins."
fi

info "Done."
echo -e "${GREEN}  Colab edition: ${VERCEL_URL}${NC}"
echo -e "${GREEN}  Normal app:    https://delta-drills.vercel.app (unchanged)${NC}"
echo -e "${YELLOW}  Sign-in needs ${VERCEL_URL} in the OAuth client's JavaScript origins.${NC}"
