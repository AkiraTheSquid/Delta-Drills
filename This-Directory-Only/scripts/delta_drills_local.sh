#!/usr/bin/env bash
# ============================================================================
# delta_drills_local.sh — the whole app, on this machine, with this machine's
# Python.
#
#   Frontend  http://localhost:5174   (Local_Deployed_Shared, served no-cache)
#   Backend   http://localhost:8000   (FastAPI + the .venv's real torch)
#   Database  127.0.0.1:54322         (dd_local_db.sh, per backend/.env)
#
# WHAT MAKES THE LOCAL APP "REAL" RATHER THAN SANDBOXED
#
# `app.js` sets defaultApiBase to http://localhost:8000 whenever the page is
# served from localhost, and `supabase-practice.js` getPracticeMode() answers
# "backend" for any localhost session holding a token. In backend mode
# `practice/runner.js` posts Run and Submit to /api/practice/run-code, which
# executes in a fork of THIS uvicorn process — the .venv's torch, this box's
# CPUs, no Pyodide anywhere. That is the difference the learner feels: torch
# drills run instead of being refused, and they run at native speed.
#
# The whole chain hangs off one thing: holding a token. No token means
# getPracticeMode() answers "local", local mode has no backend to run torch on,
# and runner.js refuses with TORCH_UNAVAILABLE — "not available in the browser
# sandbox". Which is why this script REFUSES TO CLAIM IT IS READY until it has
# proven the auth path works, and why it checks that against the database
# rather than against /health.
#
# The probe address uses guest-session.js's own @guest.delta-drills.app domain
# on purpose. The backend validates it with `email-validator` (schemas.py's
# EmailStr), which REJECTS the reserved TLDs — .invalid, .test, .example — with
# a 422 before any database query runs. A probe addressed there never reaches
# the code path it is trying to measure, and reports a broken auth path on a
# perfectly healthy one.
#
# 🔴 /health IS NOT A READINESS CHECK FOR THIS APP. It reads JSON off disk and
# answers 200 with the database completely gone — `app/lifecycle.py` catches a
# failed schema bootstrap on purpose ("never block startup on this") and only
# logs a warning. That is exactly how the local app spent a long time looking
# up while every account, drill and mastery write 500'd. The probe below posts
# credentials that do not exist and requires 401 (the query ran and found
# nobody). A 500 there means the database is not answering, and this script
# stops and says so instead of handing over a URL that will waste an hour.
#
# Ctrl+C stops everything it started. The database container is left running
# on purpose — it is cheap, it holds your local progress, and the next start
# is instant. `dd_local_db.sh down` stops it; `reset` wipes it.
# ============================================================================
set -euo pipefail

# Derived from this file's own resolved location, never hardcoded. The entry
# point is a symlink in ~/.local/bin, and there is more than one checkout of
# this repo (Delta-Drills-Local, Delta-Drills-Deployed, agent worktrees) — a
# hardcoded path means `cd`-ing into a worktree and running the local runner
# silently starts the OTHER tree's code, which is a whole afternoon of editing
# files the running app never loads.
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SELF")/../.." && pwd)"
THIS_DIR_ONLY="$ROOT/This-Directory-Only"
SHARED_DIR="$ROOT/Local_Deployed_Shared"
BACKEND_DIR="$THIS_DIR_ONLY/backend"
DB_SCRIPT="$THIS_DIR_ONLY/scripts/dd_local_db.sh"
# 🔴 Keyed to THIS checkout. It used to be a single fixed path, which every
# checkout on this box shared — Delta-Drills-Local and Delta-Drills-Deployed
# and any agent worktree all wrote the same file, so starting one told the
# next one to kill "its" old session, and that pid was a stranger's.
PID_FILE="/tmp/delta_drills_local.$(printf '%s' "$ROOT" | cksum | cut -d' ' -f1).pid"
BACKEND_LOG="/tmp/delta_drills_local_backend.log"

FRONTEND_PORT="${DELTA_DRILLS_FRONTEND_PORT:-5174}"
BACKEND_PORT="${DELTA_DRILLS_BACKEND_PORT:-8000}"
BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"

# 🔴 The browser does not read this variable. app.js hardcodes
# `http://localhost:8000` as its local API base, so moving the backend moves
# the server and this script's probes but NOT the page: every probe here would
# pass on the new port while the page talked to nothing on 8000, lost its
# token, fell back to Pyodide and refused every torch drill as "not available
# in the browser sandbox" — a green banner over the exact failure this script
# exists to prevent. There is a per-browser override, so the escape hatch is
# real; it just has to be taken deliberately.
if [ "$BACKEND_PORT" != "8000" ] && [ "${DELTA_DRILLS_API_BASE_SET:-}" != "1" ]; then
  echo "delta_drills_local: DELTA_DRILLS_BACKEND_PORT=$BACKEND_PORT, but app.js" >&2
  echo "  points the page at http://localhost:8000 regardless. Run this in the" >&2
  echo "  browser console first, then set DELTA_DRILLS_API_BASE_SET=1:" >&2
  echo "    localStorage.setItem('api_base', 'http://localhost:$BACKEND_PORT')" >&2
  exit 1
fi

die() { echo "delta_drills_local: $*" >&2; exit 1; }

# Is this pid one of THIS checkout's own servers?
#
# Two independent marks, either of which is enough, because the two servers
# look nothing alike: the backend is `python -m uvicorn app.main:app` started
# with its cwd inside this tree, and the frontend is `python serve.py <port>`
# whose script path is inside this tree and whose cwd is Local_Deployed_Shared.
# Both tests are anchored to $ROOT, so a uvicorn belonging to a DIFFERENT
# checkout of this same repo reads as foreign — which is what we want; it is
# someone else's work either way.
is_ours() {
  local pid="$1" cwd cmdline
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"

  # The frontend identifies itself completely: an absolute path to OUR serve.py.
  case "$cmdline" in *"$THIS_DIR_ONLY/serve.py"*) return 0 ;; esac

  # A previous run of THIS launcher. The pid file holds the launcher's own pid
  # (it is what the trap cleans up after), not a server's, so without this
  # branch the tightened test below would refuse to stop the session it just
  # started and every restart would leave the old one running. Matching the
  # name alone would be too loose — another checkout has a script with the
  # same name — so each token that looks like the launcher is resolved and
  # required to be this exact file.
  case "$cmdline" in
    *delta_drills_local*)
      local tok
      for tok in $cmdline; do
        case "$tok" in
          *delta_drills_local*)
            [ "$(readlink -f "$tok" 2>/dev/null)" = "$SELF" ] && return 0
            ;;
        esac
      done
      ;;
  esac

  # The backend needs both marks. cwd alone is too loose — a shell, an editor,
  # a watcher or another session's throwaway `python -m http.server` launched
  # from anywhere in this checkout would match it, and this function's answer
  # is used to decide whether to KILL something. Requiring the uvicorn target
  # as well means the only processes that qualify are the server this script
  # itself starts.
  case "$cmdline" in
    *uvicorn*app.main:app*)
      cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
      case "$cwd" in "$ROOT"/*|"$ROOT") return 0 ;; esac
      ;;
  esac
  return 1
}

stop_existing() {
  local pid=""
  if [ -f "$PID_FILE" ]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  fi

  # 🔴 A pid in a file is a CLAIM, not a fact. The file outlives crashes, and
  # Linux reuses pids, so an old file plus an unlucky wrap means this number
  # now belongs to somebody else's process — and the next two lines would TERM
  # and then KILL it. The ports below have been checked for ownership since the
  # first review; this path had not, which is the same bug with a nicer name.
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && ! is_ours "$pid"; then
    echo "delta_drills_local: $PID_FILE names pid $pid, which is not this app:" >&2
    echo "  $(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-160)" >&2
    echo "  Leaving it alone and discarding the stale file." >&2
    pid=""
  fi

  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "Stopping existing Delta Drills session (pid $pid)..."
    kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..10}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"

  # Free the two ports — but ONLY from processes that are this app.
  #
  # 🔴 Never kill by port alone. Several Claude sessions work in this repo at
  # once (see `collab`), and :8000 is the most ordinary port there is. The
  # previous version of this script was worse still: it ran `pgrep -f "$ROOT"`
  # and TERM'd everything whose command line merely mentioned the checkout —
  # another session's uvicorn, a watcher, an editor. Reclaiming a port is a
  # convenience; taking down someone else's work to get it is not a trade this
  # script may make on its own. So each listener is identified first, and one
  # that is not ours stops the script with its pid and command line instead.
  for port in "$FRONTEND_PORT" "$BACKEND_PORT"; do
    local pids pid owned
    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    [ -n "$pids" ] || continue

    owned=""
    for pid in $pids; do
      if is_ours "$pid"; then
        owned="$owned $pid"
      else
        echo "delta_drills_local: :$port is held by a process that is not this app" >&2
        echo "  pid $pid: $(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-160)" >&2
        echo "  Refusing to kill it. Stop it yourself, or pick another port with" >&2
        echo "  DELTA_DRILLS_FRONTEND_PORT / DELTA_DRILLS_BACKEND_PORT." >&2
        exit 1
      fi
    done

    [ -n "$owned" ] || continue
    kill -TERM $owned 2>/dev/null || true
    for _ in {1..6}; do
      sleep 0.5
      lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 || break
    done
    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    for pid in $pids; do
      is_ours "$pid" && kill -KILL "$pid" 2>/dev/null || true
    done
    sleep 0.5
  done
}

# --- Preflight ---------------------------------------------------------------
[ -d "$ROOT" ]        || die "path not found: $ROOT"
[ -d "$BACKEND_DIR" ] || die "backend not found: $BACKEND_DIR"
[ -d "$SHARED_DIR" ]  || die "frontend not found: $SHARED_DIR"
[ -x "$BACKEND_DIR/.venv/bin/python3" ] || die "no backend venv at $BACKEND_DIR/.venv"

# torch is not optional here — it is the entire reason to run locally instead
# of against Fly. Check it before starting anything, so a missing wheel is one
# clear line rather than a drill that mysteriously refuses to run later.
if ! "$BACKEND_DIR/.venv/bin/python3" -c "import torch" >/dev/null 2>&1; then
  die "the backend venv cannot import torch — install it with:
  $BACKEND_DIR/.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch"
fi

# STORAGE_DIR is created lazily by some writers and assumed by others; make it
# once here so neither has to care. Resolved against BACKEND_DIR, not against
# the caller's cwd: uvicorn is started after a `cd` into the backend, so a
# relative value in .env means one directory to the backend and a different one
# to this script, and the one this script made would sit empty next to wherever
# the user happened to be standing.
STORAGE_DIR="$(grep -E '^STORAGE_DIR=' "$BACKEND_DIR/.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"' || true)"
if [ -n "$STORAGE_DIR" ]; then
  case "$STORAGE_DIR" in
    /*) ;;
    *) STORAGE_DIR="$BACKEND_DIR/$STORAGE_DIR" ;;
  esac
  mkdir -p "$STORAGE_DIR"
fi

# Two launches of this same checkout, seconds apart, used to be able to kill
# each other's shell and delete each other's pid file, leaving orphaned
# servers behind and no record of them. Everything from "stop the old session"
# to "record the new one" is one critical section, held per checkout.
LOCK_FILE="${PID_FILE%.pid}.lock"
exec 9>"$LOCK_FILE"
flock -w 60 9 || die "another delta_drills_local is starting up in this checkout
  and did not finish within 60s. Check $PID_FILE and $BACKEND_LOG."

stop_existing

echo "Starting the local database..."
[ -x "$DB_SCRIPT" ] || die "missing $DB_SCRIPT"
"$DB_SCRIPT" up

echo "Starting Delta Drills..."
echo "$$" > "$PID_FILE"
exec 9>&-   # critical section over: the takeover is recorded

cleanup() {
  # Only if it still names US. A newer launch that took over has already
  # written its own pid here, and removing that would hide a running session
  # from the next start, which would then not stop it.
  [ "$(cat "$PID_FILE" 2>/dev/null || true)" = "$$" ] && rm -f "$PID_FILE"
  [ -n "${backend_pid:-}" ]  && kill -TERM "$backend_pid"  2>/dev/null || true
  [ -n "${frontend_pid:-}" ] && kill -TERM "$frontend_pid" 2>/dev/null || true
  # This runs from a trap, so its return value can become the script's exit
  # status. "the pid file was not ours" is not a failure.
  return 0
}
trap cleanup EXIT INT TERM

(
  cd "$BACKEND_DIR"
  exec .venv/bin/python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port "$BACKEND_PORT"
) > "$BACKEND_LOG" 2>&1 &
backend_pid=$!

(
  cd "$SHARED_DIR"
  exec python3 "$THIS_DIR_ONLY/serve.py" "$FRONTEND_PORT"
) &
frontend_pid=$!

# --- Prove the auth path works before saying "ready" -------------------------
# A login for an account that does not exist. 401 means the request reached
# Postgres, ran a query and found nobody — the whole chain the app needs. 500
# means the database is not answering, which is the failure that looks like a
# working app right up until the first drill.
probe_auth() {
  curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
    -X POST "$BACKEND_URL/auth/login" \
    -H 'Content-Type: application/json' \
    -d '{"email":"preflight@guest.delta-drills.app","password":"not-a-real-password"}' \
    2>/dev/null || echo "000"
}

code="000"
for _ in $(seq 1 60); do
  kill -0 "$backend_pid" 2>/dev/null || die "the backend exited during startup — see $BACKEND_LOG"
  code="$(probe_auth)"
  [ "$code" = "401" ] && break
  sleep 1
done

if [ "$code" != "401" ]; then
  echo >&2
  echo "🔴 The backend is up but its AUTH PATH IS NOT WORKING (POST /auth/login -> $code)." >&2
  echo >&2
  if [ "$code" = "422" ]; then
    echo "   A 422 is this SCRIPT's probe being rejected before it reaches the" >&2
    echo "   database — the app is probably fine. Check the probe address against" >&2
    echo "   backend/app/schemas.py; email-validator refuses reserved TLDs." >&2
  elif [ "$code" = "500" ]; then
    echo "   A 500 here is the database. /health will still answer 200 — it only reads" >&2
    echo "   files — so do not trust it. Check:  $DB_SCRIPT status" >&2
  elif [ "$code" = "000" ]; then
    echo "   Nothing answered on $BACKEND_URL at all. See $BACKEND_LOG." >&2
  fi
  echo >&2
  echo "   Not handing you a URL: with auth broken the frontend gets no token," >&2
  echo "   drops to Pyodide, and every torch drill is refused as 'not available" >&2
  echo "   in the browser sandbox'. That message would be blaming the wrong thing." >&2
  exit 1
fi

# --- Prove torch really runs HERE --------------------------------------------
# Cheap, and it is the claim this whole script exists to make. Signs up a
# throwaway local account (the local database is scratch) and runs a tensor
# through the same endpoint the Run button uses.
verify_torch() {
  python3 - "$BACKEND_URL" <<'PY'
import json, sys, urllib.request, urllib.error, uuid

base = sys.argv[1]

def post(path, payload, token=None):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

try:
    # Same domain rule as the probe above: a reserved TLD is refused by
    # EmailStr with a 422 and never reaches the runner.
    creds = {"email": f"preflight-{uuid.uuid4().hex[:12]}@guest.delta-drills.app",
             "password": uuid.uuid4().hex}
    token = post("/auth/signup", creds)["access_token"]
    out = post("/api/practice/run-code",
               {"code": "import torch, os\nprint(torch.__version__)\nprint(os.cpu_count())"},
               token)
    lines = (out.get("stdout") or "").split()
    if not lines:
        sys.exit(f"run-code returned no output: {out}")
    print(f"{lines[0]} on {lines[1]} cpus")
except Exception as exc:  # noqa: BLE001 — the message is the whole point
    sys.exit(str(exc))
PY
}

torch_line="$(verify_torch 2>&1)" || {
  echo >&2
  echo "🔴 Auth works but torch did not run through /api/practice/run-code:" >&2
  echo "   $torch_line" >&2
  echo "   See $BACKEND_LOG." >&2
  exit 1
}

# --- Sign in with Google: report it, never block on it ------------------------
# Guest mode does not need Google, so a failure here is a WARNING and the app
# still starts. It is checked anyway because every way this breaks is silent:
# the button simply does not appear, or sign-in fails with a message that
# blames the learner's account instead of the config.
#
# Three separate things have to line up, and each has its own failure text:
#   1. the backend knows the client id      -> 503 "not configured" if not
#   2. it is the SAME id the frontend uses  -> "Invalid Google token" if not,
#      which reads like a bad password rather than a mismatched audience
#   3. Google has THIS ORIGIN registered    -> the button 403s and never draws
# 🔴 (3) lives in the Google Cloud console and NOWHERE in this repo, so nothing
# here can test it by reading files — it has to be asked over the network.
check_google() {
  local cfg_id env_id code origin
  cfg_id="$(sed -n 's/^window\.GOOGLE_CLIENT_ID *= *"\(.*\)";$/\1/p' \
    "$ROOT/Local_Deployed_Shared/auth-config.js" 2>/dev/null)"
  env_id="$(sed -n 's/^GOOGLE_CLIENT_ID=//p' "$BACKEND_DIR/.env" 2>/dev/null | tail -1)"

  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
    -X POST "$BACKEND_URL/auth/google" \
    -H 'Content-Type: application/json' \
    -d '{"credential":"preflight-not-a-real-token"}' 2>/dev/null || echo "000")"

  if [ "$code" = "503" ]; then
    echo "⚠️  Sign in with Google is OFF on the backend (/auth/google -> 503)."
    echo "    Add to $BACKEND_DIR/.env:  GOOGLE_CLIENT_ID=$cfg_id"
    echo "    Guest mode is unaffected; only the sign-in button is dead."
    return 0
  elif [ "$code" != "401" ]; then
    echo "⚠️  /auth/google answered $code to a junk token (expected 401)."
    return 0
  fi

  if [ -n "$cfg_id" ] && [ -n "$env_id" ] && [ "$cfg_id" != "$env_id" ]; then
    echo "🔴 Google client id MISMATCH — sign-in will fail as 'Invalid Google token',"
    echo "    which looks like a rejected account and is not one:"
    echo "      auth-config.js: $cfg_id"
    echo "      backend .env:   $env_id"
    return 0
  fi

  # Ask Google whether it will draw a button for the origin we are about to
  # serve. 200 = registered, 403 = not registered OR saved within the last few
  # hours and still propagating ("5 minutes to a few hours", Google's wording).
  origin="http://localhost:${FRONTEND_PORT}"
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
    -H "Origin: $origin" -H "Referer: $origin/" \
    "https://accounts.google.com/gsi/button?client_id=${cfg_id}&iframe_id=preflight" \
    2>/dev/null || echo "000")"
  case "$code" in
    200) : ;;  # registered; nothing to say
    000) echo "⚠️  Could not reach Google to check the sign-in origin (offline?)." ;;
    *)
      echo "⚠️  Google will not draw the sign-in button on $origin (HTTP $code)."
      echo "    Add it under Authorized JavaScript origins on client"
      echo "    ${cfg_id%%.*}… — console.cloud.google.com → Google Auth Platform"
      echo "    → Clients. Console only; there is no API for it. Already added?"
      echo "    Google's own note says 5 minutes to a few hours to take effect."
      echo "    Guest mode and every drill work regardless."
      ;;
  esac
}

check_google

cat <<EOF

Delta Drills is running LOCALLY:
  Frontend:  http://localhost:${FRONTEND_PORT}
  Backend:   http://localhost:${BACKEND_PORT}   (torch ${torch_line})
  Database:  $("$DB_SCRIPT" status | sed -n '2p' | sed 's/^.env wants: //')
  Backend log: $BACKEND_LOG

  Torch drills execute on THIS machine, not in the browser sandbox — signed in
  or not. Open the frontend on localhost (not 0.0.0.0, not the LAN address):
  app.js only points at the local backend for localhost/127.0.0.1.

Press Ctrl+C to stop.
EOF

wait "$backend_pid" "$frontend_pid"
