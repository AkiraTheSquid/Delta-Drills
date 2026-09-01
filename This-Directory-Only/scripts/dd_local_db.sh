#!/usr/bin/env bash
# ============================================================================
# dd_local_db.sh — the Postgres the LOCAL backend talks to.
#
# WHY THIS FILE EXISTS
#
# `backend/.env` has always pointed DATABASE_URL at localhost:54322, and for a
# long time nothing was listening there. The result was not an error anybody
# saw: `app/lifecycle.py` catches the failure ("never block startup on this")
# and logs a WARNING, so `uvicorn` came up, `/health` answered 200 — it only
# reads JSON files off disk — and the app looked fine. Every endpoint that
# touches the database answered 500 instead.
#
# From the browser that failure is invisible and mis-shaped. `/auth/signup`
# 500s, so guest-session.js gets no token; with no token getPracticeMode()
# answers "local"; in local mode runner.js has nowhere to send torch, so it
# refuses with TORCH_UNAVAILABLE. The learner sees "torch is not available in
# the browser sandbox" and concludes the SANDBOX is the problem. The sandbox
# is a symptom. The problem is that there is no database.
#
# So: one command that puts a real Postgres where .env already says one is,
# and a `status` that tells the truth about whether it is there.
#
# WHAT IT IS DELIBERATELY NOT
#
# Not Neon, and not the Fly volume. This is a scratch database for one
# machine — it holds throwaway accounts and drill attempts, and `reset`
# destroys all of it on purpose. Nothing here is a copy of production and
# nothing here should be treated as one.
#
# 🔴 EVERY CONNECTION PARAMETER IS READ OUT OF backend/.env, never hardcoded
# here. A second copy of the port is how this drifts back into the exact
# silent failure above: .env moves, the container keeps listening where it
# always did, and the backend is talking to nothing again while this script
# reports "running".
#
# Usage:
#   dd_local_db.sh up       start it (idempotent) and wait until it accepts
#                           connections. This is what the runner calls.
#   dd_local_db.sh status   is it there, and does .env agree with it?
#   dd_local_db.sh down     stop the container, keep the data
#   dd_local_db.sh reset    DESTROY the data and start clean (asks first)
#   dd_local_db.sh psql     open a psql shell on it
# ============================================================================
set -euo pipefail

# Derived from this file's own resolved location, never hardcoded: the entry
# point is a symlink in ~/.local/bin, and there is more than one checkout of
# this repo (Delta-Drills-Local, Delta-Drills-Deployed, agent worktrees).
# `readlink -f` follows the symlink to the real file, so a script always acts
# on the tree it actually lives in rather than on whichever one was hardcoded
# the day it was written.
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SELF")/../.." && pwd)"
ENV_FILE="$ROOT/This-Directory-Only/backend/.env"
CONTAINER="delta-drills-local-db"
VOLUME="delta_drills_local_pg"
IMAGE="postgres:16-alpine"

die() { echo "dd_local_db: $*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker not found — it is what hosts the local database"

# --- Read the connection out of .env -----------------------------------------
# The backend's own config (app/config.py) parses DATABASE_URL with SQLAlchemy,
# so parse it the same way rather than with a regex that would disagree about
# escaping the first time a password contains a '@'.
read_env_url() {
  [ -f "$ENV_FILE" ] || die "no $ENV_FILE — copy backend/.env.example to backend/.env first"
  python3 - "$ENV_FILE" <<'PY'
import sys
from urllib.parse import urlsplit, unquote

url = None
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if line.startswith("DATABASE_URL="):
        url = line.split("=", 1)[1].strip().strip('"').strip("'")
if not url:
    sys.exit("DATABASE_URL is not set in .env")

# postgresql+psycopg://user:pass@host:port/name -> drop the driver suffix so
# urlsplit sees a normal scheme.
scheme, _, rest = url.partition("://")
parts = urlsplit("//" + rest, scheme=scheme.split("+", 1)[0])
if parts.hostname not in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
    sys.exit(
        f"DATABASE_URL points at {parts.hostname!r}, which is NOT this machine. "
        "This script only ever manages a local scratch database; refusing to act "
        "on a remote one."
    )
print(f"DB_USER={unquote(parts.username or 'postgres')}")
print(f"DB_PASS={unquote(parts.password or 'postgres')}")
print(f"DB_PORT={parts.port or 5432}")
print(f"DB_NAME={(parts.path or '/postgres').lstrip('/')}")
PY
}

eval "$(read_env_url)"

# --- Helpers -----------------------------------------------------------------
container_state() {
  docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo "absent"
}

# The host port this container publishes Postgres on — the only thing that
# matters when deciding whether it still matches .env.
#
# Asks for 5432/tcp BY NAME rather than ranging over every mapping: a range
# concatenates the ports of a container that publishes more than one, with no
# separator, and "5433254322" compares unequal to everything forever. Read off
# .HostConfig.PortBindings rather than .NetworkSettings, because that is the
# creation-time binding and it survives the container being STOPPED — which is
# exactly the case cmd_up has to get right.
container_port() {
  docker inspect -f '{{with index .HostConfig.PortBindings "5432/tcp"}}{{(index . 0).HostPort}}{{end}}' \
    "$CONTAINER" 2>/dev/null || true
}

wait_ready() {
  for _ in $(seq 1 60); do
    if docker exec "$CONTAINER" pg_isready -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

create_container() {
  docker run -d \
    --name "$CONTAINER" \
    --restart unless-stopped \
    -e POSTGRES_USER="$DB_USER" \
    -e POSTGRES_PASSWORD="$DB_PASS" \
    -e POSTGRES_DB="$DB_NAME" \
    -p "127.0.0.1:${DB_PORT}:5432" \
    -v "$VOLUME:/var/lib/postgresql/data" \
    "$IMAGE" >/dev/null
}

# --- Commands ----------------------------------------------------------------
cmd_up() {
  local state published
  state="$(container_state)"

  # 🔴 The port check comes FIRST and applies to a stopped container as much as
  # to a running one. A port mapping is fixed when the container is created and
  # `docker start` cannot change it, so a stopped container that was built for
  # the old port starts perfectly, passes pg_isready INSIDE the container, and
  # reports ready while the backend connects to nothing on the new one. That is
  # the whole failure this script exists to prevent, wearing a green light.
  if [ "$state" != "absent" ]; then
    published="$(container_port)"
    if [ -n "$published" ] && [ "$published" != "$DB_PORT" ]; then
      echo "dd_local_db: container publishes :$published but .env wants :$DB_PORT — recreating"
      # The named volume is not touched by rm, so the data survives the move.
      docker rm -f "$CONTAINER" >/dev/null
      state="absent"
    fi
  fi

  if [ "$state" = "absent" ]; then
    echo "dd_local_db: creating $CONTAINER ($IMAGE) on 127.0.0.1:$DB_PORT"
    create_container
  elif [ "$state" != "running" ]; then
    echo "dd_local_db: starting $CONTAINER"
    docker start "$CONTAINER" >/dev/null
  fi

  wait_ready || die "$CONTAINER did not accept connections within 60s (docker logs $CONTAINER)"
  echo "dd_local_db: ready — postgres://$DB_USER@127.0.0.1:$DB_PORT/$DB_NAME"
}

cmd_status() {
  local state published
  state="$(container_state)"
  echo "container: $CONTAINER ($state)"
  echo ".env wants: 127.0.0.1:$DB_PORT/$DB_NAME as $DB_USER"
  if [ "$state" != "absent" ]; then
    published="$(container_port)"
    echo "publishes:  :${published:-?}"
    if [ "$published" != "$DB_PORT" ]; then
      echo "🔴 MISMATCH — the backend would be talking to nothing. \`up\` recreates it."
    fi
  fi
  if [ "$state" = "running" ]; then
    if docker exec "$CONTAINER" pg_isready -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; then
      echo "accepting:  yes"
    else
      echo "accepting:  no"
    fi
  fi
}

cmd_down() {
  [ "$(container_state)" = "absent" ] && { echo "dd_local_db: nothing to stop"; return 0; }
  docker stop "$CONTAINER" >/dev/null
  echo "dd_local_db: stopped (data kept in volume $VOLUME)"
}

cmd_reset() {
  # Destructive and irreversible, so it asks — even though everything in here
  # is throwaway, "throwaway" is a claim about the data, and the person who
  # typed this is the one who knows whether it still holds.
  echo "This DESTROYS the local database volume ($VOLUME): every local account,"
  echo "attempt and mastery record on this machine. Production is untouched."
  read -r -p "Type 'reset' to continue: " reply
  [ "$reply" = "reset" ] || { echo "dd_local_db: cancelled"; return 1; }
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker volume rm "$VOLUME" >/dev/null 2>&1 || true
  cmd_up
}

cmd_psql() {
  cmd_up >/dev/null
  docker exec -it "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME"
}

case "${1:-up}" in
  up)     cmd_up ;;
  status) cmd_status ;;
  down)   cmd_down ;;
  reset)  cmd_reset ;;
  psql)   cmd_psql ;;
  -h|--help)
    sed -n '/^# Usage:/,/^# ====/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
    ;;
  *) die "unknown command '${1}' (up|status|down|reset|psql)" ;;
esac
