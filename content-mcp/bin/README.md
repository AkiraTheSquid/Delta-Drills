# bin

## Purpose
- The two entry points, and the one decision they both make: which Python runs
  the content tools.

## Owns
- `dd-content`: the CLI a person or a script calls.
- `dd-content-mcp`: the stdio MCP server an MCP client launches. This is the
  command registered in the repo's `.mcp.json`.

## Does NOT own
- Any behaviour. Both scripts are three lines of interpreter selection and an
  `exec` into `content_mcp`.

## Key Files
- `dd-content` → `python -m content_mcp.cli`
- `dd-content-mcp` → `python -m content_mcp.server`

## Data & External Dependencies
- Prefers `This-Directory-Only/backend/.venv/bin/python3`, falling back to
  whatever `python3` is on PATH. Both set `PYTHONPATH` to the package root so
  the package resolves without being installed.

## How It Works (Flow)
1. Resolve the repo root from the script's own location — so a symlink into
   `~/.local/bin` still finds the right checkout.
2. Pick the backend venv interpreter if it exists.
3. `exec` the module, replacing the shell rather than wrapping it, so signals
   and the stdio streams pass through untouched.

## Invariants & Constraints
- 🔴 `dd-content-mcp` must print NOTHING to stdout. The JSON-RPC stream is
  stdout; a `set -x`, an `echo`, or a venv activation banner here breaks every
  client with a parse error.
- `exec`, not a plain call: an MCP client kills the server by closing stdin and
  signalling the process it spawned.
- The backend venv is preferred because the pipeline steps need torch, but the
  read and edit tools work under any Python 3.9+ — a contributor without the
  venv still gets a working server.

## Extension Points
- A new entry point is a copy of either script with a different `-m` target.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **A banner on stdout kills the MCP server** — `ACTIVE (risk)`
  - When it happens: someone adds an echo, a `set -x`, or sources a venv
    activate script in `dd-content-mcp`.
  - Symptom: the client reports a JSON parse error and lists zero tools; the
    server itself looks fine when run by hand.
  - Root cause: stdout is the transport.
  - Prevention/fix: diagnostics go to stderr (`>&2`). `../watch.py` handshakes
    the built server and would fail on a non-JSON first line.
  - Status: `ACTIVE` as a standing risk.

## Recent Changes
- 2026-09-02: Both entry points created.
