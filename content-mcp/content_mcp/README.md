# content_mcp

## Purpose
- The Python package behind the content tools: what an "edit to the course
  content" is, and what has to be true before one lands.

## Owns
- The op registry and its dispatch (`ops.py`).
- The three content layers — KP pages (`lessons.py`), the concept graph
  (`graph.py`), the drill bank (`drills.py`).
- The write gate (`auth.py`) and the rolling snapshot (`backup.py`).
- The two front ends (`server.py`, `cli.py`) and the repo layout (`paths.py`).

## Does NOT own
- Any content file, any validation rule, any deploy step. See the parent
  folder's README for the full boundary.

## Key Files
- `ops.py`: the registry. `@op(name, summary, params, required, needs_auth)`
  around a handler is the whole contract; both front ends generate themselves
  from it. `ops.call` is the ONE place auth is demanded and the snapshot taken,
  so no handler can forget either.
- `paths.py`: every path in the repo this package touches, plus
  `python_for_content()` — the backend venv, because bare `python3` has no torch.
- `lessons.py`: KP pages, addressed by KC id and never by path. Parsing is
  delegated to the repo's `scripts/lesson_lib.py`.
- `graph.py`: `kc_registry.json`. `check()` runs before every write and refuses
  a registry that has a cycle, an unknown prereq, or a concept sequenced before
  something it needs.
- `drills.py`: the bank. Reads the compiled artifact, writes only the CSV and
  the override JSONL. `next_id()` counts CSV rows because ids are positional.
- `pipeline.py`: subprocess wrappers around the repo's own validators.
- `server.py`: MCP over stdio, no dependencies.
- `cli.py`: `dd-content`, argparse generated from the registry.

## Data & External Dependencies
- Standard library only. No third-party package is imported anywhere here —
  that is deliberate, so a contributor can run this straight after `git clone`.
- Two repo modules are imported lazily rather than copied: `lesson_lib`
  (`scripts/`) and `export_questions_json` (`Local_Deployed_Shared/pipeline/`).

## How It Works (Flow)
1. A front end parses a request into `(name, arguments)`.
2. `ops.call(name, arguments)` pops `token`, demands auth if the op writes,
   calls `backup.ensure()`, checks required args, and invokes the handler.
3. The handler validates structurally and writes the real source file.
4. `pipeline.*` runs the repo's gates when the caller asks.

## Invariants & Constraints
- 🔴 `server.py` must never write to stdout except JSON-RPC frames.
- 🔴 Ids are positional; never insert or delete a CSV row (see the parent README).
- Auth and the snapshot live in `ops.call`, not in handlers. A handler that
  writes without going through `ops.call` bypasses both.
- Only fields in `drills.OVERRIDE_FIELDS` may be written to an override record.
  The exporter silently IGNORES anything else, so an unlisted field would look
  like a successful edit that changed nothing — `drills.update` raises instead.
- The override-layer ORDER is the exporter's business, not this package's.

## Extension Points
- New tool: one `@op` in `ops.py`. Add it to `WRITE_OPS` in `../watch.py` if it
  writes, or the watcher fails — which is the point.
- New content layer: a module here, then ops that call it.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **A silently ignored override field** — `RESOLVED`
  - When it happens: an edit sets a drill field the exporter does not whitelist.
  - Symptom: the tool reports success, the bank is unchanged, and nothing errors.
  - Root cause: the exporter's merge only replaces whitelisted keys.
  - Prevention/fix: `drills.update` raises on any field outside
    `OVERRIDE_FIELDS` and names the allowed set.
  - Status: `RESOLVED`.

## Recent Changes
- 2026-09-02: Package created.
