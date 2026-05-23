# scripts

## Purpose
CLI tooling that operates on `../vocab/` and `../exercises/`: validate atom references, validate the prereq DAG, and export the graph to yEd-readable GraphML.

## Owns
- All three scripts are stdlib-only, executable from anywhere, and return non-zero exit on failure.
- Each exposes a `main()` returning an int — runnable both as `python <script>.py` and via `import` from `../watch.py`.

## Does NOT own
- The data — lives in `../vocab/` (atoms + edges) and `../exercises/` (per-notebook tags).
- The visual layout — yEd does that after import; we only emit a well-formed GraphML.

## Key Files
- `validate.py`: every atom id referenced by an exercise must exist in `vocab/atoms.json`. Hard error on missing refs.
- `validate_graph.py`: full prereq-DAG validation. Hard errors: missing endpoints, self-loops, duplicate edges, cycles (Kahn). Soft warnings: orphan atoms, transitive-reduction candidates.
- `export_graphml.py`: emits `../concept-graph.graphml` with yFiles visual hints — pastel color per domain, kind→edge-style mapping, status→line-weight. Open in yEd and run Layout → Hierarchical.

## Data & External Dependencies
- Reads `../vocab/atoms.json`, `../vocab/prereqs.json`, and all `../exercises/0_*.json`.
- Writes `../concept-graph.graphml` (exporter only).
- Stdlib only (`json`, `xml.sax.saxutils`, `colorsys`, `hashlib`, `collections`, `pathlib`).

## How It Works (Flow)
1. `validate.py` — load vocab, walk every exercise, collect any atom id not in vocab; exit 1 if any.
2. `validate_graph.py` — load vocab + prereqs, run endpoint/self-loop/dup checks, Kahn-toposort the directed graph, then print soft warnings.
3. `export_graphml.py` — emit one `<node>` per atom (with `y:ShapeNode`) and one `<edge>` per prereq (with `y:PolyLineEdge`). Color by `hash(domain) → HSV`. Edge kind → color + dash style. Edge status `accepted` → 2px, `proposed` → 1px.

## Invariants & Constraints
- Each script's `main()` returns int and is `callable` — `../watch.py` and `../scripts/watch.py` rely on this.
- Each script also runs as `python <file>.py` (the `if __name__ == "__main__"` block).
- Stdlib only — no third-party deps, so `mod watch` can run them in a minimal env.
- Exit code 0 = clean. Non-zero = at least one hard error, surfaced to Modulario as a failure.

## Extension Points
- New validation rule: add a function to `validate_graph.py` and append its errors to `hard_errors` (hard) or print as a soft warning above the final OK line.
- New edge kind: extend `kind_to_color` + `kind_to_style` in `export_graphml.py`, plus `VALID_KINDS` in `../vocab/watch.py`.
- New export format (e.g. cytoscape JSON, Mermaid): drop a new `export_<format>.py` mirroring the GraphML structure.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **defaultdict size-change during iteration** — `RESOLVED`
  - When it happens: walking `adj.items()` while accessing `adj[key]` for keys that didn't pre-exist.
  - Symptom: `RuntimeError: dictionary changed size during iteration` in `find_transitive_redundancies`.
  - Root cause: `defaultdict` silently inserts missing keys on read.
  - Prevention/fix: use a plain `dict` + `.get(key, default)` when traversing adjacency.

## Recent Changes
- 2026-05-19: Added `validate_graph.py` (DAG check) and `export_graphml.py` (yEd output); watch.py wired to all three.
