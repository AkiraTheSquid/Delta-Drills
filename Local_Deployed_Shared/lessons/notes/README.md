# notes

## Purpose
- The metadata layer of the knowledge graph: one Markdown note per concept, holding what the AUDITS found and why decisions were made — the qualitative half of docs/spec-graph-metadata-audit-layer.md §3b.

## Owns
- `<kc-id>.md` files (filename = the KC id in `../kc_registry.json`, front matter `kc:` must match).

## Does NOT own
- Lesson content (the KP pages in the sibling folders), quantitative stats (`/api/practice/kc-stats` computes those live from the attempt logs), or the registry itself.

## Key Files
- Each note: `---\nkc: <id>\n---` then Markdown. Conventional sections: `## Findings` (dated; every entry cites a measurement, a decision, or the check that now enforces it), `## Edges` (notes about a specific prereq/encompassing edge, named by the far end), `## Checks` (guards this concept's findings gave rise to). `[[kc-id]]` wikilinks connect notes — open this folder in Obsidian and the wiki IS the graph metadata.

## Data & External Dependencies
- Compiled into `lessons_structured.json` as `notes_markdown` per KP by `scripts/compile_lessons.py`, which HARD-FAILS on a note whose filename/front matter names no live KC — a renamed concept must take its note with it.

## How It Works (Flow)
1. An audit or a work session finds something about a concept → write/extend its note.
2. `python3 scripts/compile_lessons.py` folds the body into the KP's entry.
3. The (future) Metadata tab renders it beside the lesson in advanced mode.

## Invariants & Constraints
- A finding with no measurement, decision, or check behind it is an opinion — it belongs in a commit message, not here.
- Notes are GLOBAL (about the concept, not any learner). No user ids, ever.

## Extension Points
- Edge-level structured metadata (per-edge YAML) only when something actually consumes it — prose sections first.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)
- **Stale note after a KC rename** — prevented: the compile orphan-guard exits nonzero on a note naming a dead KC.

## Recent Changes
- 2026-09-01: folder created with four seed notes relocated from CLAUDE.md's frontier list and the 2026-09-01 attempt-log findings.
