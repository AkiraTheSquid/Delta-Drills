# tools

## Purpose
Developer tooling that is ABOUT the app but is not part of it. Nothing in here
ships: no file under `tools/` is loaded by `Local_Deployed_Shared/index.html` or
served to a learner, and the deploy does not need it.

## Owns
- Harnesses that measure or compare the shipped app from the outside.

## Does NOT own
- Anything the app runs. That is `Local_Deployed_Shared/` (frontend) and
  `This-Directory-Only/backend/` (API).
- The content pipeline and its audits — those live in
  `This-Directory-Only/scripts/`, which is where a script that reads the question
  bank belongs.

## Key Files
- `visual-diff/`: compares the ARENA notebook's design against LessWrong's, both
  as rendered pixels and as stylesheet-versus-source. See its own README.

## Data & External Dependencies
- Per-tool. `visual-diff` needs a debug Chrome on :9222 and a static server on
  :5175; nothing here talks to the backend or the database.

## How It Works (Flow)
1. Each subfolder is a self-contained tool with its own README and entry points.
2. They are run by hand, from this directory, while working on the surface they
   measure.

## Invariants & Constraints
- 🔴 NOTHING HERE IS WIRED INTO THE APP. A tool that the app imports is not a
  tool, it is a feature in the wrong folder.
- Output that is machine- or moment-specific (screenshots, captures) is
  gitignored, never committed.
- Third-party source read by a tool stays OUTSIDE this repo — cloned elsewhere
  and pointed at — so licences do not follow it in.

## Extension Points
- A new tool is a new subfolder with a README, a `watch.py`, and no import from
  the app.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **A tool grows into the app** — `ACTIVE`
  - When it happens: a harness turns out to have a function the app also wants.
  - Symptom: `Local_Deployed_Shared` starts depending on a path under `tools/`,
    and the next deploy ships either too much or too little.
  - Root cause: the folder boundary is by AUDIENCE (a learner vs a developer),
    and a shared helper has no audience.
  - Prevention/fix: move the shared part into the app and let the tool import
    the app, never the other way round.

## Recent Changes
- 2026-09-02: Created, with `visual-diff/` as its first tool.
