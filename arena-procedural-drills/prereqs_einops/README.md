# prereqs_einops

## Purpose
Procedural drills for atoms in the Einops/Einsum prerequisite cluster. Each `.ipynb` file targets one iter-5 v2 atom. Bridges to bank topic `"Einops"` (subtopics: Rearrange / Reduce / Repeat / Deep Learning) for EWMA reporting.

## Owns
- One `.ipynb` per einops-cluster atom. Current: `einops-rearrange.ipynb`.

## Does NOT own
- The builder scripts → `../scripts/`.
- Bridge rules that route atoms to `"Einops"` → `Local_Deployed_Shared/concept-graph/atom_readiness.js` (`ATOM_ID_TOKEN_TO_BANK_TOPIC`).

## Key Files
- `einops-rearrange.ipynb`: 5-exercise drill for `einops.rearrange`. Bridged subtopic: `"Einops: Rearrange"`.

## Data & External Dependencies
- Atom-ids must exist in `arena_iter5_v2.json` and resolve through the atom-id-token bridge to bank topic `"Einops"`.
- Notebook deps: `torch`, `einops`, `numpy`.

## How It Works (Flow)
See parent `arena-procedural-drills/README.md` for the full beacon flow. Per-notebook:
1. Setup cell imports torch + einops.
2. Auth cell takes `DD_TOKEN` + sets `DD_ATOM_ID` + `DD_SUBTOPIC`.
3. Five exercises (identity → swap → flatten → unfold → patchify), each with stub + assertion + solution-in-details.
4. Final cell calls `report_completion()` → POSTs to `/api/practice/arena-rating`.

## Invariants & Constraints
- All exercises in this folder must bridge to bank topic `"Einops"` — atoms whose token-bridge resolves elsewhere belong in a different topic folder.
- Filename stem = atom-id (kebab-case).

## Extension Points
- Next einops atoms (in `ATOM_ID_TOKEN_TO_BANK_TOPIC` Einops bucket): `einops-reduce`, `einops-repeat`. Use `einops-rearrange.ipynb` as template.
- Einsum atoms (token `einsum` → topic `"Einsum"`) belong in a sibling `prereqs_einsum/` folder when added.

## Recent Changes
- 2026-05-23: Folder created. `einops-rearrange.ipynb` ships.
