# Drillable-Atom Graph Re-Extraction — Plan

**Why:** The active graph (`arena_iter5_v3_encompassing.json`) has an edge
vocabulary largely disjoint from the drillable atoms. Only 21 of 207 drillable
atoms appear in any edge; 79% of edges (278/350) live entirely inside the
non-drillable reference vocabulary. FIRe credit + prereq gating are therefore
inert for ~90% of practice. A crosswalk can't fix this — there are only 5
prerequisite edges *among* the drillable atoms to remap. We must re-extract
edges over the drillable-atom node set.

**Goal:** A prerequisite + encompassing graph whose **nodes are exactly the 207
drillable atoms**, so every edge relates two things a student actually practices.

---

## Inputs (ready)

- `scripts/drillable_graph_nodeset.json` — 207 atoms: id, title, topic, subtopic,
  description, tags, and `drilled_by` (the problem refs + notes that test each
  atom). **Use `drilled_by.notes` as the primary semantic signal** — the bare
  `description` field is thin auto-generated text.
- Existing iter-5 pipeline conventions in `~/Applications/arena-iter5-pilot/`
  and `backend/scripts/eg_*.py`.

## What we SKIP vs the iter-5 pipeline

- **Skip stages 0a/0b (blind atom re-seed + merge).** Nodes are fixed = the 207
  drillable atoms. No discovery, no self-consistency voting on atoms.
- **Keep** the edge-build + encompassing-classification stages, with a stricter
  prior (below).

## Stage 1 — Candidate prerequisite edges over the 207 atoms

- Batch atoms by `topic` (28 topics, 1–13 atoms each — see nodeset). For edge
  recall across topics, also run cross-topic passes for topically-adjacent
  clusters (e.g. all Autograd_* together, all CNN/Resnet together).
- For each ordered pair (A, B) within a batch, ask: *is A a prerequisite of B?*
  i.e. must a student be able to do A before B is learnable. Use `title` +
  `drilled_by.notes` of both. Emit verbalized confidence [0,1] + a one-line
  rationale prefixed with the relation type.
- **Relation-type prefix is REQUIRED and constrained** to:
  `is-a | part-of | refines | uses | alternative-to`.
- n=3 self-consistency per batch; keep an edge if it appears in >=2 samples.

## Stage 2 — Encompassing classification (stricter prior — fixes the v3 bug)

Encompassing ⊆ prerequisite. In v3, the classifier treated the relation prefix
as "one signal among several" and promoted 226 `[uses]` edges to encompassing
(inflating FIRe credit). New rule:

- **Default `is_encompassing = (relation in {is-a, part-of, refines})`.**
- `uses` and `alternative-to` → `is_encompassing = false`, `propagation_weight = 0.0`.
- A `uses` edge may be promoted to encompassing **only** with an explicit
  justification that practicing B genuinely re-exercises A as a sub-skill (rare).
  Default is NO.
- `propagation_weight` for encompassing edges in [0.1, 0.9] by how fully B
  contains A (set by the classifier, looking at both `drilled_by` notes).

## Stage 3 — Validate (reuse + extend `eg_validate.py`)

- Structural: schema loads, no self-loops, weight/flag consistency
  (`is_encompassing=false ⇒ propagation_weight=0.0`), both prereq graph and
  encompassing subgraph acyclic.
- **NEW invariant: every node is drillable** (`concept_id` set == drillable set).
- **NEW coverage check: log how many of the 207 atoms have >=1 prerequisite edge.**
  If a large fraction are still isolated, recall failed — investigate before ship.
- Behavioral: FIRe prototype trickle-down sane (credit advanced→simpler, decays,
  bounded, terminates).

## Stage 4 — Wire in

- Write `arena_iter5_drillable.json`.
- Flip `bkt_mastery.py:53` to the new file (keep v3/v4 as reference graphs).
- Re-run `export_display_graph.py` (now a near no-op filter — all nodes drillable).
- Backup `Delta-Drills-Local` before/after (per checkpoint habit).

## Display graph (DONE, decoupled from re-extraction)

`scripts/export_display_graph.py` already produces `display_graph.json`: only
edges touching a drillable atom, tagged `render_kind = gating | encompassing`,
with a per-node `drillable` flag. The main-app viz must consume THIS, never the
raw graph — that keeps the 278 reference-layer edges off the student's skill map.

## Cost / scale note

207 atoms. Within-topic pairs are cheap (~few hundred). Cross-topic adjacency
passes add more. n=3 self-consistency. This is a multi-agent fan-out job — scope
it as a workflow (one agent per topic batch for Stage 1, one per cluster for
Stage 2) rather than one serial pass. Greenlight needed before running.
