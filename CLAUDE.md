## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Content work: ONE concept at a time (Seth's rule, 2026-08-28)

Seth learns on this app himself, one knowledge point at a time, and he sends
feedback to whichever Claude is working on the concept he is standing on right
now. So content work is scoped by HIS position, not by the shape of the bank.

**The rule.** Pick the single concept Seth is currently practising. Improve the
content for that concept across **every rung of the ladder** — Lesson, Faded,
Solo, Integrated — so the whole climb on that one node is good. Then stop. Do
not spread thin work across sibling concepts, do not "while I'm here" a
neighbouring KP, and do not rewrite the bank. Another session is handling
another concept; the improvements land one node at a time, moving up.

**His account.** `sethbgibson@gmail.com` — that is the account he actually
progresses through the content on, so it is the one to read. Internally the
backend keys on a UUID, not the email:

    user_id = c813fa78-7e0f-4859-bcb3-a2183ef98eb4

**Finding out which concept he is on.** Do not guess and do not ask him if you
can read it. His practice state lives on the Fly volume, not in the repo:

```bash
FLYCTL="${HOME}/.fly/bin/flyctl"
[ -z "${FLY_API_TOKEN:-}" ] && export FLY_API_TOKEN="$(awk '/access_token/ {print $2}' "$HOME/.fly/config.yml" | tr -d '"')"
U=c813fa78-7e0f-4859-bcb3-a2183ef98eb4
"$FLYCTL" ssh console -a delta-drills-backend -C \
  "tail -5 /data/user_data/$U.attempts.jsonl"          # kc of the last drills served
"$FLYCTL" ssh console -a delta-drills-backend -C \
  "cat /data/user_data/$U.json"                        # kc_ladder: rung + attempts per KC
```

The `kc` on his most recent attempts is the concept he is on. `kc_ladder[<kc>]`
tells you which rung he is stuck on (`worked`=Lesson, `faded`=Faded,
`partial`=Solo, `solo`=Integrated — the stored ids are historical, the meanings
are the four stages). A rung he keeps failing, or one the app reported as
exhausted, is where the writing goes first. `/content-gaps` and the `drill-gaps`
skill report the same thing from the other side.

**Where the content work has actually reached (keep this current).**

- `numpy.ndarray-model` — "What an ndarray is: data + shape + dtype".
  DONE 2026-08-28: four-stage ladder, 36 new drills (ids 532–567), input→output
  and "not this" examples under every prompt. This is the only concept that has
  had the full treatment.
- The six dependents of `numpy.ndarray-model` — `reshape-flatten`, `elementwise-ufuncs`,
  `constructors`, `dtype-astype`, `sorting`, `transpose-axes` — DONE 2026-08-30: 57 new
  drills (ids 619–675) so every rung has enough unseen drills for Seth's local topology
  (Faded ≥ 2 per segment, Solo ≥ 6, Integrated ≥ 3). `reshape-flatten` was re-cut into
  three segments. Seth's own record at the time: failing `reshape-flatten`'s one faded
  drill, so that node came first.
- **Worked examples now POP UP on a schedule** (2026-08-30): `This-Directory-Only/
  SPEC_WORKED_EXAMPLE_SCHEDULE.md`, table in `backend/app/example_schedule.py`. It is an
  experiment — retune the table, nothing else. That spec has a
  **"Deferred — 🤖 AI: REMIND SETH"** section: per-drill encompassing-credit propagation
  (which prerequisite nodes a drill exercises, encoded per drill), splitting the neighbour
  nodes, retuning the schedule. Bring the first one up whenever he talks about credit
  propagation or a prerequisite feeling under-practised.
- **The blob nodes need SPLITTING, not just drills** (2026-08-31):
  `This-Directory-Only/SPEC_NODE_SPLITTING.md`. Six never-segmented nodes declare
  up to ten symbols each behind ONE mastery number; 51 of the graph's 144 declared
  symbols are drilled fewer than twice on their own concept and 19 are drilled zero
  times. `scripts/audit_symbol_coverage.py` measures it and now guards it as a
  ratchet in every lesson watcher. Read that spec before proposing a node boundary —
  the test is "can a learner fail this while succeeding at the rest of the node?",
  and one-function-per-node is explicitly wrong here.
- **First split landed** (2026-08-31): `numpy.random-generator` → `numpy.random-samplers`
  → `numpy.random-seeding` → `numpy.random-threading`, 32 new drills (ids 676–707),
  every rung floor met and zero symbols under the coverage floor on all three nodes.
  Teaching order was forced by the symbols, not the pedagogy: seeding owns
  `torch.Generator`/`manual_seed`/all four `generator=` kwargs, which left threading
  with no symbols of its own — a legitimate discipline node ("use the generator you
  were handed"), the same shape as the `einops.*` pattern nodes. ⚠️ The Colab concept
  maps were NOT regenerated; they still name the dead id. See `SPEC_NODE_SPLITTING.md`.
- **The course ROOT is now at the floors** (2026-09-01): `python.values-and-names`,
  5 new drills (ids 708–712) — Solo 2→6, Integrated 2→3. It was the ONLY live entry in
  `content-gaps.json` (15 hits) and it is where the queue reaches when a numpy concept
  is locked, so running out there stops practice everywhere. `builtin.type` was declared
  in its `new_syntax` and drilled ZERO times on its own node while the page never showed
  `type(...)` at all; the Concept section now teaches it and two drills use it. 🔴 The
  other six `python.*` nodes are still 2/2/2 — this fixed the root, not the course.
  ⚠️ Re-recorded `solution_prereq_baseline.json` (1318→1374): every function-mode drill
  on the root needs `def`/`return`/`call`/`docstring`/`star-args`, and
  `python.defining-functions` comes LATER in the course, so the root drills what it has
  not taught. Zero NEW symbol families — the 56 entries are the same envelope already
  recorded for 568–573. The real fix is course ORDER, not more drills.

- **`einops` has NOT had it.** Its KPs still carry the pre-2026-08-28 content:
  thin rungs, no Solo/Integrated split, no worked input/output tables. It is the
  next frontier — do it as Seth reaches it, one concept at a time, same
  full-ladder treatment. Keep moving forward rather than circling back over numpy.
- **The einsum course is RETIRED (2026-08-30).** ARENA writes `einops.einsum` in
  61 of its 458 notebooks and `torch.einsum` in zero, and all ten einsum KPs were
  written in `torch.einsum`. The pages are in
  `This-Directory-Only/archive/retired-content-2026-08-30/`. The replacement is
  ONE `einops.einsum` node inside the einops course — the highest-frequency
  einops operation ARENA has, and nothing currently teaches it. That is the
  single most valuable concept left to author.

When you finish a concept, update the list above with the date and the ids, so
the next session can see where the frontier is without re-deriving it.

The authoring contract itself — the four stages, the vocabulary rule, the
input/output requirement — is `Local_Deployed_Shared/lessons/AUTHORING.md`.
