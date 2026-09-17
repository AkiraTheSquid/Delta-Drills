# Editing Delta Drills content with Claude Code

You do not need to know this repo. You need the editing password, a terminal,
and Claude Code. Everything below works on a fresh clone with nothing installed
but Python 3.9+.

## 1. Point Claude Code at the repo

```bash
git clone <this repo>
cd Delta-Drills-Local
claude
```

The repo ships a `.mcp.json`, so Claude Code offers to enable the
`delta-drills-content` MCP server on first launch. Say yes. Check it took with
`/mcp` — you should see 28 tools.

If you would rather register it by hand, or you use a different MCP client:

```json
{ "mcpServers": {
    "delta-drills-content": { "command": "/abs/path/to/content-mcp/bin/dd-content-mcp" }
} }
```

## 2. Unlock editing

Reading is open to anyone. Writing needs the shared password, once per session.
Ask Seth for it — it is shared out of band and deliberately not written down in
this repo (only its salted digest lives in `content-mcp/auth.json`).

> Log in to the content server with the content password

or, in a terminal:

```bash
./content-mcp/bin/dd-content login --password '<the shared password>'
```

The session lasts 12 hours. For scripts and CI, set
`DELTA_DRILLS_CONTENT_PASSWORD` in the environment instead.

## 3. Ask for what you want

The point of the MCP is that you describe the change and Claude Code makes it
against the real files:

> Read the lesson on einops.merge-axes and tell me what it covers.
>
> The worked example on numpy.dtype-astype never prints anything. Fix it so it
> shows the contrast it is teaching.
>
> Add three Solo drills to python.indexing — negative indices, and one that
> shows an out-of-range index raising.
>
> Add a concept node for einops.einsum after einops.merge-axes, then write its
> page.

Claude Code will use the tools below. You can also drive them yourself with
`./content-mcp/bin/dd-content <tool> --help`.

## The three layers of content

| Layer | What it is | Tools |
|---|---|---|
| **Pages** | One markdown file per concept, four rungs: Lesson → Faded → Solo → Integrated | `lesson_list`, `lesson_read`, `lesson_outline`, `lesson_edit`, `lesson_write`, `lesson_create` |
| **Graph** | `kc_registry.json` — the concepts, their prerequisites, and the teaching order | `graph_list`, `graph_read`, `graph_add_kc`, `graph_update_kc` |
| **Drills** | The question bank the app actually serves | `drill_search`, `drill_read`, `drill_add`, `drill_update`, `drill_retire` |

Plus `pipeline_*` to check your work and `backup_*` to undo it.

## 4. Check your work — this is not optional

```bash
./content-mcp/bin/dd-content pipeline_check
```

That runs the repo's real gate, in order: every code fence in every page is
executed, every faded solution is graded against its drill's test cases, the
pages are compiled, the Q-matrix is rebuilt, the bank is re-exported, and the
bank is audited for drills that can be gamed. It stops at the first failure and
hands you the output.

Then:

```bash
./content-mcp/bin/dd-content pipeline_watchers
```

Those carry two standing guards that `pipeline_check` does not: a drill may not
use a function that no lesson at or before its concept has taught, and a
function that appears in zero ARENA notebooks is attention spent on something no
learner needs.

**A change that has not passed both has not landed.**

## 5. If you break something

```bash
./content-mcp/bin/dd-content backup_status     # how old is the snapshot
./content-mcp/bin/dd-content backup_list       # what it holds
./content-mcp/bin/dd-content backup_restore --confirm
```

One snapshot of all content is kept, refreshed automatically before the first
write of any day (`backup_restore` deliberately never rotates it first, so
recovering cannot destroy the copy you are recovering from) — so it is up to about 24 hours old and always predates the
session you are in. Restore overwrites every content file with what the snapshot holds, and parks
the current tree at `.content-mcp/pre-restore.tar.gz` first, so a restore is
itself undoable. It does not DELETE — a file created after the snapshot is left
in place and named in the result, because deleting other people's new work
would be worse than leaving it.

To roll back one file rather than everything:

```bash
./content-mcp/bin/dd-content backup_restore --confirm \
  --paths 'Local_Deployed_Shared/lessons/numpy/kp-ndarray-model.md'
```

Git is still the real history. The snapshot is for the case where you have made
a mess across several files and want out of it in one command.

## Things that will bite you

- **Read `lesson_authoring_guide` before writing a page.** The four-rung format
  is a contract the validator enforces, not a style preference. One concept per
  segment, exactly one worked example per segment, and every segment must stand
  alone because the learner sees one segment per page.
- **A code fence gets a Run button if and only if CI runs it.** Plain
  ` ```python ` is executed by the validator and rendered as a runnable cell;
  ` ```python no-run ` is illustrative and never executed. There is no third
  state.
- **Question ids are positional.** `drill_add` appends and mints the next id;
  `drill_retire` stops a drill being served. Never delete a row from a CSV — it
  renumbers every question below it and silently re-points every tag and every
  stored attempt.
- **Never hand-edit `questions.json`, `questions_structured.json`, or
  `lessons_structured.json`.** They are generated on every export and your edit
  will vanish. Change the source and re-run the pipeline.
- **Adding a KC takes more than the registry.** A new concept also needs a
  glossary entry, or the lessons watcher fails with `kcLesson is missing`. Run
  `pipeline_watchers` and it will tell you.
- **Only whitelisted fields work in a drill override.** `drill_update` refuses
  anything else and names the allowed set, because the exporter would ignore it
  silently and your edit would look like it worked.

## Changing the password

```bash
./content-mcp/bin/dd-content set-password --password '<new one>' --current '<the old one>'
```

Rewrites the salted digest in `content-mcp/auth.json`. Changing it requires the
current password, so holding the CLI is not enough to seize the gate. Existing
sessions keep working until they expire; share the new password out of band.
