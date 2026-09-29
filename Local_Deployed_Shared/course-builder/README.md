# course-builder

## Purpose
- "Add your own course" on the Courses tab (Seth, 2026-09-29). The last row of
  the catalog opens onto two choices: copy a prompt that has the learner's own
  AI build a course through the content MCP, or build it here — an AI chat on
  the left, the course's concept graph on the right.

## Owns
- `#page-course-builder` — a page with no tab; `courses.js` opens it.
- The builder's own Cytoscape instance, node card, attachments tray, divider
  and full-screen mode.
- The MCP instruction text the Copy button puts on the clipboard.

## Does NOT own
- The chat backend, prompt, model or limits — `/api/conceptual/chat`
  (backend `app/conceptual/`). The chat is Concept Chat's
  (`conceptual/conceptual_chat.js` exports `createStream` + `loadBundle`).
- How the graph looks (`concept-graph/kg-look.js`) or how mastery is read
  (`concept-graph/lesson-graph.js` exports `deltaKcReadinessInfo`,
  `deltaKcMasteryColor`, `deltaKcMasteryBand`, `deltaKcLessonHtml`).
- Which concepts a course covers (`concept-graph/course-registry.js`).
- Writing a course. Nothing here creates nodes, lessons or drills yet; the
  MCP route (`content-mcp/`) is the only writer.

## Key Files
- `mcp-instructions.js`: the prompt text + clipboard copy (`DDCourseMcpInstructions`).
- `builder-data.js`: registry, lessons, course → concept set (`DDBuilderData`).
- `builder-graph.js`: the graph + node card (`DDBuilderGraph`).
- `builder-chat.js`: Deep Chat + attached concepts (`DDBuilderChat`).
- `course-builder.js`: page shell, divider, graph ⇄ lesson, full screen (`DDCourseBuilder.open()`).
- CSS: `styles/course-builder/builder.css`; the catalog row is `styles/courses/course-add.css`.

## Data & External Dependencies
- Reads `lessons/kc_registry.json`, `lessons/lessons_structured.json`,
  `concept-graph/lessonless-concepts.js` pages.
- POSTs `/api/conceptual/chat` through `window.apiFetch`.
- localStorage: `dd_cb_split` (chat width %), `dd_cb_course` (graph shown),
  `dd-course-builder-chat:<email>` (thread, Deep Chat's browserStorage).

## How It Works (Flow)
1. Courses → "Add your own course" → "Open the course builder" → `DDCourseBuilder.open()` → `switchTab("course-builder")`.
2. A MutationObserver on the page's `hidden` class builds the shell the first time it shows.
3. The graph is built per course; a node tap opens the card (mastery, gate state, neighbours).
4. "Add to AI chat" puts the concept in the tray; each question sent while it is there carries a plain-text note about it, prepended to that question only.
5. "View lesson" swaps the right pane to the lesson; "Back to graph" swaps back. The top-right button maximises the right pane; the same button minimises.

## Invariants & Constraints
- 🔴 Never the KG tab's Cytoscape. `deltaConceptGraphCy()` is for surfaces that layer on THAT graph with a restore contract; the builder draws its own.
- 🔴 The graph is rebuilt, not filtered, on a course change: `kg-look.js` `markShortcuts` remembers the first edge set it saw on an instance.
- 🔴 No model, key or system prompt in any file here (served to every visitor).
- Scripts load after `courses.js`, in order: instructions → data → graph → chat → page.

## Extension Points
- A course the AI actually writes: give the backend a builder mode (its own prompt + tool calls into `content-mcp` ops) and point `createStream` at it; the tray already describes the selected concepts.
- Another action on the node card: `cardHtml` + `onCardClick` in `builder-graph.js`.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)
- **The chat plans, it does not build** — `ACTIVE (by design, 2026-09-29)`. Seth picked "wire to existing AI" over a write-capable builder. The graph shows an existing course; the Copy route is how a course is actually written.

## Recent Changes
- 2026-09-29: Second critic pass. Arrow keys on the focused graph walk the concepts in reading order and open each card (Tab reaches its buttons, Escape returns to the graph); reopening the page retries a failed graph load; the copied login reads the password from a silent `read` into `--password -`, out of shell history and argv.
- 2026-09-29: Critic pass. Graph `show()` resolves `null` when a later call superseded it (rapid course switch); the chat rebuilds and clears its attachments on `delta:auth-state-changed` when the identity changed; New chat uses `DDConceptualChat.resetChat`; the copied instructions have the learner run `dd-content login` in their own terminal instead of handing the password to the AI; option relabelled "Plan it here with the AI".
- 2026-09-29: Created. Authored in `Delta-Drills-Deployed` at Seth's request; must be ported to `Delta-Drills-Local`/main before the next deploy or the deploy sync overwrites it.
