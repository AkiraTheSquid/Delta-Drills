# visual-diff

## Purpose
A design comparison harness. It measures ONE page against ANOTHER — LessWrong's
post view against the ARENA notebook that borrows its reading design — and says
what differs, in numbers rather than in impressions. Seth's ask, 2026-09-02: "we
will look at our version using the same tool, and then it will give us a
programmatic summary of the visual differences ... I want the text size to be the
same for the middle part. I want the spacing to be the same on the left."

## Owns
- Driving a real Chrome over CDP: viewport, cache, scroll, a genuine mouse hover.
- Measuring a page into a shared role vocabulary (`prose_p`, `h1`, `toc_row`, ...)
  so two entirely different markups can be compared field by field.
- Ranking the differences by how much they matter and printing them.
- Comparing OUR stylesheet against THEIR JSS source, property by property.
- Cloning their RENDERED rail — markup and computed styles — into an ordinary
  .html/.css pair, and diffing ours against it node by node.
- The reference mockup: our notebook content under their measured design.

## Does NOT own
- The rail itself. That is `Local_Deployed_Shared/practice/arena-notebook-nav.js`
  and `styles/practice/arena-notebook-nav.css`; this folder only measures it.
- Any of their code. `reference/` is OUR markup and OUR class names carrying
  values read from their source — ForumMagnum is GPL-3.0 and none of it is
  vendored here.
- The captures. `out/` is machine-specific and gitignored.

## Key Files
- `cdp.py`: the Chrome DevTools client. Opens a tab on the already-running debug
  Chrome (`chrome_dev`, port 9222), pins the viewport, navigates, evaluates,
  hovers, screenshots.
- `probe.js`: runs INSIDE the page. Resolves each role to an element and returns
  its type, box, colours, and a `series` summary (count, gap, height, indents).
- `capture.py`: one target -> `out/<name>.json` + `.png` + `.rail.png`.
  `--no-hover` measures the rail closed, `--discover` prints what a role matched.
- `diff.py`: two captures -> a ranked list of differences.
- `pixels.py`: the same comparison from raw pixels — a composite strip and an
  "ink profile" for marks a selector cannot reach (their tick is on the row).
- `source_diff.py`: their `defineStyles({...})` blocks vs our CSS rules.
- `dom_clone.js` / `dom_clone.py`: the answer to "can't we just copy their HTML
  and CSS". Their repo has neither — it is .tsx styled by a JS object — but the
  BROWSER renders one, so this walks the rail's subtree, reads every node's
  computed style, and writes `reference/lw-toc.html` + `.css`. `--diff` then
  compares ours to theirs property by property, per role.
- `targets.json`: the three targets and their role -> selector maps.
- `reference/`: the mockup. See `reference/README.md`.

## Data & External Dependencies
- A debug Chrome on port 9222 (`chrome_dev`), and `websocket-client` in Python.
- A local static server on :5175 serving the repo root, for `ours` and `mockup`.
- `Pillow`, for `pixels.py` only.
- The ForumMagnum checkout, for `source_diff.py` only: `--repo`, `$FORUMMAGNUM`,
  `reference/ForumMagnum`, `~/src/ForumMagnum`, or the session scratchpad.
- The live LessWrong post named in `targets.json`, for `lw`.

## How It Works (Flow)
1. `capture.py <name>` opens a tab, pins 1440x900, DISABLES THE CACHE, navigates,
   waits for the target's `ready` expression, runs its `pre` steps (a scroll into
   the body of the document), dispatches a real hover, then evaluates `probe.js`.
2. The result is a JSON of roles plus two screenshots — the viewport and a crop
   of the left rail.
3. `diff.py a b` walks both role maps, scores every field, and prints the
   differences worst-first. `pixels.py a b` does the same from the crops.
4. `source_diff.py` reads their JSS, normalises it to CSS, and compares the
   twelve elements our rail maps onto.
5. `dom_clone.py lw` and `dom_clone.py ours` each dump a rail's whole subtree
   with computed styles; `dom_clone.py --diff` lines the two up by role and
   prints every property that differs, with a `DEVIATIONS` table for the ones
   that differ on purpose.

## Which tool answers which question
- "did we write the same declarations they wrote?" -> `source_diff.py`
- "does the browser end up computing the same thing?" -> `dom_clone.py --diff`
- "does it LOOK the same?" -> `capture.py` + `pixels.py` + the .rail.png crops

🔴 THE FIRST TWO ARE NOT THE SAME QUESTION, and the gap between them is where
this went wrong once already: `source_diff.py` reported 32 identical properties
on a rail that plainly did not match, because it can only compare declarations
both sides WROTE — a property supplied by a parent, a theme, or a MUI baseline
is invisible to it, and so is the shape of the tree. `dom_clone.py` reads the
finished result, which is why it was the one that found the real fault: their
row is TWO elements, and ours was one.

## Invariants & Constraints
- 🔴 THE VIEWPORT IS PINNED BEFORE NAVIGATION AND THE CACHE IS OFF. A capture
  taken at whatever size the tab happened to be, or answered from a cached
  stylesheet, measures the environment or the last edit but one.
- 🔴 THE HOVER IS A REAL `Input.dispatchMouseEvent`, not a class added by hand.
  Both rails open on pointer position; setting `.is-hover` yourself measures a
  state a reader cannot reach.
- 🔴 A ROLE IS A ROLE, NOT A SELECTOR. Add the selector to every target's map or
  the field silently drops out of the comparison. `unmeasurable` names roles that
  cannot line up at all (their tick is painted on the row) so the report says so
  instead of reporting a difference.
- 🔴 A DIFFERENCE IN A ROLE IS ONLY A FINDING IF THE ROLE POINTS AT THE SAME
  PART OF BOTH TREES. Half of `dom_clone.py`'s first report was noise: `toc_dot`
  matched their row CONTAINER, `toc_row` matched their level element, and
  `toc_label` matched the title on one side and a body row on the other. Check
  the tag and the box in the report's header before believing a property.
- 🔴 CONTENT COUNTS ARE NOT DESIGN FINDINGS. Two different documents have
  different paragraph counts; `diff.py` warns and de-weights them.
- No ForumMagnum code is copied into this repo. Values are read and re-typed.

## Extension Points
- A new page to compare: add an entry to `targets.json` with `url`, `ready`,
  `pre`, `hover` and a `roles` map, then `capture.py <name>`.
- A new thing to measure: add it to `probe.js` and give it a weight in
  `diff.py`'s `NUMERIC` / `EXACT` / `COLOR` tables.
- A new CSS element to hold to their source: add a row to `source_diff.py`'s
  `MAP`; add an entry to `DEVIATIONS` for anything we differ on ON PURPOSE, so
  the report separates a decision from drift.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Chrome refuses the DevTools websocket** — `RESOLVED`
  - When it happens: any connection made by `websocket-client`.
  - Symptom: `Handshake status 403 Forbidden`, with Chrome suggesting
    `--remote-allow-origins`.
  - Root cause: the library derives an `Origin` header from the URL, and Chrome
    rejects a DevTools socket carrying an origin it was not launched to allow.
  - Prevention/fix: `create_connection(..., suppress_origin=True)`. Do not
    relaunch Seth's browser to work around it.

- **The rail screenshot comes back blank** — `RESOLVED`
  - When it happens: cropping the rail after scrolling down the page.
  - Symptom: an empty strip, or the top of the document.
  - Root cause: `Page.captureScreenshot`'s clip is in DOCUMENT coordinates, so
    `y: 0` photographs the top of the page, where a fixed rail is not.
  - Prevention/fix: clip at `window.scrollY` (`capture.py`).

- **Everything unpainted reports a black background** — `RESOLVED`
  - Root cause: `rgba(0, 0, 0, 0)` parses to `#000000`; alpha was thrown away.
  - Prevention/fix: alpha 0 is compared as the literal `transparent`.

- **The rail is photographed closed** — `RESOLVED`
  - When it happens: capturing `ours` right after the notebook renders.
  - Symptom: no labels in `ours.rail.png`, and `toc_label` measured at opacity 0.
  - Root cause: the rail is built AFTER the cells, so the synthetic hover landed
    where `.anb-toc-hit` did not exist yet and no `mouseenter` ever fired.
  - Prevention/fix: the target's `ready` expression waits for `.anb-toc-row`, not
    only for cells. Any new target whose rail is built asynchronously needs the
    same. Status: `RESOLVED`, and the shape of it will come back — check
    `capturedWith` and the label box in the JSON before trusting a rail capture.

## Recent Changes
- 2026-09-02 (later): Added `dom_clone.py` + `dom_clone.js`, after Seth asked
  "doesn't that other repository utilize HTML and CSS? Can't we just duplicate
  the HTML and CSS". It does not — but the browser renders one, so the tool
  reads it back. That found what `source_diff.py` could not see: their row is a
  wrapper (the section's share of the document) around a line (one line of
  content at the top of it), and ours was a single element, which is why our
  rail was an evenly-spaced list. Rail restructured to their tree; the report
  now reads 785 identical, 0 different. Role maps in `targets.json` re-pointed
  at matching nodes and `toc_line` / `toc_fade` / `toc_level` / `toc_fill` /
  `toc_rows` added. `reference/` no longer keeps its own copy of the rail — the
  mockup links the app's stylesheet and mounts the app's `ArenaNotebookNav`,
  because the copy had already gone stale once.
- 2026-09-02: Built. `cdp.py` / `probe.js` / `capture.py` / `diff.py` /
  `pixels.py`, the `reference/` mockup, and `source_diff.py`. Used to port the
  ARENA rail from ForumMagnum's source (dots on a rule, proportional rows, the
  scroll-window marker) and to set the notebook's reading column to their
  measure: 682px of 18.2px/26px text, headings 36.4 / 26 / 20.8.
