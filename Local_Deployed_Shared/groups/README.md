# groups

## Purpose
- The Groups tab: a handful of learners practising the same curriculum, reading
  each other's readiness AND each other's day side by side. Start a group, join
  one by invite link or out of the public directory, then see one ROW per
  member: their area mastery on the left, and on the right the three-state
  checklist they wrote for the day the picker at the top is showing.
- Ported from Delta Note's accountability groups
  (`shared/web-components/js/accountability/`), deliberately: the join model,
  the consent gate, the capability token and the initials-only directory are
  the same feature answered twice, and the second copy should not re-decide
  questions the first one already answered.
- The checklist column is a second port from the same app: Delta Note's
  per-goal sub-goals (`shared/web-components/js/subgoals/`). Same Tiptap
  `{v, doc}` document, same three-state `completion` attr, same
  open → checked → X cycle, and the same vendored engine at the same version —
  so a list written in one app is readable in the other.

## Owns
- Getting into a group and out of one: the discovery card, the consent dialog,
  the invite link (mint, copy, rotate, read out of the address bar, clear it
  again), the public directory list, and the group bar above the roster.
- Drawing the roster: one ROW per member, two columns — their circle, name and
  area bars on the left, their day's checklist on the right.
- The day: which day the page is showing, the picker that changes it, and the
  local-date arithmetic behind ‹ and ›.
- The checklist DOCUMENT and both ways of drawing it: the live three-state
  editor for your own column, and the read-only renderer for everybody else's.
- The client half of `/api/practice/groups/*` — one function per endpoint, and
  the read/action failure split described in `groups_store.js`.

## Does NOT own
- **The readiness scale.** `PlacementResults.readiness()` / `.band()`
  (`../practice/placement-results.js`) is the only theta→percentage map on this
  side, and it is a mirror of `_mastery_from_theta` in
  `backend/app/diagnostic.py` that `../practice/watch.py` fails on drift. This
  folder does no arithmetic on `theta` at all.
- **How a readiness bar looks.** `.placement-area*` is declared in
  `../styles/practice/diagnostic.css`. `../styles/groups.css` is the frame
  around those rows and nothing more.
- **Who is signed in.** `window.DDIdentity` (published by `../app.js`).
- **Anybody's mastery numbers.** The server computes each member's `areas` from
  their practice state; nothing here reads or caches a posterior.
- **The editor engine.** `../vendor/tiptap/` — one vendored bundle, imported on
  first use. Read that folder's README before touching anything Tiptap-shaped;
  the single-bundle rule and the Tiptap-3 `setContent` trap are written down
  there.
- **Routing.** `switchTab` in `../app.js` decides when this page is visible and
  calls `DDGroups.refresh()` on arrival; the invite-link landing rule is in that
  file's boot call, not here.

## Key Files
- `groups_store.js` (`window.DDGroupStore`): the nine calls a group is made of,
  plus the invite-link helpers. Loaded first, and 🔴 all three are loaded
  BEFORE `app.js` — its boot `switchTab` runs while it parses and asks this
  file whether the address bar carries an invite token, so the store must
  already exist. Nothing here touches `apiFetch` or `DDIdentity` at load time,
  only at call time, so loading early is free.
- `groups_join.js` (`window.DDGroupsJoin`): avatars, the public directory, the
  discovery card with its one consent gate, and the group bar. This is the file
  Delta Note's `accountability_discovery.js` + `_avatars.js` + `_directory.js`
  collapse into, because this app has classic scripts and no module boundaries
  to hang three files off.
- `groups_view.js` (`window.DDGroups`): the page. Reads the group, holds which
  day is showing, decides between the discovery card and the roster, and
  composes the rows. `refresh()` and `suspend()` are what `app.js` calls.
- `groups_lane.js` (`window.DDGroupsLane`): ONE member as a full-width row —
  the mastery column, the checklist column, and the only live editor mount on
  the page. `destroyAll()` is the teardown the page calls before every repaint.
- `groups_day.js` (`window.DDGroupsDay`): the day picker and the local-date
  arithmetic. No dates are computed anywhere else.
- `groups_checklist_doc.js` (`window.DDChecklistDoc`): everything about the
  checklist document that does not need Tiptap loaded — the three-state cycle,
  the `{v, doc}` wrapper, the normalisation, the counting, and the read-only
  renderer that draws other people's lists.
- `groups_checklist.js` (`window.DDChecklist`): the editor lifecycle. Imports
  the vendored bundle on first mount, extends `TaskItem` with the third state,
  and debounces the save.

## Data & External Dependencies
- `/api/practice/groups/{mine,public}` and `POST /api/practice/groups`,
  `/join`, `/join-public`, `/leave`, `/rotate-token`, `/visibility`,
  `/display-name` — `backend/app/practice/groups_router.py`, backed by
  `backend/app/study_groups.py` and the `study_groups` /
  `study_group_members` tables.
- `GET /api/practice/groups/day?date=YYYY-MM-DD` and `PUT .../groups/day` —
  the checklists, backed by `backend/app/study_group_days.py` and the
  `study_group_days` table. A separate pair of endpoints ON PURPOSE: `/mine`
  recomputes every member's posterior, and the day picker is a control people
  click through a week.
- `window.PlacementResults` from `../practice/placement-results.js`, and the
  Tiptap bundle from `../vendor/tiptap/`.
- A member on the wire is `{member_id, display_name, initials, joined_at,
  areas, probes}`. `areas` is the SAME `{topic, theta, sd, probes}` shape
  `/api/practice/diagnostic/status` answers for one learner — that is what lets
  the member card reuse the Learner Home's renderer wholesale.
- `window.apiFetch` and `window.DDIdentity` from `../app.js`;
  `window.PlacementResults` from `../practice/placement-results.js`.

## How It Works (Flow)
1. `switchTab("groups")` calls `DDGroups.refresh()`.
2. Signed out → the discovery card, with the sentence explaining why a guest
   has nothing for a group to read. No request is made.
3. Signed in → `GET /groups/mine`. `null` means "you are in no group" (or a
   failed read — the caller cannot tell, on purpose, because both mean "draw
   the discovery card").
4. In no group: the discovery card. Create / paste an invite link / pick a
   listed group all call `ask()`, which shows the consent dialog and runs the
   join only if it is accepted. The server answers the whole group, so the
   click that joined paints the roster on that click.
5. In a group: the group bar (name, count, listing state, copy invite, new
   link, leave), then the day picker, then one row per member.
6. The day is read separately, once per day change, and the columns say
   "Reading this day…" until it lands. Your own column mounts the editor when
   the read returns; everybody else's is drawn from the JSON directly.
7. Typing debounces 500 ms and `PUT`s your own row. Changing the day, leaving
   the tab, or any repaint destroys the editor — and the teardown FLUSHES that
   debounce.

## Invariants & Constraints
- **🔴 One consent gate, three doors.** Create, invite link and directory all
  run through `ask()` in `groups_join.js`. A fourth way in must go through it
  too — the one added later is the one that forgets to ask.
- **🔴 The directory never learns a name.** `/groups/public` answers initials
  and member ids. There is no `display_name ?? initials` fallback anywhere on
  that path, and there must not be: a fallback would work perfectly on the
  roster, which carries names, and silently do nothing on the directory, which
  does not — which is exactly how the tighter boundary stops being the one in
  effect.
- **🔴 A group always has a live owner.** Only the owner may rotate the token
  or change the listing, so `leave_group` hands the group to the longest-standing
  remaining member when the owner walks out. A group pointing at a departed
  owner has no error state and no UI — the controls simply stop being drawn for
  everybody, permanently.
- **🔴 The invite token is a capability.** Anyone holding it is in the group and
  can read every member's mastery. It is shown behind a click, cleared out of
  the address bar the moment it has been used, and only the owner may rotate it.
- **🔴 No second readiness scale.** Draw `.placement-area*` rows through
  `PlacementResults`, or do not draw them.
- **🔴 An unprobed area is dimmed and says "not probed".** It is a prior
  wearing a percentage. Reading somebody else's number is exactly the moment a
  false measurement does damage.
- **No polling.** The roster is read on arrival at the tab and after a mutating
  call, and at no other time.
- **No `confirm()`.** The listing toggle arms and confirms itself in two clicks;
  a native confirm blocks the page and this repo's browser checks cannot
  dismiss one.
- **🔴 EXACTLY ONE LIVE EDITOR, AND IT IS YOURS.** Everybody else's checklist
  is `DDChecklistDoc.renderStored` — the same markup, painted by the same CSS,
  with no ProseMirror behind it. Twelve editors would be twelve node-view
  registries and eleven contenteditables nobody may type into.
- **🔴 A SAVE NAMES THE DAY IT WAS TYPED ON.** `onSave` closes over the day the
  row was painted for, never the module's live `day`. The teardown flushes the
  pending save, and the teardown happens BECAUSE the day changed — read the
  live value there and Wednesday's last line is written over Tuesday's list.
- **🔴 A DAY KEY IS THE LOCAL CALENDAR DATE.** Never `toISOString()`, on either
  side of the wire: west of Greenwich it names yesterday all evening, and the
  failure is a list that reads back empty rather than an error.
- **🔴 A LOAD MUST NEVER SAVE.** `setContent(content, { emitUpdate: false })` —
  an options object. Tiptap 3 ignores a positional `false` and emits anyway.
- **The write endpoint takes no target.** `PUT /groups/day` writes the
  authenticated user's row and accepts no member id. A group is joined by
  anyone holding a link.

## Extension Points
- **A new column on a member row** → `buildRow` in `groups_lane.js`. Both
  existing columns are quoted markup — the Learner Home's `.placement-area*`
  and the editor's `.ProseMirror` task list — and a third should be too.
- **Something else stored per person per day** → it belongs beside `payload` in
  `study_group_days`, not in a second table: the read is already one query per
  day and the row is already keyed the right way.
- A new group operation: add the endpoint in `groups_router.py`, the store
  function in `groups_store.js` (through `act()` if a person clicked it,
  through the swallowing path if it is a background read), then the control in
  `groups_join.js`.
- More on a member card: it is `buildMemberCard` in `groups_view.js`. Anything
  that is a per-learner readout the Learner Home already draws should be quoted
  the way the area rows are — same markup, same writer — rather than restyled.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **A failed day read looked exactly like a slow one** — `RESOLVED` (2026-09-02)
  - When it happens: the day endpoint answers anything but 200 — an expired
    token, a dropped connection, the backend restarting.
  - Symptom: "Reading this day…" under every member, forever, with nothing in
    flight behind it. The page looks busy and is not.
  - Root cause: `readDay` answers `null` on failure and the view stored that in
    `entries`, which the column already read as "not back yet". Two different
    states spelled the same way. The comment in the view even claimed the row
    builder rendered an honest failure — it did not.
  - Prevention: the view carries `dayState` (`loading`/`ready`/`failed`)
    alongside `entries`, the column says the day could not be read, and a Retry
    button appears beside the picker. `watch.py` asserts both spellings.

- **A save that failed said nothing** — `RESOLVED` (2026-09-02)
  - When it happens: a list past the 20 000-character cap (413), or any network
    failure while the debounce is in flight.
  - Symptom: the words are on screen, the checklist is not in the database, and
    nobody finds out until the next load.
  - Root cause: `void store().saveDay(...)`. The answer carries `{error}` and it
    was being thrown away, and the write is debounced half a second, so the
    person who typed has already looked away.
  - Prevention: `saveDay` returns the promise and the column shows
    "Not saved: …" under the editor until the next save succeeds.

- **Two writes in flight let the network pick the winner** — `RESOLVED` (2026-09-02)
  - When it happens: typing through a day change (the teardown flushes) or any
    two debounces landing close together on a slow link.
  - Symptom: a sentence that was typed, saved and confirmed reappears deleted.
  - Root cause: every save sends the WHOLE document, and two independent PUTs
    have no order. The read after a day change could also overtake the flush,
    hand the editor the document as it was BEFORE it, and save that back.
  - Prevention: `groups_store.js` keeps one write chain — saves queue behind it
    and `readDay` waits for it. Last typed is last written.

- **Leaving the tab did not stop the read** — `RESOLVED` (2026-09-02)
  - When it happens: clicking away while a day is still loading.
  - Symptom: nothing visible — which is the problem. A ProseMirror view mounted
    into a hidden page, saving on its own debounce, with nothing left to destroy it.
  - Root cause: `suspend()` destroyed the editor but the in-flight `loadDay`
    repainted afterwards, and a repaint mounts.
  - Prevention: `suspend()` clears `active` and bumps `daySeq`; `render()`
    refuses to paint while inactive.

- **A second copy of the theta→readiness map** — `RESOLVED` (never shipped)
  - When it happens: writing any surface that turns a `theta` into a percentage.
  - Symptom: two screens quoting different readiness for the same learner —
    exactly the "45% ready" / "0% ready" split `placement-results.js` documents.
  - Root cause: the four constants are four lines, so re-typing them looks
    cheaper than reaching for `PlacementResults`.
  - Prevention/fix: `groups_view.js` calls `PlacementResults.readiness()` and
    `.band()` and declares no constants. `../practice/watch.py` parses the JS
    and the Python and fails on drift — but only for the copy it knows about.
  - Status: `RESOLVED`.

- **`apiFetch` read off `window` and falling back to bare `fetch`** — `ACTIVE`
  - When it happens: any new module in here that fetches.
  - Symptom: on Vercel the SPA rewrite answers a relative `/api/...` with 200
    `text/html`; `res.ok` is true, `res.json()` throws, the catch writes null,
    and the surface renders the signed-out reading for a signed-in learner with
    nothing in the console.
  - Root cause: `apiFetch` is a top-level `const` in a classic script, so it is
    not always a window property.
  - Prevention/fix: `groups_store.js` has ONE fetch helper and it throws
    "app not ready" rather than falling back. Route new calls through it.
  - Status: `ACTIVE` — the trap is a property of the deployment, not of a bug.

- **The groups scripts loaded after `app.js`** — `RESOLVED`
  - When it happens: adding a script tag to `index.html` by putting it at the
    bottom, which is where nearly every other one goes.
  - Symptom: `?group=<token>` opens the Learner Home. The banner, the consent
    dialog and the join all work if you then reach the tab by hand, so nothing
    looks broken; the token just sits in the address bar doing nothing.
  - Root cause: `app.js` runs its boot `switchTab` as it parses, and that call
    reads `DDGroupStore.inviteFromLocation()`. Below `app.js` the store does not
    exist yet and the question answers `undefined`.
  - Prevention/fix: the trio loads immediately above `app.js`, and
    `watch.py`'s `check_public_api` fails if it ever drops below it.
  - Status: `RESOLVED` — found in the browser, not by a check; the check exists
    now because of it.

- **The checklist editor could not load** — `RESOLVED`
  - When it happens: moving the vendored bundle, or writing the import
    specifier the way every other URL in this app is written.
  - Symptom: your own column reads "The checklist editor could not load" while
    the day picker, the roster, and everybody else's checklist all work. A 404
    in the console for `/groups/vendor/tiptap/tiptap.bundle.esm.js`.
  - Root cause: a dynamic `import()` resolves against the SCRIPT's URL, not the
    document's — the opposite of `<script src>`, `fetch()` and every href here.
    `./vendor/…` from `groups/` asks for `/groups/vendor/…`.
  - Prevention/fix: the specifier is `../vendor/tiptap/tiptap.bundle.esm.js`,
    and `watch.py` asserts that exact string.
  - Status: `RESOLVED` — found in the browser; the check exists because of it.

- **The last line typed before a day change landed on the WRONG day** —
  `RESOLVED`
  - When it happens: typing and immediately clicking ‹ or ›, which is the
    ordinary way somebody reviews yesterday after writing today.
  - Symptom: nothing is lost and nothing errors — the words are simply on the
    other day, on top of whatever was written there.
  - Root cause: `onSave: (text) => saveDay(day, text)` read the module's `day`
    at SAVE time, and the save it mattered for is the teardown flush, which
    fires after `day` has already moved.
  - Prevention/fix: `renderGroup` captures `paintedDay` and every row closes
    over that; `setDay` also tears the editor down before reassigning. Pinned
    by `watch.py`'s `const paintedDay = day;` / `saveDay(paintedDay,` check.
  - Status: `RESOLVED` — found by typing a word, clicking ‹, and reading the
    database.

## Recent Changes
- 2026-09-02: Folder created. Groups tab built — discovery card, consent gate,
  public directory, invite links, group bar, and a roster of member cards
  carrying the Learner Home's area bars.
- 2026-09-02 (second pass, Seth's redesign): a member is now a full-width ROW
  with two columns. The left one is unchanged; the right one is the day's
  three-state Tiptap checklist, ported from Delta Note's `js/subgoals/` along
  with the vendored engine (`../vendor/tiptap/`). New: a day picker above the
  roster, `groups_day.js`, `groups_lane.js`, `groups_checklist{,_doc}.js`, the
  `study_group_days` table and `GET`/`PUT /api/practice/groups/day`. The
  consent dialog now says the checklist is shared too, because it is. Two
  defects found in the browser and fixed with checks: the bundle's import
  specifier, and a flush filing an edit under the wrong day.
- 2026-09-02: Fixed in browser testing — the trio moved above `app.js` so
  invite links land here; `.dd-group-row` buttons no longer inherit
  `.primary`'s `width: 100%`; the "let anyone find it" tick stopped being
  styled as a field caption. Backend: the owner leaving now hands the group to
  the longest-standing remaining member.
- 2026-09-02 (third pass, critic findings): five defects the browser would not
  have shown on a good day, all found by reading the code against its own
  comments — a failed day read stuck on "Reading this day…", a silently
  discarded save error, two unordered writes, a read overtaking the teardown
  flush, and a late read mounting an editor into a hidden tab. Plus a
  `:focus-visible` ring on the date field, which had none. Each has a check in
  `watch.py`.
- 2026-09-02 (fourth pass, Seth's layout notes): the roster is ONE island. The
  day picker is its first row (`.dd-board` / `.dd-board-day`) and a member is a
  row inside it, separated by a rule, with your own row tinted rather than
  outlined. The editor's placeholder is gone — inside a task item it drew on
  top of the checkbox — and the date field's calendar glyph is masked art tinted
  with `--white`, the same token the ‹ › arrows take, instead of an `invert()`
  of whatever the browser drew (which left it black on the blue theme).
- 2026-09-02 (fifth pass, second critic round): a failed read of the GROUP
  itself no longer reads as "you are in no group" — `readMyGroup` answers
  `undefined` on failure and the page says so with a Try again, instead of
  offering an existing member the create/join card during an outage. The
  roster read now checks the tab is still open before calling `onGroup()`.
  Both checklist columns paint their boxes through one helper, so the X state
  is `indeterminate` + `aria-checked="mixed"` and each box carries its own
  line as its accessible name.
