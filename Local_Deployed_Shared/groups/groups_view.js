/* ================================================================
   GROUPS PAGE — everybody's readiness and everybody's day, side by
   side.

   The Groups tab (Seth, 2026-09-02). One ROW per member, two columns:
   their area mastery on the left — the SAME `.placement-area*` rows
   the Learner Home draws, through the same `PlacementResults` — and
   on the right the three-state checklist they wrote for the day the
   picker at the top is showing.

   This file is the composition and the state; the row itself is
   `groups_lane.js`, the day control is `groups_day.js`, and the
   checklist document is `groups_checklist{,_doc}.js`.

   ── 🔴 THE DAY IS READ SEPARATELY FROM THE ROSTER ────────────────
   `/groups/mine` recomputes every member's posterior — one practice
   state load and one recompute per member. The day picker is a
   control people click through a week, so walking back seven days
   would be eighty-four state loads if the checklists rode along with
   the roster. `/groups/day` is its own endpoint precisely so a day
   change costs one cheap read.

   ── 🔴 A REPAINT TEARS THE EDITOR DOWN FIRST ─────────────────────
   Your column is a live ProseMirror view. `render()` replaces the
   root's children, so anything already mounted has to be destroyed
   BEFORE that — the teardown is what flushes the half-second save
   debounce, and without it the last words typed before a click on ▶
   are the ones that vanish.

   ── WHEN IT LOADS ────────────────────────────────────────────────
   On arrival at the tab, on a day change, and after a mutating call.
   No poll: a roster changes when somebody joins, and a page that
   refreshed itself would be a request per open tab per interval —
   and a repaint under a live caret.
   ================================================================ */

const DDGroups = (() => {
  const HOST_ID = "groups-root";

  let current = null;      // the group as last read, or null
  let loading = false;
  let fetchSeq = 0;        // only the newest roster read may render
  let daySeq = 0;          // and only the newest day read

  /* The day on screen, and its checklists keyed by member_id. `null`
     entries mean "still being read", which the column says out loud —
     `{}` would mean "read, and nobody wrote anything", a claim this page
     may only make once the request has come back. */
  let day = "";
  let entries = null;
  /* "loading" | "ready" | "failed" — see loadDay. `entries === null` alone
     cannot tell a read still in flight from a read that came back empty
     handed, and the column has to say a different sentence for each. */
  let dayState = "loading";
  /* 🔴 THE TAB CAN BE LEFT WHILE A READ IS IN FLIGHT. `suspend()` tears
     the editor down, but the promise it was racing does not know that: it
     lands, repaints the hidden page and mounts a fresh ProseMirror nobody
     will ever destroy. This flag is what the late answer checks. */
  let active = false;

  const host = () => document.getElementById(HOST_ID);
  const store = () => window.DDGroupStore;
  const join = () => window.DDGroupsJoin;
  const days = () => window.DDGroupsDay;
  const lane = () => window.DDGroupsLane;

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const myMemberId = () => (current && current.member_id) || "";

  /* ---- the day --------------------------------------------------- */

  /**
   * Read one day's checklists.
   *
   * The sequence guard is not decoration: clicking ‹ four times starts
   * four reads and they can come back in any order. Without it the page
   * settles on whichever the network happened to finish last, which is a
   * roster showing Tuesday under a picker that says Friday.
   */
  const loadDay = async (key) => {
    const seq = ++daySeq;
    const answer = await store().readDay(key);
    if (seq !== daySeq || !active) return;
    /* `null` is a failed read. Keeping it as `{}` here would draw every
       column blank and quietly say nobody wrote anything today, so the state
       is carried separately and the row builder renders the honest "could
       not read" instead. */
    entries = answer;
    dayState = answer === null ? "failed" : "ready";
    render();
  };

  /** Re-read the day on screen. The button beside the picker after a failure. */
  const retryDay = () => {
    if (!day) return;
    lane()?.destroyAll();
    entries = null;
    dayState = "loading";
    render();
    void loadDay(day);
  };

  const setDay = (key) => {
    if (!key || key === day) return;
    /* 🔴 Tear the editor down BEFORE `day` moves. The teardown flushes the
       pending save, and its `onSave` was closed over the day it was mounted
       on — but a reader should not have to trace that to see this is safe,
       so the window in which the two could disagree is closed here as well. */
    lane()?.destroyAll();
    day = key;
    entries = null;
    dayState = "loading";
    render();          // repaint immediately: the picker must feel instant
    void loadDay(key); // …and the columns fill in when the read lands
  };

  /** Write your own checklist for the day it was typed on.
   *
   *  🔴 `key` MUST be a day captured when the editor was built — never the
   *  module's live `day`. This function is reached two ways: from a
   *  debounce that fired while the page sat still, and from the editor's
   *  TEARDOWN, which happens because the day just changed. Read the live
   *  `day` on that second path and the last sentence typed on Wednesday is
   *  filed under Tuesday: it is not lost, so nothing looks broken — it is
   *  simply on the wrong day, and it has overwritten whatever was there.
   *  Found in the browser by typing a word and clicking ‹. */
  const saveDay = (key, payload) => {
    /* Keep the local copy in step so the next repaint (a day change and
       back, a group action) re-mounts the editor on what was typed rather
       than on what the last read returned. */
    if (key === day && entries && myMemberId()) entries[myMemberId()] = payload;
    /* Returned, not `void`ed: the column shows "Not saved: …" when this
       answers `{error}`, which is the only way a person finds out that the
       debounce they had already stopped thinking about hit the size cap or
       a dead connection. */
    return store().saveDay(key, payload);
  };

  /* ---- the page ---------------------------------------------------- */

  const renderGroup = (root, group) => {
    root.replaceChildren();
    root.appendChild(
      join().buildGroupBar({
        group,
        onChanged: (next) => {
          /* The server answers the whole group on every mutating call, so
             the click that changed something repaints on that click rather
             than one round trip later. `null` means we just left. */
          current = next;
          render();
        },
      })
    );

    /* 🔴 ONE ISLAND FOR EVERYBODY (Seth, 2026-09-02). The day and every member
       live in the SAME panel: the picker is the board's first row, not a
       control hovering above a stack of separate cards. A person is a row
       inside it, and the two columns of every row line up down the page
       because there is only one edge for them to line up against. */
    const board = el("div", "dd-board");
    const dayRow = el("div", "dd-board-day");
    const bar = days().buildPicker({ value: day, onChange: setDay });
    if (dayState === "failed") {
      const retry = el("button", "dd-day-retry", "Retry");
      retry.type = "button";
      retry.addEventListener("click", retryDay);
      bar.appendChild(retry);
    }
    dayRow.appendChild(bar);
    board.appendChild(dayRow);

    const you = group.member_id;
    const roster = el("div", "dd-member-grid");
    /* 🔴 THE DAY THIS PAINT IS OF, captured. Every editor built below closes
       over THIS value, not over the module's `day`, so a save that fires
       after the day has moved on — which is exactly what a teardown flush
       is — still names the day the words were typed on. */
    const paintedDay = day;
    (group.members || []).forEach((member) => {
      const isYou = member.member_id === you;
      const payload = entries === null ? null : entries[member.member_id] || "";
      roster.appendChild(
        lane().buildRow({
          member,
          isYou,
          day: paintedDay,
          dayState,
          payload,
          onSave: (text) => saveDay(paintedDay, text),
        })
      );
    });
    board.appendChild(roster);
    root.appendChild(board);

    const foot = el("p", "dd-group-note dd-group-foot");
    foot.textContent =
      "Each bar is that person's estimated readiness in one area of the curriculum, on the same scale as your own Learner Home. An area nobody probed is a starting assumption, not a measurement. The checklist beside it is what they wrote for the day above — click a box to cycle it: open, done, or won't do.";
    root.appendChild(foot);
  };

  /** The card that replaces the page when the group itself could not be read. */
  const renderReadFailure = (root) => {
    root.replaceChildren();
    const card = el("div", "dd-group-card");
    card.appendChild(el("h2", "", "Your group could not be read"));
    card.appendChild(
      el(
        "p",
        "dd-group-note",
        "The connection to the server failed. This is not the same as being in no group — nothing has changed on their side."
      )
    );
    const again = el("button", "ghost", "Try again");
    again.type = "button";
    again.addEventListener("click", () => void refresh());
    card.appendChild(again);
    root.appendChild(card);
  };

  const render = () => {
    const root = host();
    if (!root || !active) return;
    /* 🔴 Before anything replaces the DOM. See the header. */
    lane()?.destroyAll();
    if (current) renderGroup(root, current);
    else join().renderDiscovery(root, { onJoined: (group) => { current = group; onGroup(); } });
  };

  /** Everything that has to happen once we know we are in a group. */
  const onGroup = () => {
    if (!day) day = days().todayKey();
    entries = null;
    dayState = "loading";
    render();
    void loadDay(day);
  };

  /**
   * Read the group and draw the page.
   *
   * Guarded against overlap: arriving at the tab twice quickly starts two
   * reads and the slower one must not paint over the faster one's answer.
   */
  const refresh = async () => {
    const root = host();
    if (!root) return;
    active = true;
    if (!store()?.hasAccount()) {
      /* Signed out. `readMyGroup` would answer null anyway; this skips the
         401 and goes straight to the card that says why. */
      current = null;
      render();
      return;
    }
    const seq = ++fetchSeq;
    if (!current && !loading) {
      loading = true;
      lane()?.destroyAll();
      root.replaceChildren(el("p", "dd-group-note", "Loading your group…"));
    }
    const group = await store().readMyGroup();
    /* 🔴 `!active` as well as the sequence. Leaving the tab mid-read has to
       stop what follows: `onGroup()` starts a day read, and a day read that
       lands after the tab is opened again paints — and a paint MOUNTS AN
       EDITOR. The sequence number alone does not know the tab is gone. */
    if (seq !== fetchSeq || !active) return;
    loading = false;
    if (group === undefined) {
      /* A failed read, not an empty one. Keep whatever is on screen if we
         have a group already; otherwise say so, because the alternative is
         the create/join card telling a member they are in no group. */
      if (!current) renderReadFailure(root);
      return;
    }
    current = group;
    if (group) onGroup();
    else render();
  };

  /* 🔴 THE INVITE ROUTE IS IN app.js, NOT HERE. An invite in the address
     bar is a person who clicked a link somebody sent them, and the app
     opens on the Learner Home — so the boot call at the bottom of app.js
     is what has to land them on this tab. Doing it from here would mean a
     second switchTab AFTER the first one had already run its
     practice-refresh and placement-lock passes. */

  return {
    refresh,
    reset() {
      lane()?.destroyAll();
      current = null;
      entries = null;
      dayState = "loading";
    },
    /* Leaving the tab is a teardown, not a pause: it flushes the save
       debounce. app.js calls this from switchTab on the way out. */
    suspend() {
      lane()?.destroyAll();
      /* 🔴 Both, and neither is enough alone. `active = false` stops a
         read that is already in flight from repainting a page nobody is
         looking at — a repaint mounts an editor — and the bumped sequence
         stops it from being mistaken for the newest read when the tab is
         opened again. */
      active = false;
      daySeq += 1;
    },
  };
})();

window.DDGroups = DDGroups;
