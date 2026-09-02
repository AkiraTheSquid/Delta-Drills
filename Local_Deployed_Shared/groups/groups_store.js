/* ================================================================
   GROUPS STORE — the nine calls a study group is made of.

   The thin side of `backend/app/practice/groups_router.py`. Ported from
   Delta Note's `accountability_group_store.js`, and the division of
   labour is the same one:

   ── Two kinds of failure, two kinds of answer ────────────────────
   A READ is background work behind a readout that must keep working:
   `readMyGroup` and `readPublicGroups` swallow, log once, and answer
   `null`, which every caller treats as "no news" rather than as "you
   are in no group". Spelling a failed read as an empty directory would
   tell somebody nobody is running an open group during an outage.

   A CREATE, JOIN, LEAVE, ROTATE or RENAME is something a person just
   clicked. Those answer `{ group }` or `{ error }` and the card says
   which, because a button that silently does nothing is worse than one
   that says why.

   ── 🔴 No account, no call ───────────────────────────────────────
   Every endpoint here is behind `get_current_user`, and a guest's
   progress lives in this browser — there is nothing for a group to
   read. Asked on every boot, that is one 401 per guest per load on an
   app that already watches its request budget. `hasAccount()` answers
   it without the round trip and the card tells them to sign in.

   ── 🔴 The invite token is a CAPABILITY ──────────────────────────
   Anyone holding it is in the group and can read every member's
   mastery. It is shown behind a deliberate click, taken out of the
   address bar the moment it has been used, and replaceable by the
   owner without breaking anyone's membership.
   ================================================================ */

const DDGroupStore = (() => {
  const INVITE_PARAM = "group";

  /* One line per failure kind, not one per failure — this module is
     called on every arrival at the tab and a flaky network should not
     fill the console with the same sentence. */
  /* The tail of the day-checklist write chain: every save queues behind it
     and every read waits for it. `settled` is what keeps one rejection from
     poisoning the chain for the rest of the session. */
  let dayWrites = Promise.resolve();
  const settled = (promise) => Promise.resolve(promise).catch(() => {});

  const complained = new Set();
  const complain = (where, error) => {
    const message = String(error?.message || error || "failed");
    const key = `${where}:${message}`;
    if (complained.has(key)) return;
    complained.add(key);
    console.warn(`[groups] ${where}: ${message}`);
  };

  /* `apiFetch` is app.js's top-level `const` — a global lexical binding,
     not a window property in every build — so this is the same guarded
     read practice/activity-chart.js does. */
  const call = async (path, options) => {
    const fetcher = typeof apiFetch === "function" ? apiFetch : window.apiFetch;
    if (typeof fetcher !== "function") throw new Error("app not ready");
    const res = await fetcher(path, options);
    if (!res?.ok) {
      let detail = "";
      try {
        detail = (await res.json())?.detail || "";
      } catch (_) {
        /* A non-JSON body — an HTML error page from a proxy, or nothing at
           all. The status is still worth reporting. */
      }
      throw new Error(detail || `${path} ${res?.status || "failed"}`);
    }
    return res.json();
  };

  const post = (path, body) =>
    call(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });

  /* An ACTION's wrapper. The message is whatever the server said, because
     the server's refusals are already written for the person who clicked
     ("That group is full (12 members).", "Only the person who started the
     group can do that."). */
  const act = async (where, run) => {
    try {
      return { group: (await run())?.group || null };
    } catch (error) {
      complain(where, error);
      return { error: String(error?.message || "That did not work.") };
    }
  };

  /** Is anybody signed in? A guest holds a token but is not signed in as
   *  anyone — see `isSignedIn` in app.js, which is what this reads. */
  const hasAccount = () => window.DDIdentity?.isSignedIn?.() === true;

  /**
   * What to call you in a group, before you have said otherwise.
   *
   * 🔴 Never the email itself. A roster is shown to everyone in the group,
   * and a group is joined by anyone holding a link; publishing addresses to
   * it would make an invite link a way of harvesting them. The local part
   * with its punctuation softened is a starting point, and the server
   * softens it again for anything that slips through.
   */
  const defaultDisplayName = () => {
    const email = String(window.DDIdentity?.email?.() || "").trim();
    if (!email) return "Learner";
    return email.split("@")[0].replace(/[._-]+/g, " ") || "Learner";
  };

  return {
    hasAccount,
    defaultDisplayName,

    /** Your group, or `null` for "you are in none", or `null` for a failed
     *  read — the caller cannot tell them apart on purpose, because both
     *  mean "draw the discovery card" and one of them is transient. */
    /* 🔴 `null` IS "YOU ARE IN NO GROUP"; A FAILED READ IS `undefined`.
       The two used to be the same answer, and the page drew the create/join
       card on both — so an outage told a member of a group that they were in
       none, with the buttons to start another one right there. The endpoint
       itself is explicit about this (`{"group": null}`, never a 404), so the
       store can be too. */
    async readMyGroup() {
      if (!hasAccount()) return null;
      try {
        return (await call("/api/practice/groups/mine"))?.group || null;
      } catch (error) {
        complain("readMyGroup", error);
        return undefined;
      }
    },

    /** The listed groups, or `null` for a failed read. 🔴 `null` is NOT
     *  `[]` here: the directory says something about other people, and
     *  "nobody is running an open group" is not a claim this app can make
     *  on the evidence of a request that did not come back. */
    async readPublicGroups() {
      if (!hasAccount()) return null;
      try {
        const data = await call("/api/practice/groups/public");
        return Array.isArray(data?.groups) ? data.groups : [];
      } catch (error) {
        complain("readPublicGroups", error);
        return null;
      }
    },

    createGroup: (name, displayName, visibility) =>
      act("createGroup", () =>
        post("/api/practice/groups", { name, display_name: displayName, visibility })),

    joinGroup: (token, displayName) =>
      act("joinGroup", () =>
        post("/api/practice/groups/join", { token, display_name: displayName })),

    joinPublicGroup: (groupId, displayName) =>
      act("joinPublicGroup", () =>
        post("/api/practice/groups/join-public", {
          group_id: groupId,
          display_name: displayName,
        })),

    leaveGroup: (groupId) =>
      act("leaveGroup", () => post("/api/practice/groups/leave", { group_id: groupId })),

    rotateToken: (groupId) =>
      act("rotateToken", () =>
        post("/api/practice/groups/rotate-token", { group_id: groupId })),

    setVisibility: (groupId, visibility) =>
      act("setVisibility", () =>
        post("/api/practice/groups/visibility", { group_id: groupId, visibility })),

    setDisplayName: (displayName) =>
      act("setDisplayName", () =>
        post("/api/practice/groups/display-name", { display_name: displayName })),

    /* ---- the day checklists ------------------------------------- */

    /**
     * Every member's checklist for one day, keyed by `member_id`.
     *
     * `null` is a FAILED read and `{}` is a real empty answer, and the two
     * must stay apart: a member with no row for the day comes back as `""`,
     * so a caller that saw `{}` may honestly draw twelve empty columns.
     * Answering `{}` on a network failure would draw the same twelve empty
     * columns and quietly claim nobody wrote anything today.
     *
     * @param {string} day a LOCAL `YYYY-MM-DD` — see DDGroupsDay.todayKey
     */
    async readDay(day) {
      if (!hasAccount() || !day) return null;
      try {
        /* 🔴 NEVER READ ACROSS A PENDING WRITE. Changing the day tears the
           editor down, and that teardown flushes a save — then the new day is
           read. Change the day and change it straight back and this read can
           overtake that write, hand the editor the row as it was BEFORE the
           flush, and the next keystroke saves the stale document over the
           newer one. Waiting on the write chain costs nothing when there is
           no write pending, which is almost always. */
        await settled(dayWrites);
        const data = await call(`/api/practice/groups/day?date=${encodeURIComponent(day)}`);
        return data && data.entries && typeof data.entries === "object" ? data.entries : {};
      } catch (error) {
        complain("readDay", error);
        return null;
      }
    },

    /**
     * Store YOUR OWN checklist for one day.
     *
     * 🔴 Takes no member id, and the endpoint accepts none: the row is keyed
     * by the authenticated user. A group is joined by anyone holding a link,
     * so an endpoint that took a target would let any member rewrite
     * anybody's day.
     *
     * Answers `{ok}` / `{error}` rather than going through `act()`, which
     * exists to unwrap a `group` — there is no group in this answer.
     */
    async saveDay(day, payload) {
      if (!hasAccount() || !day) return { error: "Not signed in." };
      /* 🔴 ONE WRITE AT A TIME. Each save sends the WHOLE document, so two
         in flight together are a race whose loser is decided by the network:
         land the older one second and it overwrites the newer text with the
         text that came before it. Chaining them makes the last one typed the
         last one written, which is the only ordering a person would predict. */
      const run = settled(dayWrites).then(async () => {
        try {
          await call("/api/practice/groups/day", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ date: day, payload: String(payload || "") }),
          });
          return { ok: true };
        } catch (error) {
          complain("saveDay", error);
          return { error: String(error?.message || "That did not save.") };
        }
      });
      dayWrites = run;
      return run;
    },

    /* ---- the invite link ---------------------------------------- */

    /** The token out of whatever somebody pasted. People paste the whole
     *  link far more often than the token, and a field that only accepts
     *  the token is a field most people get wrong on the first go. */
    tokenFromPaste(raw) {
      const text = String(raw || "").trim();
      if (!text) return "";
      try {
        const fromQuery = new URL(text).searchParams.get(INVITE_PARAM);
        if (fromQuery) return fromQuery.trim();
      } catch (_) {
        /* Not a URL. It may still be the bare token, checked next. */
      }
      return /^[a-f0-9]{32}$/i.test(text) ? text : "";
    },

    /** An invite link, built by EDITING this page's address rather than
     *  re-assembling one from origin + pathname — re-assembly drops every
     *  other query parameter and the hash the app routes on. */
    inviteLink(token) {
      try {
        const url = new URL(String(window.location?.href || ""));
        url.searchParams.set(INVITE_PARAM, token);
        return url.toString();
      } catch (_) {
        return `?${INVITE_PARAM}=${encodeURIComponent(token)}`;
      }
    },

    /** The token this page was opened with, or `""`. */
    inviteFromLocation() {
      const params = new URLSearchParams(String(window.location?.search || ""));
      const token = String(params.get(INVITE_PARAM) || "").trim();
      return /^[a-f0-9]{32}$/i.test(token) ? token : "";
    },

    /** 🔴 Take the invite out of the address bar once it has been acted on.
     *  A token left in the URL is a token that gets pasted, bookmarked and
     *  shared in a screenshot of the address bar. */
    clearInviteFromLocation() {
      try {
        const url = new URL(String(window.location?.href || ""));
        if (!url.searchParams.has(INVITE_PARAM)) return;
        url.searchParams.delete(INVITE_PARAM);
        window.history?.replaceState?.({}, "", `${url.pathname}${url.search}${url.hash}`);
      } catch (_) {
        /* No history API, or a URL this browser will not parse. A token
           left in the bar is a smaller problem than a throw on the join
           path. */
      }
    },
  };
})();

window.DDGroupStore = DDGroupStore;
