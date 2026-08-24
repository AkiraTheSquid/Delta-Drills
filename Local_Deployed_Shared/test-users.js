/* ================================================================
   TEST-USERS.JS — demo the app as somebody else, without losing yourself

   WHY THIS EXISTS
     Showing the app to a person at a meetup means handing them a learner
     state that is not yours: an empty mastery record, no XP, no placement
     result, questions they have not already been shown. Doing that used to
     mean signing out (which drops you into the shared browser-wide guest
     account, so the second person inherits the first person's session) or
     making a throwaway Google account per person. Neither keeps the
     demos apart, and neither gets you back to your own account afterwards.

     A test user here is a REAL backend account, minted exactly the way
     guest-session.js mints a guest — POST /auth/signup with a generated
     address and a random password — and given a display NAME you choose.
     The roster lives in this browser, keyed by the email of the account
     that created it, so "your test users" follow your account and not the
     browser's current session. Switching to one swaps the token; the
     backend then has no idea it is a demo, which is the point: the
     diagnostic, the BKT student model, the lessons and the adaptive queue
     are the real ones, on a genuinely empty record.

   WHY SWITCHING RELOADS
     The same reason app.js's signOutAndReload does. practiceMode is
     detected ONCE at init (practice/mode.js detectPracticeMode), the
     practice surface hydrates its progress once at init, and xp.js reads
     its store key at load. Swapping the token in place would leave every
     one of those pointing at the previous identity, which is the failure
     that looks like "the test user has my XP".

   THE TWO KINDS OF LOCAL STATE
     1. Already per-identity. Anything keyed off `auth_email` isolates
        itself the moment the email changes — practice_progress_<email>
        (practice/storage.js) and its _session / _kc_exposure suffixes,
        adaptive_state_<email> (practice/adaptive.js), dd_xp_v1_<email>
        (xp.js). Nothing to do for these.
     2. GLOBAL, and therefore leaky. `drills_shown`, the ARENA-prereq
        equivalent, the session-setup preferences and the queued problem
        reports are single localStorage keys shared by whoever is signed
        in. Left alone, the second person you demo to never sees the
        questions the first one saw, and a report queued under a test
        user flushes into YOUR account the next time you come back.
        SHARED_KEYS below is that list: on every identity change the
        outgoing identity's values are PARKED in a per-email bucket and
        the incoming identity's are restored. The owner is parked and
        restored like everyone else, so nothing of yours is thrown away.

   WHAT THIS IS NOT
     Not a security boundary and not an admin feature. Signup is already
     open — this only skips the form and remembers the name you gave it.
     Anyone can already create as many accounts as they like.

     There is no server-side delete: the backend has /auth/signup,
     /auth/login and /auth/google and nothing else. "Remove" therefore
     forgets a test user locally and leaves an unreferenced account behind
     on the backend, and "Reset" mints a FRESH account under the same
     display name rather than wiping the old one. The UI says so rather
     than implying a deletion that did not happen.

   THE SURFACES LIVE NEXT DOOR. `test-users-ui.js` owns the Account-tab
   roster and the floating pill; this file owns the store, the identity
   swap and nothing that touches the DOM. They talk over `DDTestUsers`
   below, and a page that loads only this one is a working store with no
   way to reach it — never the other way round.

   LOADED after app.js and guest-session.js: it reads and writes app.js's
   top-level `authToken` / `authEmail` / `API_BASE` bindings, which do not
   exist until app.js has evaluated, and it must see guest-session.js's
   provisioning decision before it decides whether there is an owner to
   act on behalf of. Classic scripts share one global lexical scope, so
   those are the real bindings — `window.authToken = …` would create a
   shadow property nothing reads.
   ================================================================ */

(function installTestUsers(global) {
  /* Roster: { "<owner email>": [ {id, name, email, password, createdAt, lastUsedAt} ] }
     Keyed by OWNER so signing in as somebody else on the same browser does
     not show them your demo accounts, and so the roster survives the
     identity swap that acting-as performs. */
  const ROSTER_KEY = "dd_test_users_v1";
  /* Present ONLY while acting as a test user:
     { owner: {email, token}, user: {id, name, email} }
     The owner's token is stashed here because switching overwrites
     localStorage auth_token, and the way back is to put it straight back.
     Backend tokens are 30 days (backend app/config.py access_token_ttl_minutes),
     so a stashed one outlives any demo. */
  const SESSION_KEY = "dd_test_user_session_v1";
  /* Parked copies of the SHARED_KEYS, per identity email. */
  const SCRATCH_KEY = "dd_identity_scratch_v1";
  /* app.js's one-shot landing key (FIRST_RUN_TAB_KEY there), read once on the
     next load and cleared. sessionStorage on purpose: it describes the reload
     this file is about to trigger and must not outlive the tab. */
  const FIRST_RUN_TAB_KEY = "dd_first_run_tab";
  /* The page a first-time visitor lands on — the two-arrow fork, #page-welcome
     in index.html. NOT a tab; app.js reaches it by name only. */
  const ONBOARDING_TAB = "welcome";

  /* Well-formed for the backend's EmailStr validation and obviously not a
     person. Nothing is ever sent to it. */
  const TEST_EMAIL_DOMAIN = "test.delta-drills.app";
  const PASSWORD_BYTES = 24;
  const MAX_NAME = 40;

  /* Single global keys that are really per-learner. See the header. Adding
     one here is all that is needed to make it follow the identity. */
  const SHARED_KEYS = [
    "drills_shown",                 // practice/drills-catalog.js
    "drills_shown_schema",
    "arena_prereqs_temp_shown",     // stats/predicted-prereqs-temp.js
    "arena_prereqs_temp_shown_schema",
    "delta_drills_session_setup",   // practice/timer.js
    "problem_feedback_queue",       // practice/api.js
  ];

  // ── tiny helpers ────────────────────────────────────────────────
  const randomHex = (bytes) => {
    const buf = new Uint8Array(bytes);
    (global.crypto || global.msCrypto).getRandomValues(buf);
    return Array.from(buf, (b) => b.toString(16).padStart(2, "0")).join("");
  };

  const readJson = (key, fallback) => {
    try {
      const raw = localStorage.getItem(key);
      if (!raw) return fallback;
      const parsed = JSON.parse(raw);
      return parsed === null || parsed === undefined ? fallback : parsed;
    } catch (_) {
      return fallback;
    }
  };

  const writeJson = (key, value) => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (err) {
      console.warn("[test-users] could not write %s:", key, err);
      return false;
    }
  };

  const slug = (name) =>
    String(name || "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 18) || "user";

  // ── identity ────────────────────────────────────────────────────
  const session = () => {
    const s = readJson(SESSION_KEY, null);
    if (!s || !s.owner || !s.owner.email || !s.user || !s.user.email) return null;
    return s;
  };

  const isActing = () => !!session();

  /* Signed in as a real person, in the sense app.js means: holding a token
     that is not the browser-wide guest one. While acting as a test user
     that is TRUE of the test account too, which is why the owner question
     is answered from the session record first. */
  const signedInReal = () =>
    typeof isSignedIn === "function" ? isSignedIn() : !!authToken;

  /** The account these test users belong to — yours, even mid-demo. */
  const ownerEmail = () => {
    const s = session();
    if (s) return s.owner.email;
    return signedInReal() ? (authEmail || "") : "";
  };

  const readRoster = (owner) => {
    const all = readJson(ROSTER_KEY, {});
    const list = Array.isArray(all[owner]) ? all[owner] : [];
    return list.filter((u) => u && u.id && u.email && u.password);
  };

  const writeRoster = (owner, list) => {
    const all = readJson(ROSTER_KEY, {});
    all[owner] = list;
    return writeJson(ROSTER_KEY, all);
  };

  // ── the shared-key park/restore ─────────────────────────────────
  const parkIdentity = (email) => {
    if (!email) return;
    const all = readJson(SCRATCH_KEY, {});
    const bucket = {};
    SHARED_KEYS.forEach((k) => {
      const v = localStorage.getItem(k);
      if (v !== null) bucket[k] = v;
      localStorage.removeItem(k);
    });
    all[email] = bucket;
    writeJson(SCRATCH_KEY, all);
  };

  const unparkIdentity = (email) => {
    const bucket = readJson(SCRATCH_KEY, {})[email] || {};
    /* Clear first, unconditionally: park() has normally emptied these
       already, but a switch that was interrupted between the two halves
       would otherwise hand the incoming identity the outgoing one's list. */
    SHARED_KEYS.forEach((k) => localStorage.removeItem(k));
    Object.keys(bucket).forEach((k) => {
      if (SHARED_KEYS.includes(k)) localStorage.setItem(k, bucket[k]);
    });
  };

  /** Forget everything this app stores under one email. Used by Reset and
      Remove, so a re-used display name does not inherit the old record. */
  const forgetLocalState = (email) => {
    if (!email) return;
    const prefixes = [
      `practice_progress_${email}`, // and its _session / _kc_exposure suffixes
      `adaptive_state_${email}`,
      `dd_xp_v1_${email}`,
    ];
    const doomed = [];
    for (let i = 0; i < localStorage.length; i += 1) {
      const key = localStorage.key(i);
      if (key && prefixes.some((p) => key === p || key.startsWith(`${p}_`))) doomed.push(key);
    }
    doomed.forEach((k) => localStorage.removeItem(k));
    const scratch = readJson(SCRATCH_KEY, {});
    if (email in scratch) {
      delete scratch[email];
      writeJson(SCRATCH_KEY, scratch);
    }
  };

  // ── backend ─────────────────────────────────────────────────────
  /* Same shape as guest-session.js postAuth, and for the same reason: the
     caller has to tell "the backend refused these credentials" (401 — the
     account is gone) apart from "the backend could not answer" (0, 429,
     5xx). Collapsing both to null is how a bad minute costs a test user
     its progress. */
  const postAuth = async (path, credentials) => {
    try {
      const res = await fetch(`${API_BASE}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(credentials),
      });
      if (!res.ok) return { token: null, status: res.status };
      const data = await res.json().catch(() => ({}));
      return { token: data.access_token || null, status: res.status };
    } catch (err) {
      console.warn("[test-users] %s failed:", path, err);
      return { token: null, status: 0 }; // 0 = never reached the backend
    }
  };

  const mintCredentials = (name) => ({
    email: `test-${slug(name)}-${randomHex(4)}@${TEST_EMAIL_DOMAIN}`,
    password: randomHex(PASSWORD_BYTES),
  });

  /** Log in, creating the account if the backend has never heard of it.
      Returns a token, or null with `reason` set for the message line. */
  const tokenFor = async (user) => {
    const creds = { email: user.email, password: user.password };
    const login = await postAuth("/auth/login", creds);
    if (login.token) return { token: login.token };
    if (login.status !== 401) {
      return { token: null, reason: "The backend did not answer. Nothing was changed." };
    }
    /* 401 means these credentials are not an account — a reset database, a
       different backend, or a test user whose signup never completed. The
       display name and its local record are still meaningful, so sign the
       same credentials up rather than stranding the row. */
    const signup = await postAuth("/auth/signup", creds);
    if (signup.token) return { token: signup.token };
    if (signup.status === 409) {
      /* The address exists but the password does not match it — the only
         way back is a fresh account, which is what Reset is for. */
      return {
        token: null,
        reason: "That test account exists with a different password. Use Reset to start it over.",
      };
    }
    return { token: null, reason: "Could not sign this test user in. Nothing was changed." };
  };

  // ── switching ───────────────────────────────────────────────────
  const adoptToken = (token, email) => {
    authToken = token;
    authEmail = email;
    localStorage.setItem("auth_token", token);
    localStorage.setItem("auth_email", email);
    /* A test user is NOT the browser-wide guest. Leaving this set would put
       the guest banner back and let guest-session.js's silent refresh log
       the guest account in over the top of the demo. */
    localStorage.removeItem("dd_auth_is_guest");
  };

  /* A test user who has never been used IS a first-time learner, and the app
     already has a first screen for one: the welcome fork, with the optional
     "learn how this works" path on one side and the placement test on the
     other. app.js picks that screen on `authToken` being absent, which a test
     user can never satisfy — it holds a real token from the moment it is
     minted — so the demo would otherwise open on the practice surface and skip
     the fork that the person being handed the app is exactly the audience for.
     Asking for it here rather than teaching app.js about test users keeps the
     identity rules in this file.

     Best-effort: a browser that refuses the write costs the demo its fork, and
     nothing else, so it must not abort a switch that is otherwise fine. */
  const requestOnboardingLanding = () => {
    try {
      sessionStorage.setItem(FIRST_RUN_TAB_KEY, ONBOARDING_TAB);
    } catch (err) {
      console.warn("[test-users] could not ask for the onboarding landing:", err);
    }
  };

  /** Become `user`. Safe to call while already acting as somebody else. */
  const actAs = async (user) => {
    const owner = ownerEmail();
    if (!owner) return { ok: false, reason: "Sign in first — test users belong to an account." };
    const current = session();
    if (current && current.user.id === user.id) return { ok: true };

    const ownerToken = current ? current.owner.token : authToken;
    if (!ownerToken) {
      return { ok: false, reason: "No saved session for your own account, so there would be no way back." };
    }

    const got = await tokenFor(user);
    if (!got.token) return { ok: false, reason: got.reason };

    /* Nothing above this line has touched stored state; from here down the
       swap is committed, and the reload is what makes it take effect.

       🔴 THE WAY BACK IS WRITTEN FIRST, and a failure to write it aborts.
       This record is the ONLY copy of the owner's token once adoptToken
       overwrites `auth_token`. Swapping first and recording afterwards
       means a storage write that does not land (quota, a browser refusing
       to persist) reloads the page as the test user with the owner's
       session gone and nothing to restore it from — the one failure this
       whole file exists to prevent. */
    if (
      !writeJson(SESSION_KEY, {
        owner: { email: owner, token: ownerToken },
        user: { id: user.id, name: user.name, email: user.email },
      })
    ) {
      return {
        ok: false,
        reason: "This browser would not save the way back to your account, so nothing was switched.",
      };
    }
    parkIdentity(authEmail);
    adoptToken(got.token, user.email);
    unparkIdentity(user.email);

    /* BEFORE lastUsedAt is stamped below, which is the thing that says this
       account has been demoed. Reset clears it back to null, so a test user
       started over gets the first screen again — which is the whole point of
       starting one over. */
    if (!user.lastUsedAt) requestOnboardingLanding();

    const list = readRoster(owner).map((u) =>
      u.id === user.id ? { ...u, lastUsedAt: new Date().toISOString() } : u,
    );
    writeRoster(owner, list);

    global.location.reload();
    return { ok: true };
  };

  /** Put your own account back. */
  const stopActing = () => {
    const s = session();
    if (!s) return false;
    parkIdentity(authEmail);
    adoptToken(s.owner.token, s.owner.email);
    unparkIdentity(s.owner.email);
    localStorage.removeItem(SESSION_KEY);
    global.location.reload();
    return true;
  };

  // ── roster operations ───────────────────────────────────────────
  const addUser = async (name) => {
    const owner = ownerEmail();
    if (!owner) return { ok: false, reason: "Sign in first — test users belong to an account." };
    const clean = String(name || "").trim().slice(0, MAX_NAME);
    if (!clean) return { ok: false, reason: "Give the test user a name." };

    const creds = mintCredentials(clean);
    const signup = await postAuth("/auth/signup", creds);
    if (!signup.token) {
      return {
        ok: false,
        reason:
          signup.status === 0
            ? "Could not reach the backend, so no test user was created."
            : `The backend refused to create the account (${signup.status}).`,
      };
    }
    /* Written only after signup succeeded, so a failed attempt does not
       leave a row whose credentials will never log in. */
    const user = {
      id: randomHex(8),
      name: clean,
      email: creds.email,
      password: creds.password,
      createdAt: new Date().toISOString(),
      lastUsedAt: null,
    };
    writeRoster(owner, readRoster(owner).concat([user]));
    return { ok: true, user };
  };

  const renameUser = (id, name) => {
    const owner = ownerEmail();
    const clean = String(name || "").trim().slice(0, MAX_NAME);
    if (!owner || !clean) return false;
    writeRoster(
      owner,
      readRoster(owner).map((u) => (u.id === id ? { ...u, name: clean } : u)),
    );
    const s = session();
    if (s && s.user.id === id) writeJson(SESSION_KEY, { ...s, user: { ...s.user, name: clean } });
    return true;
  };

  /** Start this test user over: a brand-new backend account under the same
      display name. The old one is abandoned, not deleted — there is no
      endpoint for that. Refuses while you are acting AS that user, because
      the token in hand would stop being theirs. */
  const resetUser = async (id) => {
    const owner = ownerEmail();
    if (!owner) return { ok: false, reason: "Sign in first." };
    const s = session();
    if (s && s.user.id === id) {
      return { ok: false, reason: "Go back to your own account before resetting this one." };
    }
    const list = readRoster(owner);
    const user = list.find((u) => u.id === id);
    if (!user) return { ok: false, reason: "No such test user." };

    const creds = mintCredentials(user.name);
    const signup = await postAuth("/auth/signup", creds);
    if (!signup.token) {
      return { ok: false, reason: "Could not create the replacement account. Nothing was changed." };
    }
    forgetLocalState(user.email);
    writeRoster(
      owner,
      list.map((u) =>
        u.id === id
          ? { ...u, email: creds.email, password: creds.password, createdAt: new Date().toISOString(), lastUsedAt: null }
          : u,
      ),
    );
    return { ok: true };
  };

  const removeUser = (id) => {
    const owner = ownerEmail();
    if (!owner) return false;
    const s = session();
    if (s && s.user.id === id) return false; // can't delete the seat you're sitting in
    const list = readRoster(owner);
    const user = list.find((u) => u.id === id);
    if (!user) return false;
    forgetLocalState(user.email);
    writeRoster(owner, list.filter((u) => u.id !== id));
    return true;
  };
  // ── where this UI is allowed to appear ──────────────────────────
  /* The Colab edition is the same code on a different host and is the
     surface Seth does NOT demo this way; a solo route (/notebooks,
     /knowledge-graph, ?embed=1) is a chromeless single page, often in an
     iframe, and a floating pill over an embed is just litter. */
  const surfaceAllowed = () => {
    const root = document.documentElement;
    if (root.classList.contains("dd-colab-edition")) return false;
    if (root.classList.contains("dd-solo")) return false;
    if (global.parent !== global) return false;
    return true;
  };

  // ── the record vs. the token ────────────────────────────────────
  /* The session record and the token in localStorage can disagree: Log out
     (app.js signOutAndReload) clears the token without knowing this file
     exists, and so does an expired-token fallback. Believing the record
     over the token would show a "Test user: Alice" pill on a session that
     is no longer Alice's. The token wins; the record is dropped. The
     ROSTER is never touched here — the names are the thing worth keeping. */
  const reconcile = () => {
    const s = readJson(SESSION_KEY, null);
    if (!s || !s.user || !s.user.email) return;
    const live = (localStorage.getItem("auth_email") || "").trim();
    if (live === s.user.email) return;
    console.warn(
      "[test-users] the signed-in account (%s) is no longer the test user this " +
        "session recorded (%s) — dropping the acting-as record. Your test users are kept.",
      live || "(none)",
      s.user.email,
    );
    localStorage.removeItem(SESSION_KEY);
  };

  /* Run at EVAL time, not on DOMContentLoaded: `test-users-ui.js` asks
     isActing() as soon as it boots, and a stale record answered once is a
     "Test user: Alice" pill over somebody else's session. */
  reconcile();

  global.DDTestUsers = {
    /** Longest display name accepted. The UI's inputs read this. */
    MAX_NAME,
    /** Is this session pretending to be somebody? */
    isActing,
    /** The acting-as record, or null. */
    session,
    /** The account the roster belongs to — yours, even mid-demo. */
    ownerEmail,
    /** Everyone on the roster for the current owner. */
    list: () => readRoster(ownerEmail()),
    add: addUser,
    actAs,
    stopActing,
    rename: renameUser,
    reset: resetUser,
    remove: removeUser,
    /** May this deploy show the test-user surfaces at all? */
    surfaceAllowed,
  };
})(window);
