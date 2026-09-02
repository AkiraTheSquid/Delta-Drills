/* ================================================================
   APP.JS — Core app logic: auth, tab switching, API helper, account
   ================================================================ */

const tabs = document.querySelectorAll(".tab");
const authOnlyTabs = document.querySelectorAll(".auth-only");
const guestOnlyTabs = document.querySelectorAll(".guest-only");
const pages = document.querySelectorAll(".page");
const authStatus = document.getElementById("auth-status");
const logoutButton = document.getElementById("logout-button");
const accountForm = document.getElementById("account-form");
const accountMessage = document.getElementById("account-message");
const accountLogout = document.getElementById("account-logout");

const isLocalHost = ["localhost", "127.0.0.1", "0.0.0.0"].includes(window.location.hostname);
const defaultApiBase = isLocalHost
  ? "http://localhost:8000"
  : "https://delta-drills-backend.fly.dev";
let API_BASE = localStorage.getItem("api_base") || defaultApiBase;
let authToken = localStorage.getItem("auth_token") || "";
let authEmail = localStorage.getItem("auth_email") || "";

// --- Guest sessions ---------------------------------------------------
// A visitor with no account still gets the WHOLE app: guest-session.js
// mints one against the backend on first load and keeps its credentials in
// this browser. That is what makes the diagnostic, the lessons and the BKT
// student model — every one of them `practiceMode === "backend"` only —
// work without a sign-in.
//
// The consequence is that `authToken` stopped meaning "a person signed in".
// It means "this session can call the backend". Anything about IDENTITY —
// the guest banner, the topbar email, the Account tab, which tab we land on
// — must ask isSignedIn(); anything about CAPABILITY still asks authToken.
const GUEST_CREDENTIALS_KEY = "dd_guest_credentials";
const GUEST_ACTIVE_KEY = "dd_auth_is_guest";
// Where the learner was when a recovery reload took the page out from under
// them. Written by guest-session.js, read once on the next load.
const RECOVERED_TAB_KEY = "dd_recovered_tab";
// The landing page for a session that is starting as somebody who has never
// used the app. Written by test-users.js immediately before the reload that
// commits an identity swap into a NEVER-USED test user, and read once here.
// A test user holds a real token, so the `authToken ? "practice"` line at the
// bottom of this file would otherwise drop a first-time demo learner straight
// onto the work and skip the fork every other first-time visitor gets.
const FIRST_RUN_TAB_KEY = "dd_first_run_tab";
const isGuestSession = () => localStorage.getItem(GUEST_ACTIVE_KEY) === "1";
const isSignedIn = () => !!authToken && !isGuestSession();

// MIGRATION FLAG (2026-05-30): route auth + practice to the Fly backend
// (own JWT auth + Neon Postgres + BKT gate) instead of Supabase, in prod too.
// REVERSIBLE: set false to restore Supabase auth + supabase practice mode
// (the Supabase code paths below + getPracticeMode's supabase branch stay intact).
window.DELTA_USE_BACKEND = true;

// Guests can now use the learning surface (practice/drills) without an
// account — progress is saved locally only (see getPracticeMode → "local").
// Only account-management / admin tabs still require a real login.
//
// Account is NOT in this list: it now hosts the basic/advanced mode toggle,
// which is a display preference kept in localStorage and is the only way to
// reach the advanced tabs. Blocking it would leave a signed-out visitor with
// no route into advanced mode at all. The credential controls on that page
// are already keyed off `authToken` and render as "Not signed in" for guests.
const guestBlockedTabs = ["split-tool"];

// --- Basic / Advanced app mode ---------------------------------------
// Basic is the DEFAULT and the app most people should see: Practice, the
// Diagnostic and the explainer tab. Advanced adds back the tabs that
// show the machinery. Flipped from the toggle on the Account tab; stored
// per-browser, like api_base and the other local prefs.
// Nothing is unloaded — the pages stay in the DOM and their scripts still
// run; only the nav entries (and the one in-page CTA that points at a
// hidden tab) are taken out, and switchTab refuses to route to them.
/* Knowledge Graph is NOT in this list any more. It is one of the five rows in
   the topbar account menu (index.html #account-menu), which is the only
   navigation basic mode has — putting a menu row behind a mode the menu itself
   is the way to reach would have made it a row that silently routes to Practice
   instead. Seth asked for it there by name, 2026-08-24. */
/* 🔴 "courses" IS NOT IN THIS LIST ANY MORE EITHER (Seth, 2026-09-01: "bring
   back the courses tab, but just in the account and settings ... drop-down").
   It is a row in the topbar account menu now, for exactly the reason Knowledge
   Graph is: that menu IS the navigation basic mode has, and switchTab REFUSES
   to route to an advanced-only tab while basic mode is on — so a Courses row in
   the menu would have opened Practice and read as a dead button. The `.tab`
   itself still exists and still only shows in the advanced strip; what changed
   is that the ROUTE is no longer gated. */
const advancedOnlyTabs = ["notebooks", "targeted-practice"];
const ADVANCED_MODE_KEY = "dd_advanced_mode";
const isAdvancedMode = () => localStorage.getItem(ADVANCED_MODE_KEY) === "1";
const isAdvancedOnlyTab = (tabName) => advancedOnlyTabs.includes(tabName);

/* TAB NAMES THAT USED TO EXIST. "why-this-app" and "how-to-use" were two tabs
   until 2026-08-23 and are one page now. solo-route.js keeps the two PATHNAMES
   working, but a name also reaches switchTab from places a URL alias cannot
   reach: `dd_recovered_tab` in sessionStorage, written by guest-session.js
   before a recovery reload. A reload that crosses a deploy boundary — the old
   build wrote the name, the new build reads it — would ask for a page that is
   not in the document any more. */
const renamedTabs = {
  "why-this-app": "learn-about-app",
  "how-to-use": "learn-about-app",
  /* 🔴 AND "diagnostic", which is an ALIAS AGAIN rather than a redirect to a
     different surface. The Placement test was merged into the Learner Home on
     2026-08-24 and split back out on 2026-09-01 (Seth: "keep the interface for
     the diagnostic ... separate ... it only gets displayed whenever you click
     on the drop-down one and you go to it specifically"), so the page exists
     once more — under the learner-facing name this time. The internal name
     `diagnostic` still arrives from three live places: `/diagnostic` in
     solo-route.js, a stale `dd_recovered_tab`, and practice/events.js. All
     three mean the placement, and all three now land on it. */
  diagnostic: "placement",
};

/* `opts.leavingPlacement` is the ONE way past the practice->placement redirect
   below, and it is set by the placement page's own exit button. Everything
   else that asks for Practice mid-test is a learner who wandered, and the test
   is what they should see; the button that says "go to practice" is a learner
   who read the sentence and meant it. */
const switchTab = (tabName, opts) => {
  tabName = renamedTabs[tabName] || tabName;
  /* 🔴 AND THE GENERAL CASE, because the rename is only the instance we know
     about. The `pages.forEach` line below hides EVERY page when the name
     matches none of them — no error, no console warning, just a blank app
     under a topbar. Any stale stored tab, any [data-goto-tab] typo
     and any future rename lands here, so the fallback is the same pair the
     boot call at the bottom of this file chooses between. */
  if (!document.getElementById(`page-${tabName}`)) {
    tabName = authToken ? "practice" : "welcome";
  }
  if (guestBlockedTabs.includes(tabName) && !authToken) {
    // No forced login page anymore — send guests to practice; the guest
    // banner is the standing CTA to log in.
    tabName = "practice";
  }
  if (isAdvancedOnlyTab(tabName) && !isAdvancedMode()) {
    // A tab that isn't in the nav must not be reachable by other means
    // either — e.g. a [data-goto-tab] button that outlived a mode flip.
    // EXCEPT a solo pathname deep link (/notebooks, /knowledge-graph): that
    // URL is an explicit request for that one page, it renders without app
    // chrome anyway, and it's what embeds point at. Redirecting it would
    // serve a chromeless Practice page to someone who asked for a notebook.
    if (window.DDSoloRoute?.read?.() !== tabName) tabName = "practice";
  }
  /* 🔴 THE PRACTICE/PLACEMENT LOCK, BACK WITH THE PAGE IT PROTECTS. Practice
     and the placement test share ONE editor and one `PracticeAPI.currentQuestion`
     — there is a single `.practice-container` and diagnostic-page.js re-parents
     it between the two pages — so a route to Practice while a probe is on
     screen would drag the probe onto the Learner Home and render it under the
     practice page's name (Seth, 2026-08-23). Two pages again means that is
     reachable again, so the redirect is again.

     It is a REDIRECT, not a disabled tab: the learner asked to leave a test
     they are in the middle of, and the honest answer is the test, not a dead
     control. `#placement-skip-btn` is the way out that does not lose the probe.

     🔴 `running` ONLY, not "a placement exists". A completed or not-yet-started
     placement holds nothing, and locking on it would strand a learner on the
     placement page every time they pressed Learner home. */
  if (
    tabName === "practice" &&
    !opts?.leavingPlacement &&
    window.DiagnosticPage?.isRunning?.() === true &&
    document.getElementById("page-placement")
  ) {
    tabName = "placement";
  }
  tabs.forEach((t) => t.classList.toggle("active", t.dataset.tab === tabName));
  pages.forEach((p) => p.classList.toggle("hidden", p.id !== `page-${tabName}`));
  /* #page-welcome is a PAGE WITHOUT A TAB — the two-arrow fork a first-time
     visitor lands on. It is reached by name from the boot call at the bottom
     of this file and by nothing else, and no `.tab` carries that name, so the
     line above simply leaves the strip with nothing active, which is correct.

     The body class is here because the fork has to be the ONLY thing on that
     screen (Seth, 2026-08-23) and the guest banner lives OUTSIDE every
     `.page`, so no page-scoped rule can reach it. styles/learn-about.css. */
  document.body.classList.toggle("dd-welcome", tabName === "welcome");
  // Returning to Practice normally re-fetches the question so a preference
  // change takes effect. But that fetch is DESTRUCTIVE to a session in
  // progress: the timer's resume check is
  // `resumeReady = _questionId() === pausedState.questionId`, so swapping in a
  // different question makes the saved one look gone — Resume greys out with
  // "Saved question is no longer available", and the re-render leaves the
  // submit area hidden, which is the "no submit button" dead end. A session
  // that is running or paused owns the question; leave it alone.
  // A lesson page is held the same way. The "See in knowledge graph" button in
  // the lesson topbar switches tabs on purpose; refetching on the way back would
  // replace the lesson the learner is mid-way through with a question.
  const sessionHoldsQuestion =
    document.body.classList.contains("lesson-mode") ||
    (typeof PracticeSession !== "undefined" &&
      (PracticeSession.isActive() || PracticeSession.hasPausedSession?.()));
  if (
    tabName === "practice" &&
    !sessionHoldsQuestion &&
    typeof refreshPracticeQuestionForPreferences === "function"
  ) {
    refreshPracticeQuestionForPreferences().catch((err) => {
      console.warn("[practice] failed to refresh preferences:", err);
    });
  }
  // The concept-graph viz can only size itself once its page is visible, so
  // (re)initialise it when the Knowledge Graph tab opens.
  if (tabName === "knowledge-graph") {
    const initConceptGraph = () => {
      if (typeof window.deltaInitConceptGraph === "function") {
        requestAnimationFrame(() => window.deltaInitConceptGraph());
      }
    };
    if (typeof window.deltaInitConceptGraph === "function") initConceptGraph();
    else window.addEventListener("load", initConceptGraph, { once: true });
  }
  /* 🔴 BOTH PAGES, because one /diagnostic/status payload feeds both. The
     placement card, its results and the start button are on #page-placement;
     the AREA BARS the same payload writes are on the Learner Home, on the idle
     surface a learner opens every day. Refreshing on only one of them leaves
     the other reading whatever the last visit left behind — and when this call
     said "diagnostic" against a page name that did not exist, every entry took
     the `leave` branch and the Home rendered "Loading placement status…" with
     an empty area list, forever. Caught in the browser, not by a check:
     nothing throws. */
  if (window.DiagnosticPage) {
    if (tabName === "practice" || tabName === "placement") window.DiagnosticPage.refresh();
    else window.DiagnosticPage.leave(tabName);
  }
  /* The Groups roster is read on ARRIVAL and at no other time. Somebody
     joining is the only thing that changes it, and a page that polled would
     be a request per open tab per interval for a list that changes once a
     week. The group is kept in memory, so the next arrival paints before its
     read comes back.

     🔴 LEAVING IS NOT A NO-OP ANY MORE. Since the member rows grew a live
     three-state checklist, that tab holds a ProseMirror editor with a
     half-second save debounce. `suspend()` tears it down, and the teardown is
     what FLUSHES the debounce — without it, typing a line and immediately
     clicking another tab loses the line, silently, because the page it was
     typed on is gone by the time the timer fires. */
  if (tabName === "groups") window.DDGroups?.refresh();
  else window.DDGroups?.suspend?.();
};

tabs.forEach((t) => {
  t.addEventListener("click", () => switchTab(t.dataset.tab));
});

// In-page buttons that jump to a tab (e.g. the How It Works → Knowledge Graph CTA).
document.querySelectorAll("[data-goto-tab]").forEach((b) => {
  b.addEventListener("click", () => switchTab(b.dataset.gotoTab));
});

/* THE PLACEMENT PAGE'S WAY OUT — bound here rather than through
   [data-goto-tab], because the generic binding cannot say "and I mean it".

   diagnostic-page.js keeps this button off the screen while a placement is
   ACTIVE (there is no practice stream beside a live test — the backend serves
   probes), so the redirect above should never see this click. The explicit
   route is kept anyway: it is three lines, and it is what guarantees the one
   button on the page that promises an exit can never be inert, whatever a
   future status rule decides to show it in.

   🔴 AND THE PROBE CLOCK STOPS ON THE WAY OUT. `#placement-timer` hangs off the
   topbar notch, which is on every page: left running, it would count a probe
   the learner is no longer looking at down to 00:00 and expire it against a
   practice screen. The test itself stays ACTIVE and resumable — the account
   menu row reads "Resume the placement test" — which is what "skip for now"
   promises. */
document.getElementById("placement-skip-btn")?.addEventListener("click", async () => {
  /* 🔴 CONFIRM THE STATE BEFORE HONOURING THE SKIP. This button is hidden the
     moment a status says the placement is ACTIVE — but it is visible before the
     first status lands (the markup ships it visible, and `render(null)` keeps it
     so a signed-out visitor can still leave). Clicked inside that window on a
     learner who IS mid-test, `leavingPlacement` walks straight past the redirect
     below; the status then arrives, `moveWorkspace(true)` parks
     `.practice-container` inside the now-hidden placement page, and Practice is
     left with an empty workspace and no way to reach the probe. Found by codex,
     2026-09-01.

     `refresh()` renders before it resolves, so `isRunning()` is authoritative
     here — and the same render has already replaced this button with "Load next
     placement question", which is the honest control for that state. An
     unreachable backend does NOT set it, so an outage still lets the learner
     out rather than trapping them on a page whose status never loads. */
  await window.DiagnosticPage?.refresh?.();
  if (window.DiagnosticPage?.isRunning?.() === true) return;
  window.PlacementTimer?.stop?.();
  switchTab("practice", { leavingPlacement: true });
});

// Tabs a guest is allowed to see/use (the learning surface). Account-only
// tabs (Account, Split Tool) stay hidden until login.
const guestVisibleTabs = ["knowledge-graph", "courses", "practice", "targeted-practice"];

// Second visibility pass, run AFTER the auth pass below and never before it.
// It only ever ADDS .hidden, so it can't reveal a tab auth just hid — which
// is what keeps the two passes composable in either mode.
const applyModeVisibility = () => {
  const advanced = isAdvancedMode();
  document.body.classList.toggle("dd-basic-mode", !advanced);
  // Tabs: ADD-only. Turning advanced back on must NOT un-hide these here —
  // the auth pass above already restored the ones this viewer is allowed,
  // and a symmetric toggle would hand a guest the tabs auth just took away.
  if (!advanced) {
    // `.tab` alone. Each tab used to have a sibling `.tab-info` ⓘ carrying the
    // same data-tab, and this selector took out the pair; the dots were
    // deleted on 2026-08-23 (index.html).
    document.querySelectorAll(".tab").forEach((el) => {
      if (isAdvancedOnlyTab(el.dataset.tab)) el.classList.add("hidden");
    });
  }
  // In-page jumps to a tab that isn't there (How It Works → Knowledge Graph).
  // No auth pass touches these, so this one has to be symmetric or the CTA
  // stays hidden forever after one trip through basic mode. Hide the wrapper
  // where there is one so no empty block is left behind.
  document.querySelectorAll("[data-goto-tab]").forEach((b) => {
    if (!isAdvancedOnlyTab(b.dataset.gotoTab)) return;
    (b.closest(".hiw-graph-cta") || b).classList.toggle("hidden", !advanced);
  });
};

const updateTabVisibility = () => {
  if (authToken) {
    authOnlyTabs.forEach((t) => t.classList.remove("hidden"));
    guestOnlyTabs.forEach((t) => t.classList.add("hidden"));
  } else {
    authOnlyTabs.forEach((t) =>
      t.classList.toggle("hidden", !guestVisibleTabs.includes(t.dataset.tab))
    );
    guestOnlyTabs.forEach((t) => t.classList.remove("hidden"));
  }
  const guestBanner = document.getElementById("guest-banner");
  // isSignedIn, not authToken: a guest HAS a token (guest-session.js) and
  // the banner is the only thing telling them their progress lives in this
  // browser and how to make it follow them elsewhere.
  if (guestBanner) guestBanner.classList.toggle("hidden", isSignedIn());
  applyModeVisibility();
  updateAuthIndicators();
};

// Logged-in indicators the tester asked for: he couldn't tell he was signed
// in, and couldn't find his token. Called from updateTabVisibility (login,
// logout, init).
//
// The topbar half of that is GONE (Seth, 2026-08-23): #topbar-auth carried the
// signed-in email across the top right of every screen and the tabs live on
// that side now. "Who am I" is answered on the Account tab, which leads with
// the address and the status — see #account-identity-email below, which is the
// same fact from the same `signedIn` check.
const accountTokenInput = document.getElementById("account-dd-token");
const accountTokenCopy = document.getElementById("account-dd-token-copy");
// The Account tab now leads with plain identity (email + signed-in status);
// credentials live in the collapsed Advanced block.
const accountIdentityEmail = document.getElementById("account-identity-email");
const accountIdentityStatus = document.getElementById("account-identity-status");
const updateAuthIndicators = () => {
  // Every readout here answers "who am I", so every one of them is keyed on
  // isSignedIn(). A guest holds a token but is not signed in as anyone, and
  // showing them their generated guest-<hex>@… address as their email would
  // be worse than saying nothing.
  const signedIn = isSignedIn();
  if (accountIdentityEmail) {
    accountIdentityEmail.textContent = signedIn ? (authEmail || "—") : "Not signed in";
  }
  if (accountIdentityStatus) {
    accountIdentityStatus.textContent = signedIn
      ? "Signed in"
      : "Guest — progress is saved in this browser";
  }
  // Nothing to log out OF as a guest: the button would clear a session that
  // guest-session.js immediately re-establishes from the same credentials,
  // which reads as a broken button.
  if (accountLogout) accountLogout.hidden = !signedIn;
  if (accountTokenInput) {
    accountTokenInput.value = signedIn ? authToken : "";
    accountTokenInput.placeholder = signedIn ? "" : "Sign in to see your token";
  }
};
if (accountTokenCopy && accountTokenInput) {
  accountTokenCopy.addEventListener("click", () => {
    const text = accountTokenInput.value || "";
    if (!text) return;
    const done = () => {
      accountTokenCopy.textContent = "✓ Copied";
      setTimeout(() => { accountTokenCopy.textContent = "📋 Copy"; }, 1500);
    };
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text).then(done).catch(() => {});
    } else {
      accountTokenInput.select();
      try { document.execCommand("copy"); done(); } catch (_) { /* ignore */ }
    }
  });
}

const setAuthState = (token, email) => {
  const wasAuthed = !!authToken;
  authToken = token || "";
  authEmail = email || "";
  if (authToken) {
    localStorage.setItem("auth_token", authToken);
    localStorage.setItem("auth_email", authEmail);
    // This token belongs to a person now. GUEST_CREDENTIALS_KEY is left
    // alone on purpose: logging out returns them to the same guest account
    // and the progress it already had, instead of a blank third one.
    localStorage.removeItem(GUEST_ACTIVE_KEY);
    authStatus.textContent = authEmail ? `Logged in as ${authEmail}` : "Logged in";
    // Guest → logged in: practice mode is detected once at init, so reload to
    // re-init in backend mode and hydrate server-side progress cleanly.
    if (!wasAuthed) {
      window.location.reload();
      return;
    }
    switchTab("practice");
  } else {
    localStorage.removeItem("auth_token");
    localStorage.removeItem("auth_email");
    authStatus.textContent = "";
    // No forced login page — drop back to the guest practice surface.
    switchTab("practice");
  }
  updateTabVisibility();
  window.dispatchEvent(
    new CustomEvent("delta:auth-state-changed", {
      detail: { token: authToken, email: authEmail },
    })
  );
};

// Signing out drops back to the GUEST session, not to a dead-end. That is a
// different backend account, and practiceMode + the whole practice surface
// are wired up once at init, so it takes a reload rather than an in-place
// swap. setAuthState has already cleared the token by then, so the reload
// boots token-less and guest-session.js logs the guest back in.
const signOutAndReload = async () => {
  // Awaited, and reloaded only afterwards: supabaseSignOut() is async, and
  // navigating out from under an in-flight sign-out can cancel it — after
  // which maybeRefreshSupabaseAuth() finds the session still alive on the
  // next load and signs the user back in, which reads as a Log out button
  // that does nothing. Failure still reloads: a provider that will not
  // answer must not be able to trap someone in a session.
  if (typeof supabaseSignOut === "function") {
    try {
      await supabaseSignOut();
    } catch (err) {
      console.warn("Supabase sign-out failed; signing out locally anyway:", err);
    }
  }
  setAuthState("", "");
  window.location.reload();
};
logoutButton.addEventListener("click", signOutAndReload);
accountLogout.addEventListener("click", signOutAndReload);

const maybeRefreshSupabaseAuth = async () => {
  if (isLocalHost || typeof supabaseGetSession !== "function") return false;
  try {
    const session = await supabaseGetSession();
    const refreshedToken = session?.access_token || "";
    const refreshedEmail = session?.user?.email || authEmail || "";
    if (!refreshedToken) {
      // No Supabase session is the NORMAL state for Google/backend sign-ins
      // (they never create one). Signing the user out here turned every
      // stray 401 into a full logout. Let the caller's 401 handling decide
      // (handleExpiredToken shows the banner); just report "not refreshed".
      return false;
    }
    if (refreshedToken === authToken) return false;
    setAuthState(refreshedToken, refreshedEmail);
    return true;
  } catch (err) {
    console.warn("Supabase session refresh failed:", err);
    return false;
  }
};

const apiFetch = async (path, options = {}, allowSessionRefresh = true) => {
  const headers = options.headers ? { ...options.headers } : {};
  // Remember which token this request actually carried. Several calls are in
  // flight whenever a token dies, and the first one back refreshes it — the
  // stragglers then arrive holding a 401 for a token that has ALREADY been
  // replaced. Those need a retry, not a second login.
  const sentWith = authToken;
  // ...and WHOSE it was. A token that changed underneath a request is only
  // safe to replay if it still belongs to the same account: signing in as
  // someone else also changes `authToken`, and replaying the original options
  // — a graded submit's POST body included — would write one person's answer
  // into another person's account.
  const sentAs = authEmail;
  if (authToken) {
    headers.Authorization = `Bearer ${authToken}`;
  }
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (response.status === 401 && allowSessionRefresh) {
    const refreshed = await maybeRefreshSupabaseAuth();
    if (refreshed && authToken) {
      return apiFetch(path, options, false);
    }
    /* A GUEST token that expired is recoverable without anyone noticing: the
       account is fine and this browser still holds its password. Mint a new
       token and retry the one request that failed.

       This has to happen HERE, at the fetch, rather than at each call site.
       The 401 that matters most is the graded submit — the backend records
       the attempt and the old path reloaded the page over the top of the
       result, so the learner saw Submit do nothing. Retrying in place is the
       difference between a hiccup and losing the answer they just wrote. */
    if (authToken && authToken !== sentWith && authEmail === sentAs) {
      return apiFetch(path, options, false);
    }
    const guested = await global_DDGuest()?.refreshExpiredSession?.();
    if (guested && authToken) {
      return apiFetch(path, options, false);
    }
  }
  return response;
};

/* 🔴 PUBLISHED ON PURPOSE — do not delete this line as redundant.

   `apiFetch` is a top-level `const` in a classic script, so it is NOT a window
   property (the same trap this tree documents for `switchTab`, `PracticeAPI`
   and `PracticeSession`). Two readers guard on exactly that and fall back:

     concept-graph/kc_lattice_read.js:  window.apiFetch || window.fetch
     concept-graph/lesson-graph.js:     window.apiFetch || fetch

   The fallback is bare `fetch` with a RELATIVE path, which never reaches the
   backend — it hits whatever is serving the page. Locally that is a 404 in the
   console; on Vercel the SPA rewrite answers `/api/practice/kc-lattice` with
   200 text/html, `res.ok` is true, `res.json()` throws, the catch writes null,
   and both surfaces render the guest/offline reading for a signed-in learner
   with no error anywhere. Found 2026-08-24 by counting boot 404s. Same shape as
   the `.vercelignore` runtime-fetch trap: check the content type, not the code.

   Assigning here rather than converting the const to a `function` declaration
   keeps the token-refresh recursion above (`apiFetch(path, options, false)`)
   bound to this exact implementation. */
window.apiFetch = apiFetch;

/* 🔴 PUBLISHED FOR THE SAME REASON, and only these two facts.

   `isSignedIn` and `authEmail` are top-level bindings in this classic script,
   so groups/groups_store.js cannot see them — and it needs both: whether to
   make the call at all (a guest's progress lives in this browser, so every
   /api/practice/groups/* call would be a 401 per boot) and what to call the
   person in a roster before they have said otherwise.

   A FUNCTION for the email, not the value. `authEmail` is reassigned by
   setAuthState on every sign-in and sign-out; a snapshot taken here would be
   the address the page booted with, which for anyone who signed in without a
   reload is the empty string. */
window.DDIdentity = {
  isSignedIn: () => isSignedIn(),
  email: () => authEmail,
};

// Read through `window` rather than the binding: guest-session.js is loaded
// AFTER this file, so `DDGuest` does not exist yet when apiFetch is defined —
// only when it is called.
const global_DDGuest = () => window.DDGuest;

// --- Sign in with Google (Google Identity Services) ---

// Decode a JWT payload (no verification — display only; the backend verifies
// the signature). Used to show "Logged in as <email>" after Google sign-in.
const decodeJwtPayload = (jwtStr) => {
  try {
    const part = jwtStr.split(".")[1];
    const json = atob(part.replace(/-/g, "+").replace(/_/g, "/"));
    return JSON.parse(decodeURIComponent(escape(json)));
  } catch {
    return {};
  }
};

const googleSignInMessage = () => document.getElementById("google-signin-message");

// Exchange the Google credential for our app JWT, then sign in.
const handleGoogleCredential = async (response) => {
  const msg = googleSignInMessage();
  const credential = response && response.credential;
  if (!credential) {
    if (msg) msg.textContent = "Google sign-in was cancelled.";
    return;
  }
  if (msg) msg.textContent = "Signing in…";
  try {
    const res = await fetch(`${API_BASE}/auth/google`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ credential }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      if (msg) msg.textContent = data.detail || "Google sign-in failed.";
      return;
    }
    const email = decodeJwtPayload(credential).email || "";
    if (msg) msg.textContent = "Signed in!";
    setAuthState(data.access_token, email); // reloads into backend mode
  } catch (e) {
    if (msg) msg.textContent = e.message || "Google sign-in failed.";
  }
};

// Render the Google button once both the GIS library and a client id are ready.
let _googleSignInInited = false;
const initGoogleSignIn = () => {
  if (_googleSignInInited) return;
  const clientId = window.GOOGLE_CLIENT_ID || "";
  const buttonEl = document.getElementById("google-signin-banner");
  const noteEl = document.getElementById("google-signin-note");
  if (!buttonEl) return;

  if (!clientId) {
    if (noteEl) noteEl.textContent = "Google sign-in isn't configured yet.";
    return;
  }
  if (!(window.google && google.accounts && google.accounts.id)) {
    return; // GIS script not loaded yet; the poller will retry
  }

  try {
    google.accounts.id.initialize({
      client_id: clientId,
      callback: handleGoogleCredential,
      auto_select: false,
      cancel_on_tap_outside: true,
    });
    google.accounts.id.renderButton(buttonEl, {
      theme: "filled_blue",
      size: "large",
      text: "continue_with",
      shape: "pill",
      logo_alignment: "left",
    });
    if (noteEl) noteEl.textContent = "";
    _googleSignInInited = true;
  } catch (e) {
    if (noteEl) noteEl.textContent = "Could not load Google sign-in. Please refresh and try again.";
  }
};

// GIS loads async (`async defer`), so poll briefly until it's ready.
(() => {
  let tries = 0;
  const tick = () => {
    initGoogleSignIn();
    if (_googleSignInInited || tries++ > 40) return; // ~10s max
    setTimeout(tick, 250);
  };
  tick();
})();

// --- Account settings ---

// Advanced-mode toggle. Applies on change with no Save step — it rewrites the
// nav the user is looking at, and a nav that only updates after a form submit
// reads as broken. If the tab they're on is one of the ones that just went
// away, carry them to Practice rather than leaving a page with no tab.
const advancedModeToggle = document.getElementById("account-advanced-mode");
if (advancedModeToggle) {
  advancedModeToggle.checked = isAdvancedMode();
  advancedModeToggle.addEventListener("change", () => {
    localStorage.setItem(ADVANCED_MODE_KEY, advancedModeToggle.checked ? "1" : "0");
    const activeTab = document.querySelector(".tab.active")?.dataset.tab || "";
    updateTabVisibility();
    if (!advancedModeToggle.checked && isAdvancedOnlyTab(activeTab)) switchTab("practice");
  });
}

accountForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const apiBase = document.getElementById("account-api-base").value.trim();
  const openaiKey = document.getElementById("account-openai-key").value.trim();
  const mathpixId = document.getElementById("account-mathpix-id").value.trim();
  const mathpixKey = document.getElementById("account-mathpix-key").value.trim();
  const githubUsername = document.getElementById("account-github-username").value.trim();

  if (apiBase) {
    localStorage.setItem("api_base", apiBase);
    API_BASE = apiBase;
  } else {
    localStorage.removeItem("api_base");
    API_BASE = defaultApiBase;
  }
  localStorage.setItem("account_openai_key", openaiKey);
  localStorage.setItem("account_mathpix_id", mathpixId);
  localStorage.setItem("account_mathpix_key", mathpixKey);
  if (githubUsername) localStorage.setItem("account_github_username", githubUsername);
  else localStorage.removeItem("account_github_username");

  if (typeof saveUserSettingsToSupabase === "function" && authEmail) {
    const result = await saveUserSettingsToSupabase(authEmail, openaiKey, mathpixId, mathpixKey);
    if (result?.ok) {
      accountMessage.textContent = "Saved to your account.";
    } else {
      accountMessage.textContent = `Error saving to account: ${result?.error || "unknown"}`;
    }
  } else {
    accountMessage.textContent = "Saved to browser only (not logged in to Supabase).";
  }
  setTimeout(() => (accountMessage.textContent = ""), 2000);
});

// Load saved account settings into form
const savedApiBase = localStorage.getItem("api_base") || "";
const savedOpenai = localStorage.getItem("account_openai_key") || "";
const savedMathpixId = localStorage.getItem("account_mathpix_id") || "";
const savedMathpixKey = localStorage.getItem("account_mathpix_key") || "";
const savedGithubUsername = localStorage.getItem("account_github_username") || "";
document.getElementById("account-api-base").value = savedApiBase;
document.getElementById("account-openai-key").value = savedOpenai;
document.getElementById("account-mathpix-id").value = savedMathpixId;
document.getElementById("account-mathpix-key").value = savedMathpixKey;
document.getElementById("account-github-username").value = savedGithubUsername;

// --- Initial state ---

if (authToken) {
  authStatus.textContent = authEmail ? `Logged in as ${authEmail}` : "Logged in";
}
// A pathname deep link wins over the normal auth-aware landing page. Read it
// before switching, then confirm the optimistic pre-paint class only after the
// requested page is visible so no other page flashes first.
const soloTab = window.DDSoloRoute?.read?.() || "";
/* A reload the LEARNER did not ask for should put them back where they were.
   The only one left is guest-session.js's last-resort recovery, and landing a
   learner mid-placement on the practice setup card reads as the app having
   thrown their work away. Read once and clear: this is for the reload that
   just happened, never for the next one. */
/* Read once and clear, for BOTH one-shot landing keys. Every one of them is
   about the reload that just happened and never about the next one, so a key
   that is read must also be consumed — including on a load that ends up
   preferring a different one, or a stale value lands the session on that page
   the next time anything reloads. */
const takeSessionTab = (key) => {
  try {
    const tab = sessionStorage.getItem(key) || "";
    if (tab) sessionStorage.removeItem(key);
    return tab;
  } catch (_) {
    return "";
  }
};
const recoveredTab = takeSessionTab(RECOVERED_TAB_KEY);
/* A brand-new test user has never seen this app. See FIRST_RUN_TAB_KEY. It
   sits BELOW recoveredTab because a recovery reload is the app taking the page
   away from somebody mid-work, and putting them back is the more urgent of the
   two — the two cannot both be pending in practice (a recovery only ever fires
   for a guest, and a test user is not one). */
const firstRunTab = takeSessionTab(FIRST_RUN_TAB_KEY);
// First visit lands on the FORK, every visit after it lands on the work.
// `authToken` is the right test for that and isSignedIn() is not: a returning
// GUEST has a token (guest-session.js stored one last time) and wants their
// practice queue, not a choice they already made. A first-time visitor has
// nothing stored yet, which is the one moment the fork is the right screen.
//
// "welcome" replaced "why-this-app" on 2026-08-23: landing straight on the
// argument for the app assumed the visitor wanted to read one. #page-welcome
// offers that path and the other one, says which is optional, and shows
// nothing else.
/* A group invite in the address bar is somebody who just clicked a link a
   friend sent them, and every other landing rule would drop them on the
   Learner Home with the token still in the URL and nothing saying what it
   was for. It sits under `soloTab` because a pathname deep link is an
   explicit request for one chromeless page, and above the rest because
   nothing else here was asked for by the person arriving.

   Reading it does NOT consume it: groups/groups_store.js clears the
   parameter only once the join has actually happened, so a reload before
   then still lands here. */
const invitedToGroup = window.DDGroupStore?.inviteFromLocation?.() ? "groups" : "";
switchTab(soloTab || invitedToGroup || recoveredTab || firstRunTab || (authToken ? "practice" : "welcome"));
updateTabVisibility();
window.DDSoloRoute?.apply?.();
// Auth is the Continue-with-Google button rendered into the guest banner by
// initGoogleSignIn() above — no login/signup pages or CTA buttons to wire.
