/* ================================================================
   GUEST-SESSION.JS — the app, with no sign-in

   The learning surface is not evenly available to a signed-out visitor.
   The diagnostic (practice/api.js diagnosticStart/Status/Answer), the
   lessons (practice/lessons.js), the stage ladder (practice/ladder.js)
   and the BKT student model behind them are all guarded by
   `practiceMode === "backend"`, and practiceMode is "local" for anyone
   without a token. So "guest" used to mean a static drill pool with no
   placement, no lessons and no mastery — i.e. not the app this page
   spends its landing tab describing.

   Rather than fork every one of those into a second client-side
   implementation that would drift from the server's, a first-time
   visitor is given an ACCOUNT: this file mints one against the backend
   (POST /auth/signup) and keeps its credentials in this browser's
   localStorage. The visitor is never asked for anything, and their
   progress is this browser's — clear site data and it is gone, which is
   what the guest banner says.

   THINGS THIS MAKES TRUE
     1. `authToken` no longer means "a person signed in". It means "this
        session can call the backend". Identity questions — the guest
        banner, the topbar email, which tab we land on — go through
        app.js's isSignedIn(), never through authToken.
     2. The credentials are never deleted, not even by Log out. Signing
        in with Google and then signing out returns you to the SAME guest
        account, with the progress it had.
     3. Failure is not fatal. If the backend cannot be reached, nothing
        is set, practiceMode stays "local", and the app behaves exactly
        as it did before this file existed.
     4. This is not a security boundary. Signup is already open (anyone
        can create an account with an email and a password); this only
        skips the form.

   Loaded straight after app.js, whose top-level `let authToken` /
   `let authEmail` / `let API_BASE` bindings this file writes to. Classic
   scripts share one global lexical scope, so those are the real
   bindings, not copies — `window.authToken = …` would create a shadow
   property that nothing reads.
   ================================================================ */

(function installGuestSession(global) {
  // A domain that exists only to make the address well-formed for the
  // backend's EmailStr validation. Nothing is ever sent to it.
  const GUEST_EMAIL_DOMAIN = "guest.delta-drills.app";
  // Backend requires >= 8 chars (app/schemas.py UserCreate).
  const PASSWORD_BYTES = 24;
  // Set once per tab while a guest whose token expired is being re-logged-in,
  // so a backend that 401s everything cannot reload forever. It is NEVER
  // cleared by a successful login: /auth/login handing back a token does not
  // prove the token works on the protected endpoints that 401'd, and clearing
  // it there would let a backend that mints tokens it then rejects bounce the
  // page for as long as the tab is open. One recovery per tab is the budget;
  // sessionStorage forgets it when the tab does.
  const RECOVERY_FLAG = "dd_guest_recovering";

  const randomToken = (bytes) => {
    const buf = new Uint8Array(bytes);
    (global.crypto || global.msCrypto).getRandomValues(buf);
    return Array.from(buf, (b) => b.toString(16).padStart(2, "0")).join("");
  };

  const readCredentials = () => {
    try {
      const raw = localStorage.getItem(GUEST_CREDENTIALS_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed || !parsed.email || !parsed.password) return null;
      return parsed;
    } catch (_) {
      return null;
    }
  };

  const writeCredentials = (credentials) => {
    localStorage.setItem(GUEST_CREDENTIALS_KEY, JSON.stringify(credentials));
  };

  const mintCredentials = () => ({
    email: `guest-${randomToken(8)}@${GUEST_EMAIL_DOMAIN}`,
    password: randomToken(PASSWORD_BYTES),
  });

  // POST to an auth endpoint. Reports the token AND why there isn't one,
  // because the caller has to tell two failures apart that look identical
  // from here: credentials the backend REFUSED (401 — the account is gone,
  // mint a new one) and a backend that could not answer (429, 5xx, offline
  // — say nothing, keep the credentials, try again next load). Collapsing
  // both to null is how one bad minute costs a learner their progress
  // permanently: the login fails, a fresh account is minted over the top of
  // the only copy of their credentials, and the old one is unreachable
  // forever after.
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
      console.warn("[guest] %s failed:", path, err);
      return { token: null, status: 0 }; // 0 = never reached the backend
    }
  };

  // Adopt a guest token WITHOUT going through app.js setAuthState: that
  // function reloads the page on a guest -> signed-in transition, which
  // is exactly the transition this is, and reloading here would loop.
  const adopt = (token, email) => {
    authToken = token;
    authEmail = email;
    localStorage.setItem("auth_token", token);
    localStorage.setItem("auth_email", email);
    localStorage.setItem(GUEST_ACTIVE_KEY, "1");
    // The nav was painted for a token-less visitor a moment ago.
    if (typeof updateTabVisibility === "function") updateTabVisibility();
    global.dispatchEvent(
      new CustomEvent("delta:auth-state-changed", { detail: { token, email, guest: true } })
    );
  };

  let pending = null;
  /* One silent re-login at a time. A page does not make one API call when a
     token dies — the practice surface makes several at once, and without this
     every one of them would start its own /auth/login. They all wait on the
     same promise instead, and all of them see the same answer. */
  let refreshing = null;
  /* A failed silent refresh is not retried on a loop. The backend is either
     down or refusing these credentials, and hammering it once per API call
     for as long as the tab is open turns one bad minute into a flood. */
  let refreshFailedAt = 0;
  const REFRESH_RETRY_MS = 30000;

  const provision = async () => {
    // Already holding a token — a returning guest, or a real sign-in.
    if (authToken) return true;

    const existing = readCredentials();
    if (existing) {
      const login = await postAuth("/auth/login", existing);
      if (login.token) {
        adopt(login.token, existing.email);
        return true;
      }
      // 401 is the backend saying these credentials are not an account —
      // a reset database, a different backend. Nothing is recoverable, so
      // mint a new guest rather than stranding the visitor in local mode
      // forever. ANY OTHER answer is the backend having a bad moment, and
      // the one thing that must not happen then is overwriting the only
      // copy of credentials that still work. Fall back to local mode for
      // this load and try again on the next one.
      if (login.status !== 401) {
        console.warn(
          "[guest] guest login could not be completed (status %s); keeping the " +
          "stored credentials and staying in local mode for this load",
          login.status
        );
        return false;
      }
      console.warn("[guest] stored guest credentials were rejected; minting a new guest");
    }

    const fresh = mintCredentials();
    const signup = await postAuth("/auth/signup", fresh);
    if (!signup.token) return false;
    // Written only after signup succeeded, so a failed attempt does not
    // leave behind credentials that will never log in.
    writeCredentials(fresh);
    adopt(signup.token, fresh.email);
    return true;
  };

  /* Log this guest back in WITHOUT reloading the page.

     The reload in recoverExpiredSession() below is correct at the moment a
     page loads holding a dead token, and destructive at any other moment: the
     first call to 401 is often a graded submit, and reloading throws away the
     result the learner just earned (the backend has recorded it — they simply
     never see it) and lands them on whatever tab a fresh load picks. From
     their seat, Submit did nothing.

     We still hold the password, so the honest fix is to use it: mint a new
     token in place, hand it to the app, and let the caller retry the one
     request that failed. Nothing is lost and nothing moves on screen. */
  const refreshSilently = async () => {
    if (typeof isGuestSession !== "function" || !isGuestSession()) return false;
    const credentials = readCredentials();
    if (!credentials) return false;
    if (refreshFailedAt && Date.now() - refreshFailedAt < REFRESH_RETRY_MS) return false;
    if (refreshing) return refreshing;
    refreshing = (async () => {
      /* Never REJECT. The caller is apiFetch, in the middle of handling a 401
         it is expected to hand back to the call site — a throw here would
         replace that 401 with an exception the call site has no branch for,
         and the last-resort reload path would never be reached. Every failure
         is `false`, and every failure starts the cooldown. */
      const startedWith = authToken;
      try {
        const login = await postAuth("/auth/login", credentials);
        /* A sign-in can land WHILE this login is in flight — setAuthState does
           not reload on guest -> person (the guest already had a token, so
           `wasAuthed` is true), so nothing stops adopt() from putting the guest
           token back over the person's one and flipping GUEST_ACTIVE_KEY on.
           From the learner's seat they signed in and were silently a guest
           again. If the identity moved, this refresh is stale: drop it. */
        if (!isGuestSession() || authToken !== startedWith) return false;
        if (!login.token) {
          // 401 here means the account itself is gone (a reset database),
          // which adopt() cannot fix — leave it to the reload path, which
          // re-runs ensure() and will mint a fresh guest.
          refreshFailedAt = Date.now();
          return false;
        }
        adopt(login.token, credentials.email);
        refreshFailedAt = 0;
        return true;
      } catch (err) {
        console.warn("[guest] silent re-login failed:", err);
        refreshFailedAt = Date.now();
        return false;
      }
    })();
    try {
      return await refreshing;
    } finally {
      refreshing = null;
    }
  };

  global.DDGuest = {
    /**
     * Make sure this browser has a backend session, creating a guest
     * account if it has to. Safe to await more than once; the work
     * happens on the first call. Resolves false when the backend could
     * not be reached, which leaves the app in local mode.
     */
    ensure() {
      if (!pending) pending = provision();
      return pending;
    },

    /**
     * Re-login this guest in place and keep the page exactly as it is.
     * Resolves true when a fresh token is live, in which case the caller
     * should retry the request that 401'd. Never reloads, never touches a
     * real signed-in session.
     */
    refreshExpiredSession() {
      return refreshSilently();
    },

    /**
     * Is a 401 on this browser recoverable, rather than proof of being signed
     * out? True while this is a guest session whose password is still here —
     * refreshExpiredSession() failing on such a browser means the backend is
     * down, refusing, or in its cooldown, NOT that there is nobody signed in.
     * A surface that renders a "sign in" prompt off a 401 has to ask this
     * first, or one bad minute tells a mid-placement learner to sign in.
     */
    canRecover() {
      if (typeof isGuestSession !== "function" || !isGuestSession()) return false;
      return !!readCredentials();
    },

    /**
     * A 401 on a GUEST token that refreshExpiredSession() could not fix:
     * the account itself is unreachable. Clear the dead token and reload so
     * ensure() logs back in or mints a new guest. Returns true if a reload
     * was started, in which case the caller must stop.
     *
     * This is the LAST resort, not the first: a reload discards whatever the
     * learner was looking at. Try refreshExpiredSession() first.
     *
     * Deliberately does nothing for a REAL signed-in user — silently
     * turning them into a guest would hide the fact that their sign-in
     * lapsed. They keep practice/mode.js's expired-session notice.
     */
    recoverExpiredSession() {
      if (typeof isGuestSession !== "function" || !isGuestSession()) return false;
      if (!readCredentials()) return false;
      if (sessionStorage.getItem(RECOVERY_FLAG) === "1") return false;
      sessionStorage.setItem(RECOVERY_FLAG, "1");
      // Come back to the tab the learner was on, not to whatever a fresh load
      // would pick. app.js reads this once and clears it.
      try {
        const tab = document.querySelector(".tab.active")?.dataset.tab || "";
        if (tab) sessionStorage.setItem(RECOVERED_TAB_KEY, tab);
      } catch (_) {}
      localStorage.removeItem("auth_token");
      localStorage.removeItem("auth_email");
      localStorage.removeItem(GUEST_ACTIVE_KEY);
      global.location.reload();
      return true;
    },
  };
})(window);
