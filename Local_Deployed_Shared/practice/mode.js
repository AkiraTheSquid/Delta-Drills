/* ================================================================
   PRACTICE MODE — auth + routing
   ================================================================ */

/* #practice-mode-notice was DELETED on 2026-08-23 at Seth's request, together
   with #practice-mode-intro (the paragraph it anchored itself to).

   ⚠️ READ THIS BEFORE ADDING ANOTHER BANNER. The notice existed for one
   reason: a tester spent a whole session rating questions out of the static
   demo pool without knowing his sign-in had expired. The practice UI renders
   its target-difficulty and progress widgets identically in demo mode, so a
   silent demotion looks exactly like a working adaptive session. Deleting the
   banner brings that failure mode back — the demotion is now only visible in
   the console, which no learner reads.

   The right fix, if this bites again, is to make the DEMOTED STATE itself
   visible on a surface that is already on screen (the session status row),
   not to reintroduce a floating banner. Do not put this one back. */

/**
 * Call when a backend API returns 401. Clears the stale token and
 * switches this session to local mode so the page stays usable.
 */
function handleExpiredToken() {
  // A GUEST token that expired is not a lapsed sign-in — the account and its
  // progress are still there and guest-session.js still holds the password.
  // It reloads and logs back in, so there is nothing to notify and nothing to
  // demote. Returns false for a real signed-in user, who does get the notice.
  if (window.DDGuest?.recoverExpiredSession?.()) return;
  console.warn(
    "[practice] Token expired or invalid — falling back to local mode. " +
    "Progress is no longer saved to the account; sign in again for the adaptive queue.",
  );
  if (typeof setAuthState === "function") {
    setAuthState(""); // clears localStorage and resets authToken in app.js
  } else {
    localStorage.removeItem("auth_token");
    localStorage.removeItem("auth_email");
  }
  practiceMode = "local";
  // Clear any stale backend question so local mode picks a fresh one
  practiceProgress.currentQuestion = null;
}

// practiceMode is set once at init based on email + environment
let practiceMode = "local"; // default fallback

function detectPracticeMode() {
  const email = typeof authEmail === "string" ? authEmail : "";
  if (typeof getPracticeMode === "function") {
    practiceMode = getPracticeMode(email);
  } else if (typeof apiFetch === "function" && typeof authToken === "string" && !!authToken) {
    practiceMode = "backend";
  }
  console.log("[practice] mode:", practiceMode);
}
