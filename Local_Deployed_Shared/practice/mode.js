/* ================================================================
   PRACTICE MODE — auth + routing
   ================================================================ */

/**
 * Show (or update) a prominent notice at the top of the practice panel.
 * Used for mode demotions the learner MUST see — a console.warn is not a
 * notice. (Tester spent a session rating questions in the static demo pool
 * without knowing his sign-in had expired.)
 */
function showPracticeModeNotice(message) {
  let el = document.getElementById("practice-mode-notice");
  if (!el) {
    const anchor = document.getElementById("practice-mode-intro");
    if (!anchor || !anchor.parentNode) return;
    el = document.createElement("div");
    el.id = "practice-mode-notice";
    el.className = "practice-mode-notice";
    anchor.parentNode.insertBefore(el, anchor);
  }
  el.textContent = message;
  el.classList.remove("hidden");
}

function hidePracticeModeNotice() {
  const el = document.getElementById("practice-mode-notice");
  if (el) el.classList.add("hidden");
}

/**
 * Call when a backend API returns 401. Clears the stale token and
 * switches this session to local mode so the page stays usable.
 */
function handleExpiredToken() {
  console.warn("[practice] Token expired or invalid — falling back to local mode.");
  showPracticeModeNotice(
    "Your sign-in expired, so progress is no longer being saved to your account. " +
    "Sign in again to get back to your adaptive queue.",
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
  // Back on the real adaptive queue — clear any stale demotion notice.
  if (practiceMode === "backend") hidePracticeModeNotice();
  console.log("[practice] mode:", practiceMode);
}
