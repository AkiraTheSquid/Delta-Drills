/* ================================================================
   PRACTICE MODE — auth + routing
   ================================================================ */

/**
 * Call when a backend API returns 401. Clears the stale token and
 * switches this session to local mode so the page stays usable.
 */
function handleExpiredToken() {
  console.warn("[practice] Token expired or invalid — falling back to local mode.");
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
