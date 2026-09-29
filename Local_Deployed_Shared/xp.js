/* ================================================================
   XP.JS — the one reader of measured learning.

   `GET /api/practice/xp` (app/learning_xp.py): XP is the model's knowledge
   gained, each concept paying its own worth — more for what the course
   builds on (Seth, 2026-09-29). This file reads it, caches it per account
   so the Learner Home paints at once on load, and re-broadcasts every
   summary as `delta:xp-summary`, which practice/xp-panel.js draws from.

   🪦 THE LEVEL PILL (the topbar `Level N` chip that filled as a progress
   bar) is gone: Seth, 2026-09-29, "remove the leveling system". The home
   shows the course and its percentage instead.

   Nothing here ADDS XP. `award()` is kept because every recording path
   already calls it (practice/api.js wraps the PracticeAPI methods; targeted
   practice and the lesson page dispatch `delta:xp`), and what those calls
   mean is "the learner just entered evidence — the number may have moved":
   they schedule a re-read.
   ================================================================ */
(function () {
  "use strict";

  // v3: v2 held the old 80-XP-per-concept unit and levels.
  const CACHE_PREFIX = "dd_xp_v3_";
  const REFRESH_DEBOUNCE_MS = 900;
  const cacheKey = () => {
    let email = "";
    try {
      email = (localStorage.getItem("auth_email") || "").trim();
    } catch (_) {
      /* private mode — fall through to the guest key */
    }
    return CACHE_PREFIX + (email || "guest");
  };

  const readCache = () => {
    try {
      const raw = localStorage.getItem(cacheKey());
      const parsed = raw ? JSON.parse(raw) : null;
      return parsed && Array.isArray(parsed.days) ? parsed : null;
    } catch (_) {
      return null;
    }
  };

  const writeCache = (summary) => {
    try {
      localStorage.setItem(cacheKey(), JSON.stringify(summary));
    } catch (_) {
      /* best-effort */
    }
  };

  let summary = readCache();

  // ── the server read ────────────────────────────────────────────
  let fetchSeq = 0;
  let debounceTimer = null;

  const tzQuery = () => {
    const offset = new Date().getTimezoneOffset();
    let name = "";
    try {
      name = Intl.DateTimeFormat().resolvedOptions().timeZone || "";
    } catch (_) {
      /* older engines — the offset alone is enough */
    }
    return `tz_offset=${offset}${name ? `&tz_name=${encodeURIComponent(name)}` : ""}`;
  };

  // apiFetch is app.js's top-level const — a global lexical binding, not
  // always a window property; the same guarded read the practice modules use.
  const fetcher = () => (typeof apiFetch === "function" ? apiFetch : window.apiFetch);
  const hasSession = () => typeof authToken !== "undefined" && !!authToken;

  const accept = (next) => {
    summary = next;
    writeCache(next);
    publish(next);
  };

  // Tell the panel what was read — including "nothing" — so it never
  // straddles an account change (codex, 2026-09-26: after sign-out the
  // panel kept the previous learner's history on screen).
  const publish = (detail) => window.dispatchEvent(new CustomEvent("delta:xp-summary", { detail }));

  const refresh = async () => {
    const seq = ++fetchSeq;
    const _fetch = fetcher();
    if (!hasSession()) {
      summary = null;
      publish(null);
      return null;
    }
    if (typeof _fetch !== "function") {
      publish(null);
      return null;
    }
    try {
      const res = await _fetch(`/api/practice/xp?${tzQuery()}`);
      if (!res?.ok) throw new Error(`xp ${res?.status}`);
      const data = await res.json();
      if (seq !== fetchSeq) return null; // a newer read owns the panel
      accept(data);
      return data;
    } catch (_) {
      if (seq === fetchSeq) publish(null);
      return null;
    }
  };

  const refreshSoon = () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(refresh, REFRESH_DEBOUNCE_MS);
  };

  /** Save the learner's target; resolves to the new summary or throws the
      server's sentence. */
  const setTarget = async (body) => {
    const _fetch = fetcher();
    if (typeof _fetch !== "function") throw new Error("Not connected.");
    const res = await _fetch(`/api/practice/xp-target?${tzQuery()}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data?.detail || "The target could not be saved.");
    fetchSeq += 1; // an older in-flight read must not paint over this
    accept(data);
    return data;
  };

  window.DeltaXP = {
    /** Evidence was just entered; the measured number may have moved. */
    award: () => refreshSoon(),
    refresh,
    setTarget,
    /** `tz_offset=…&tz_name=…` for any per-day read (practice/xp-group-view.js). */
    tzQuery,
    summary: () => summary,
    reload: () => {
      summary = readCache();
      publish(summary);
      refresh();
    },
  };

  // ── wiring ─────────────────────────────────────────────────────
  window.addEventListener("delta:xp", refreshSoon);
  window.addEventListener("delta:practice-state-changed", refreshSoon);
  window.addEventListener("delta:practice-mode-ready", () => refresh());
  // Sign in, sign out, guest provisioning — all change whose XP this is.
  window.addEventListener("delta:auth-state-changed", () => window.DeltaXP.reload());

  const start = () => refresh();
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
