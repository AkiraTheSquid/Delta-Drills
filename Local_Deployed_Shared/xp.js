/* ================================================================
   XP.JS — levels, and the progress bar that IS the level pill.

   The bar is not a widget parked somewhere in the chrome: it is the
   `Level N` chip itself, colouring in from the left. `styles/xp.css` owns
   the drawing; this file owns the number, and hands the drawing ONE value:
   `--dd-xp-pct` on `.dd-level`.

   NO NUMBER IS RENDERED FOR THE PROGRESS. Not a count, not a percent. The
   only numeral on the pill is the level itself (plus the hover `title`).
   The Learner Home XP panel (practice/xp-panel.js) is where the numbers live.

   ── XP IS MEASURED LEARNING (2026-09-26) ──────────────────────────
   Until 2026-09-26 this file counted points in localStorage: 25 for a
   correct answer, 10 for a miss, 1 per 15 s of typing. Seth: XP must be
   "measured by your actual learning progress", the same for every learner.
   The number now comes from the backend, `GET /api/practice/xp`
   (app/learning_xp.py): 1 XP = one point of knowledge gained on one
   concept, where knowledge is the model's smoothed belief the concept is
   learned × its FSRS recall. A level is one concept's worth, 80 XP.

   So nothing here ADDS XP any more. `award()` is kept because every
   recording path already calls it (practice/api.js wraps the PracticeAPI
   methods; targeted practice and the lesson page dispatch `delta:xp`), and
   what those calls mean now is "the learner just entered evidence — the
   number may have moved": they schedule a re-read. The generic typing tick
   is gone: typing is not learning.

   The last summary is cached per account so the pill paints at once on
   load instead of flashing Level 1; the server's answer replaces it.
   Every fetched summary is re-broadcast as `delta:xp-summary`, which the
   Learner Home panel draws from, so the pill and the panel never make two
   requests for one answer.
   ================================================================ */
(function () {
  "use strict";

  const CACHE_PREFIX = "dd_xp_v2_";
  const REFRESH_DEBOUNCE_MS = 900;
  /* Long enough for the width transition in xp.css to finish before the
     fill is snapped back to the remainder. */
  const LEVELUP_HOLD_MS = 560;
  const LEVELUP_FLASH_MS = 1100;

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
      return parsed && Number.isFinite(parsed.level) ? parsed : null;
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

  // ── the DOM ────────────────────────────────────────────────────
  // Looked up lazily: the ?embed=1 knowledge-graph frame strips app
  // chrome, so the elements are allowed to be absent.
  let elChip = null;
  let elNum = null;
  // The second copy of the numeral, drawn on the fill — see the two-layer
  // note in styles/xp.css. Written in lockstep with elNum.
  let elNumOn = null;
  let flashTimer = null;
  let snapTimer = null;

  const grabDom = () => {
    elChip = document.getElementById("dd-level");
    elNum = document.getElementById("dd-level-num");
    elNumOn = document.getElementById("dd-level-num-on");
  };

  const paint = (pct) => {
    if (elChip) elChip.style.setProperty("--dd-xp-pct", pct + "%");
  };

  const currentPct = () => {
    if (!summary || !(summary.need > 0)) return 0;
    return Math.max(0, Math.min(100, (summary.into / summary.need) * 100));
  };

  const render = (levelledUp) => {
    const level = summary ? summary.level : 1;
    const levelText = String(level);
    if (elNum) elNum.textContent = levelText;
    if (elNumOn) elNumOn.textContent = levelText;
    if (!elChip) return;
    if (summary) {
      const into = Math.round(summary.into);
      elChip.setAttribute("aria-label", `Level ${level}, ${into} of ${summary.need} XP of learning`);
      // Hover-only. The pill itself still states nothing.
      elChip.title = `Level ${level} · ${into}/${summary.need} XP · one level = one concept learned`;
    } else {
      // No account, or a different one: nothing of the last learner stays.
      elChip.removeAttribute("aria-label");
      elChip.removeAttribute("title");
    }

    if (!levelledUp) {
      if (snapTimer === null) paint(currentPct());
      return;
    }
    /* Level-up: run the fill to the end, then snap back to the remainder
       WITHOUT a transition (animating backwards reads as losing progress)
       and let it grow again from zero. */
    paint(100);
    elChip.classList.add("dd-level--up");
    clearTimeout(flashTimer);
    flashTimer = setTimeout(() => elChip.classList.remove("dd-level--up"), LEVELUP_FLASH_MS);
    clearTimeout(snapTimer);
    snapTimer = setTimeout(() => {
      elChip.classList.add("dd-level--snap");
      paint(0);
      // Force the zero width to land before the transition comes back.
      void elChip.offsetWidth;
      elChip.classList.remove("dd-level--snap");
      paint(currentPct());
      snapTimer = null;
    }, LEVELUP_HOLD_MS);
  };

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

  const accept = (next, { animate }) => {
    const before = summary;
    summary = next;
    writeCache(next);
    // A level can also go DOWN: later answers can revise when the model
    // believes a concept was learned. That repaints plainly — no fanfare.
    render(animate && before && next.level > before.level);
    publish(next);
  };

  // Tell the panel what the pill now shows — including "nothing" — so the two
  // never straddle an account change (codex, 2026-09-26: after sign-out the
  // panel kept the previous learner's history on screen).
  const publish = (detail) => window.dispatchEvent(new CustomEvent("delta:xp-summary", { detail }));

  const refresh = async () => {
    const seq = ++fetchSeq;
    const _fetch = fetcher();
    if (!hasSession()) {
      summary = null;
      render(false);
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
      if (seq !== fetchSeq) return null; // a newer read owns the pill
      accept(data, { animate: true });
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
    accept(data, { animate: false });
    return data;
  };

  window.DeltaXP = {
    /** Evidence was just entered; the measured number may have moved. */
    award: () => refreshSoon(),
    refresh,
    setTarget,
    summary: () => summary,
    state: () => (summary ? { level: summary.level, into: summary.into, need: summary.need } : null),
    reload: () => {
      summary = readCache();
      render(false);
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

  const start = () => {
    grabDom();
    render(false);
    refresh();
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
