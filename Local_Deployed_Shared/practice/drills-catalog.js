/* ================================================================
   PROCEDURAL DRILLS CATALOG

   Pure metadata for the 13 procedural drill notebooks shipped under
   arena-procedural-drills/. Surfaced via the existing Targeted
   Practice search + the existing ArenaUnlock card GUI — this file
   is data only, no UI of its own.

   Each entry:
     - id          stable catalog key
     - title       what the student sees in search + on the card
     - sub         second-line subtitle in the search row
     - notebookPath repo-relative path inside AkiraTheSquid/Delta-Drills.
                    stats/predicted-links.js#colabUpstreamHref detects
                    the arena-procedural-drills/ prefix and routes to
                    the student's fork of Delta-Drills (account page
                    "GitHub username" field).
     - subtopics   adaptive-state subtopic keys the drill targets.
                   ArenaUnlock.postArenaRating reads ex.subtopics here
                   to POST /api/practice/arena-rating with the right
                   subtopic so the new EWMA pipeline bumps. Format
                   matches the backend's egress: bare "Topic: Subtopic".
     - targetSeconds wall-clock budget for the arena-unlock timer.
   ================================================================ */

(function () {
  // Default unlock floor: when the student's EWMA accuracy on the targeted
  // subtopic crosses this percent, the drill becomes eligible for auto-
  // surface via ArenaUnlock.tryShow. Drill BUILDS the subtopic too, so this
  // is "you've practiced enough text-bank questions on this subtopic, now
  // apply it in a real notebook" — same pattern as ARENA legacy unlocks,
  // just sourced from the new EWMA algo instead of legacy hard prereqs.
  const DEFAULT_UNLOCK_MIN_PCT = 50;

  const D = (folder, atom, subtopic, opts = {}) => ({
    id: `drill:${atom}`,
    title: atom,
    sub: `Procedural drill · ${subtopic}`,
    notebookPath: `arena-procedural-drills/${folder}/${atom}.ipynb`,
    subtopics: [subtopic],
    targetSeconds: opts.targetSeconds ?? 600,
    unlockMinPct: opts.unlockMinPct ?? DEFAULT_UNLOCK_MIN_PCT,
    isDrill: true,
  });

  window.DRILLS_CATALOG = [
    D("prereqs_einops", "einops-rearrange",            "Einops: Rearrange"),
    D("prereqs_einops", "einops-reduce",               "Einops: Reduce"),
    D("prereqs_einops", "einops-repeat",               "Einops: Repeat"),
    D("prereqs_einops", "einops-einsum",               "Einops: Deep Learning"),
    D("prereqs_numpy",  "tensor-zeros-init",           "Numpy: Core array literacy"),
    D("prereqs_numpy",  "tensor-item-scalar",          "Numpy: Core array literacy"),
    D("prereqs_numpy",  "broadcasting-rules",          "Numpy: Vectorization and broadcasting"),
    D("prereqs_numpy",  "boolean-mask-identity-replace","Numpy: Indexing and selection"),
    D("prereqs_numpy",  "tensor-unbind",               "Numpy: Indexing and selection"),
    D("prereqs_numpy",  "vector-normalisation",        "Numpy: Applied patterns and advanced"),
    D("prereqs_numpy",  "softmax-from-logits",         "Numpy: Applied patterns and advanced"),
    D("prereqs_numpy",  "rotation-matrix-3d-y-axis",   "Numpy: Applied patterns and advanced"),
    D("prereqs_numpy",  "as-strided-noncontig-source", "Numpy: Applied patterns and advanced"),
  ];

  // ── Drill auto-surface logic (new-algo, no legacy ARENA_PREREQS_TEMP) ──
  // Mirrors the ARENA legacy "shown once" tracking so the student sees each
  // drill recommendation exactly once. Reads EWMA via the same score lookup
  // ARENA uses (window.getArenaPrereqSubtopicScore), which already reads
  // from both backend cache (window.__arenaSubtopicsCache) and Pyodide
  // adaptive state.

  const _DRILL_SHOWN_LS_KEY = "drills_shown";
  const _DRILL_SHOWN_SCHEMA_VERSION = "v1-2026-05-24";
  const _DRILL_SHOWN_VERSION_KEY = "drills_shown_schema";
  try {
    if (localStorage.getItem(_DRILL_SHOWN_VERSION_KEY) !== _DRILL_SHOWN_SCHEMA_VERSION) {
      localStorage.removeItem(_DRILL_SHOWN_LS_KEY);
      localStorage.setItem(_DRILL_SHOWN_VERSION_KEY, _DRILL_SHOWN_SCHEMA_VERSION);
    }
  } catch (_) { /* localStorage unavailable — fine */ }

  const _readShownSet = () => {
    try {
      const raw = localStorage.getItem(_DRILL_SHOWN_LS_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return new Set(Array.isArray(arr) ? arr : []);
    } catch (_) { return new Set(); }
  };
  const _writeShownSet = (set) => {
    try { localStorage.setItem(_DRILL_SHOWN_LS_KEY, JSON.stringify([...set])); } catch (_) {}
  };

  // Subtopic score lookup — bare subtopic strings ("Einops: Rearrange") match
  // the format the backend egresses on /api/practice/subtopics.
  const _scoreFor = (bareSubtopic) => {
    if (typeof window.getArenaPrereqSubtopicScore !== "function") return null;
    // The fn signature is (topic, subtopic) but it also accepts the full
    // composed key as the second arg (it tries both raw and composed).
    return window.getArenaPrereqSubtopicScore(null, bareSubtopic);
  };

  const _isDrillUnlocked = (drill) => {
    const subs = Array.isArray(drill.subtopics) ? drill.subtopics : [];
    if (!subs.length) return false;
    // Every drill subtopic must clear unlockMinPct. (All current drills have
    // exactly one targeted subtopic, but the AND-logic matches ARENA.)
    return subs.every((s) => {
      const sc = _scoreFor(s);
      return sc != null && sc >= (drill.unlockMinPct ?? DEFAULT_UNLOCK_MIN_PCT);
    });
  };

  window.getNextUnshownUnlockedDrill = () => {
    const shown = _readShownSet();
    for (const d of window.DRILLS_CATALOG) {
      if (shown.has(d.id)) continue;
      if (_isDrillUnlocked(d)) return d;
    }
    return null;
  };

  window.markDrillShown = (drillId) => {
    if (!drillId) return;
    const shown = _readShownSet();
    shown.add(drillId);
    _writeShownSet(shown);
  };

  // Diagnostic — `window.debugDrillUnlock()` mirrors window.debugArenaUnlock.
  window.debugDrillUnlock = () => {
    const shown = _readShownSet();
    const rows = (window.DRILLS_CATALOG || []).map((d) => {
      const subs = d.subtopics || [];
      const checks = subs.map((s) => {
        const sc = _scoreFor(s);
        return { subtopic: s, need: d.unlockMinPct, have: sc == null ? "null" : sc.toFixed(1), met: sc != null && sc >= d.unlockMinPct };
      });
      return {
        atom: d.title,
        shown: shown.has(d.id),
        unlocked: checks.every((c) => c.met),
        blocking: checks.filter((c) => !c.met).map((c) => `${c.subtopic}(${c.have}<${c.need})`).join(", ") || "—",
      };
    });
    console.group("[Drills] unlock snapshot");
    console.table(rows);
    console.groupEnd();
    return rows;
  };
})();
