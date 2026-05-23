/* ================================================================
   STATS PREDICTED — TEMP HARDCODED PREREQ SCAFFOLD (DEPRECATED-ON-LAND)

   *** TEMPORARY FILE — DELETE WHEN THE CONCEPT GRAPH SHIPS ***

   Purpose: get the predicted-scores frontend pipeline running end-to-
   end with a tiny slice of ARENA (just chapter 0.0 Prerequisites)
   before the real concept-graph backend is wired up. This file:

     1. Restricts the predicted-scores table to chapter 0.0 only.
     2. Provides the 26 0.0 exercises (the upstream registry currently
        has [] for this notebook).
     3. Hardcodes per-exercise prerequisites: a list of
        {topic, subtopic, minPct} entries against Delta Drills topics
        (Numpy / Einops / Einsum and their subtopics).
     4. Exposes a score lookup that reads the same adaptive state
        `arena/manifest.js#computeArenaReadiness` reads (baseline per
        subtopic, 0–100).

   Once the real concept graph lands (another agent is building it
   alongside this scaffold), DELETE this file and remove:
     - The `<script src="stats/predicted-prereqs-temp.js">` tag in index.html
     - The `ARENA_PREREQS_TEMP_*` reads in stats/predicted.js
     - The 'predicted-prereqs-temp.js' entry in stats/watch.py
   ================================================================ */

window.ARENA_PREREQS_TEMP_ENABLED = true;
window.ARENA_PREREQS_TEMP_RESTRICT_CHAPTER = "chapter0_fundamentals";
window.ARENA_PREREQS_TEMP_RESTRICT_PROBLEM_ID = "arena-0.0-prereqs";
window.ARENA_PREREQS_TEMP_NOTEBOOK_PATH =
  "content/ARENA_5.0-main/chapter0_fundamentals/exercises/part0_prereqs/0.0_Prerequisites_exercises.ipynb";

// Anchor convention matches jupyter-book/myst slugify: lowercase, non-alnum → '-',
// strip leading/trailing hyphens. Used by the Read ↗ deep-link pill.
const _slugifyAnchor = (text) =>
  String(text)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

// 26 exercises shipped with ARENA 0.0 Prerequisites (25 Einops + 1 Einsum).
// Same shape `arena/exercises.js` produces: {title, anchor, targetSeconds}.
// targetSeconds is the rough wall-clock budget the student should aim for —
// the unlock-card timer compares elapsed vs. this to auto-rate the attempt.
// Numbers are rough first-pass guesses; the real concept graph will tune them.
const _EXERCISE_DEFS = [
  ["(1) Column-stacking", 180],
  ["(2) Column-stacking and copying", 180],
  ["(3) Row-stacking and double-copying", 240],
  ["(4) Stretching", 180],
  ["(5) Split channels", 240],
  ["(6) Stack into rows & cols", 240],
  ["(7) Transpose", 120],
  ["(8) Shrinking", 240],
  ["(A1) rearrange", 240],
  ["(A2) rearrange", 240],
  ["(B1) temperature average", 240],
  ["(B2) temperature difference", 300],
  ["(C1) normalize a matrix", 300],
  ["(C2) pairwise cosine similarity", 420],
  ["(D) sample distribution", 360],
  ["(E) classifier accuracy", 240],
  ["(F1) total price indexing", 300],
  ["(F3) total price gather", 360],
  ["(G) indexing", 300],
  ["(H1) batched logsumexp", 360],
  ["(H2) batched softmax", 240],
  ["(H3) batched logsoftmax", 240],
  ["(H4) batched cross entropy loss", 420],
  ["(I1) collect rows", 300],
  ["(I2) collect columns", 300],
  ["einsum: trace, mv, mm, inner, outer", 420],
];

window.ARENA_PREREQS_TEMP_EXERCISES = _EXERCISE_DEFS.map(([title, targetSeconds]) => ({
  title,
  anchor: _slugifyAnchor(title),
  targetSeconds,
}));

// Baseline prereqs every 0.0 exercise inherits — floors before any ARENA
// work makes sense. Tagged "always" so the renderer concatenates them with
// each exercise's specific prereqs.
//
// TEMP-SCAFFOLD thresholds are intentionally LOW (20–30%) so the unlock
// interstitial actually fires during early practice for pipeline testing.
// The real concept graph (replacing this file) will choose realistic gates.
const _ALWAYS_PREREQS = [
  { topic: "Numpy", subtopic: "Core array literacy", minPct: 30 },
  { topic: "Numpy", subtopic: "Vectorization and broadcasting", minPct: 25 },
];

// Per-exercise additional prereqs, keyed by exact exercise title.
// Buckets chosen by inspecting the canonical solution per exercise:
//   - rearrange-only      → Einops/Rearrange
//   - repeat (copy/stretch) → Einops/Repeat
//   - reduce (shrink/agg) → Einops/Reduce
//   - indexing/gather     → Numpy/Indexing and selection
//   - softmax/logsumexp   → Einops/Reduce + Numpy/Vectorization
//   - einsum              → Einsum/Core array literacy
const _SPECIFIC_PREREQS = {
  "(1) Column-stacking":              [{ topic: "Einops", subtopic: "Rearrange", minPct: 20 }],
  "(2) Column-stacking and copying":  [{ topic: "Einops", subtopic: "Repeat",    minPct: 20 }],
  "(3) Row-stacking and double-copying": [{ topic: "Einops", subtopic: "Repeat", minPct: 22 }],
  "(4) Stretching":                   [{ topic: "Einops", subtopic: "Repeat",    minPct: 22 }],
  "(5) Split channels":               [{ topic: "Einops", subtopic: "Rearrange", minPct: 22 }],
  "(6) Stack into rows & cols":       [{ topic: "Einops", subtopic: "Rearrange", minPct: 25 }],
  "(7) Transpose":                    [{ topic: "Einops", subtopic: "Rearrange", minPct: 20 }],
  "(8) Shrinking":                    [{ topic: "Einops", subtopic: "Reduce",    minPct: 25 }],
  "(A1) rearrange":                   [{ topic: "Einops", subtopic: "Rearrange", minPct: 25 }],
  "(A2) rearrange":                   [{ topic: "Einops", subtopic: "Rearrange", minPct: 28 }],
  "(B1) temperature average":         [{ topic: "Einops", subtopic: "Reduce",    minPct: 25 }],
  "(B2) temperature difference":      [
    { topic: "Einops", subtopic: "Reduce", minPct: 25 },
    { topic: "Einops", subtopic: "Repeat", minPct: 20 },
  ],
  "(C1) normalize a matrix":          [{ topic: "Einops", subtopic: "Reduce",    minPct: 28 }],
  "(C2) pairwise cosine similarity":  [
    { topic: "Einops", subtopic: "Reduce", minPct: 28 },
    { topic: "Numpy",  subtopic: "Vectorization and broadcasting", minPct: 30 },
  ],
  "(D) sample distribution":          [{ topic: "Numpy", subtopic: "Vectorization and broadcasting", minPct: 28 }],
  "(E) classifier accuracy":          [{ topic: "Numpy", subtopic: "Vectorization and broadcasting", minPct: 28 }],
  "(F1) total price indexing":        [{ topic: "Numpy", subtopic: "Indexing and selection",         minPct: 25 }],
  "(F3) total price gather":          [{ topic: "Numpy", subtopic: "Indexing and selection",         minPct: 28 }],
  "(G) indexing":                     [{ topic: "Numpy", subtopic: "Indexing and selection",         minPct: 28 }],
  "(H1) batched logsumexp":           [{ topic: "Einops", subtopic: "Reduce", minPct: 30 }],
  "(H2) batched softmax":             [{ topic: "Einops", subtopic: "Reduce", minPct: 30 }],
  "(H3) batched logsoftmax":          [{ topic: "Einops", subtopic: "Reduce", minPct: 30 }],
  "(H4) batched cross entropy loss":  [
    { topic: "Einops", subtopic: "Reduce", minPct: 30 },
    { topic: "Numpy",  subtopic: "Indexing and selection", minPct: 28 },
  ],
  "(I1) collect rows":                [{ topic: "Numpy", subtopic: "Indexing and selection", minPct: 30 }],
  "(I2) collect columns":             [{ topic: "Numpy", subtopic: "Indexing and selection", minPct: 30 }],
  "einsum: trace, mv, mm, inner, outer": [
    { topic: "Einsum", subtopic: "Core array literacy", minPct: 25 },
    { topic: "Einsum", subtopic: "Vectorization and broadcasting", minPct: 22 },
  ],
};

window.ARENA_PREREQS_TEMP_BY_EXERCISE = Object.fromEntries(
  window.ARENA_PREREQS_TEMP_EXERCISES.map((ex) => [
    ex.title,
    [..._ALWAYS_PREREQS, ...(_SPECIFIC_PREREQS[ex.title] || [])],
  ])
);

// ── Score lookup ──────────────────────────────────────────────
// Returns a 0–100 score for the (topic, subtopic) pair, or null if no
// state yet. Prefers `p * 100` (the EWMA accuracy 0–1, same signal the
// practice-page bars show) so the user's perceived "high score" matches
// what unlocks gates. Falls back to `baseline` when `p` is unavailable.
//
// Two state sources, checked in order:
//   1. window.__arenaSubtopicsCache — populated by ArenaUnlock.refreshScores
//      from the backend /api/practice/subtopics endpoint (BACKEND MODE,
//      keyed by full "Topic: Subtopic" name).
//   2. adaptiveStateJson — local pyodide adaptive engine (LOCAL MODE,
//      keyed by full "Topic: Subtopic" subtopic name).
//
// Both sources key subtopics as "<Topic>: <bareSubtopic>" so we compose
// the key from (topic, subtopic) args. We also try a bare-subtopic
// fallback for the unlikely case state was written without the topic prefix.
const _composeSubtopicKey = (topic, subtopic) => {
  const t = String(topic || "").trim();
  const s = String(subtopic || "").trim();
  // If subtopic was already passed as "Topic: X", don't double-prefix.
  if (s.startsWith(`${t}:`)) return s;
  return t ? `${t}: ${s}` : s;
};

const _readScoreFromSubState = (subState) => {
  if (!subState) return null;
  const p = Number(subState.p);
  if (Number.isFinite(p) && p > 0) return p * 100;
  const baseline = Number(subState.baseline);
  if (Number.isFinite(baseline) && baseline > 0) return baseline;
  // p === 0 with no baseline means "no answers yet" — return null so the
  // unlock gate treats it as "not measured" rather than "scored zero".
  if (Number.isFinite(p)) return p * 100;
  return null;
};

window.getArenaPrereqSubtopicScore = (topic, subtopic) => {
  const fullKey = _composeSubtopicKey(topic, subtopic);
  const bareKey = String(subtopic || "").trim();

  // 1. Backend cache (backend mode)
  const cache = window.__arenaSubtopicsCache;
  if (cache && typeof cache === "object") {
    const fromFull = cache[fullKey];
    if (fromFull) {
      const sc = _readScoreFromSubState(fromFull);
      if (sc != null) return sc;
    }
    const fromBare = cache[bareKey];
    if (fromBare) {
      const sc = _readScoreFromSubState(fromBare);
      if (sc != null) return sc;
    }
  }

  // 2. Local pyodide adaptive state (local mode)
  try {
    if (typeof adaptiveStateJson === "string" && adaptiveStateJson) {
      const state = JSON.parse(adaptiveStateJson);
      const subStates = state?.subtopic_states || {};
      const subState = subStates[fullKey] || subStates[bareKey];
      const sc = _readScoreFromSubState(subState);
      if (sc != null) return sc;
    }
  } catch (_) { /* ignore */ }

  return null;
};

// Diagnostic helper — call window.debugArenaUnlock() in the console to
// see exactly which prereqs are met vs. blocking each exercise. Helps
// answer "why isn't the interstitial firing for me?" without hunting
// through JSON state by hand.
window.debugArenaUnlock = () => {
  const exercises = window.ARENA_PREREQS_TEMP_EXERCISES || [];
  const map = window.ARENA_PREREQS_TEMP_BY_EXERCISE || {};
  console.group("[ArenaUnlock] prereq snapshot");
  const rows = exercises.map((ex) => {
    const prereqs = map[ex.title] || [];
    const checks = prereqs.map((p) => {
      const sc = window.getArenaPrereqSubtopicScore(p.topic, p.subtopic);
      return {
        gate: `${p.topic}/${p.subtopic}`,
        need: p.minPct,
        have: sc == null ? "null" : sc.toFixed(1),
        met: sc != null && sc >= p.minPct,
      };
    });
    return {
      title: ex.title,
      unlocked: checks.every((c) => c.met),
      checks,
    };
  });
  console.table(rows.map((r) => ({
    title: r.title,
    unlocked: r.unlocked,
    blocking: r.checks.filter((c) => !c.met).map((c) => `${c.gate}(${c.have}<${c.need})`).join(", ") || "—",
  })));
  console.log("shown list:", localStorage.getItem("arena_prereqs_temp_shown"));
  console.log("schema version:", localStorage.getItem("arena_prereqs_temp_shown_schema"));
  console.groupEnd();
  return { unlockedCount: rows.filter((r) => r.unlocked).length, totalCount: rows.length };
};

// ── Unlock logic + "already shown" tracking ───────────────────
// AND-logic across every prereq for an exercise. ALL listed
// {topic, subtopic, minPct} entries must be met. If even one subtopic
// has no recorded score yet (null), the exercise is NOT unlocked —
// otherwise a fresh student with no scores would pass the gate.
window.isArenaExerciseUnlocked = (exTitle) => {
  if (!window.ARENA_PREREQS_TEMP_ENABLED) return false;
  const prereqs = window.ARENA_PREREQS_TEMP_BY_EXERCISE &&
                  window.ARENA_PREREQS_TEMP_BY_EXERCISE[exTitle];
  if (!Array.isArray(prereqs) || !prereqs.length) return false;
  return prereqs.every((p) => {
    const score = (typeof window.getArenaPrereqSubtopicScore === "function")
      ? window.getArenaPrereqSubtopicScore(p.topic, p.subtopic)
      : null;
    return score != null && score >= p.minPct;
  });
};

const _ARENA_SHOWN_LS_KEY = "arena_prereqs_temp_shown";

// Bump SHOWN_SCHEMA_VERSION whenever thresholds drop, exercises change,
// or you want every student's interstitials to re-fire. On load, if the
// stored version doesn't match, the shown-set is wiped — this defeats
// dev-test pollution and lets a threshold change re-surface unlocks
// the student had already dismissed under the old gates.
const _ARENA_SHOWN_SCHEMA_VERSION = "v4-backend-subtopics-cache-2026-05-19";
const _ARENA_SHOWN_VERSION_KEY = "arena_prereqs_temp_shown_schema";
try {
  if (localStorage.getItem(_ARENA_SHOWN_VERSION_KEY) !== _ARENA_SHOWN_SCHEMA_VERSION) {
    localStorage.removeItem(_ARENA_SHOWN_LS_KEY);
    localStorage.setItem(_ARENA_SHOWN_VERSION_KEY, _ARENA_SHOWN_SCHEMA_VERSION);
  }
} catch (_) { /* localStorage unavailable — fine */ }

const _readArenaShownSet = () => {
  try {
    const raw = localStorage.getItem(_ARENA_SHOWN_LS_KEY);
    if (!raw) return new Set();
    const arr = JSON.parse(raw);
    return new Set(Array.isArray(arr) ? arr : []);
  } catch (_) {
    return new Set();
  }
};

const _writeArenaShownSet = (set) => {
  try {
    localStorage.setItem(_ARENA_SHOWN_LS_KEY, JSON.stringify([...set]));
  } catch (_) { /* localStorage unavailable — silently skip */ }
};

// Returns the first {title, anchor} whose prereqs are all met AND that
// the student hasn't been shown the unlock interstitial for yet. Returns
// null if none. Caller marks it shown via markArenaExerciseShown after
// the interstitial is dismissed.
window.getNextUnshownUnlockedArenaExercise = () => {
  if (!window.ARENA_PREREQS_TEMP_ENABLED) return null;
  const list = Array.isArray(window.ARENA_PREREQS_TEMP_EXERCISES)
    ? window.ARENA_PREREQS_TEMP_EXERCISES
    : [];
  if (!list.length) return null;
  const shown = _readArenaShownSet();
  for (const ex of list) {
    if (shown.has(ex.title)) continue;
    if (window.isArenaExerciseUnlocked(ex.title)) return ex;
  }
  return null;
};

window.markArenaExerciseShown = (exTitle) => {
  const shown = _readArenaShownSet();
  shown.add(exTitle);
  _writeArenaShownSet(shown);
};

// Exposed for debugging / testing — wipes the seen-list so every
// currently-unlocked exercise pops up again on the next Submit cycle.
window.resetArenaUnlockShownTracking = () => {
  try { localStorage.removeItem(_ARENA_SHOWN_LS_KEY); } catch (_) {}
};

// Prereq list for one exercise — used by the interstitial to render
// "why you're ready" with concrete numbers. Returns []
// for unknown titles.
window.getArenaPrereqsForExercise = (exTitle) => {
  const map = window.ARENA_PREREQS_TEMP_BY_EXERCISE || {};
  return Array.isArray(map[exTitle]) ? map[exTitle] : [];
};
