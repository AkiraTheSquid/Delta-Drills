/* Account-scoped curriculum focus. No localStorage: shared browsers must not
   inherit another learner's target. The server filters the actual queue.

   🔴 THE SERVER FILTER ONLY COVERS WHAT THE SERVER SERVES. Seth, 2026-09-28:
   focus set to Ray Tracing 0.1, and q849 (`einops.einsum`) came up anyway. The
   picker was clean — replaying his prod state served in-scope drills every
   time — and the log showed q849 painted one second after an in-scope
   /next-question with no request of its own. It was a question the CLIENT put
   back: a paused session, or the drill already on screen when the focus
   changed, both saved from before the focus existed. So the focus is checked
   here too, against the same q-matrix the server reads
   (lessons/qmatrix_tags.json ↔ kc_graph._QMATRIX_PATH): a paused session on
   an out-of-focus drill is discarded, and an out-of-focus drill on screen is
   swapped for the server's next pick. `redirects(q)` is the same check for
   renderQuestion (practice/ui.js), which every other door goes through.

   🔴 SAME LEAK, COURSE TOGGLE. Seth, 2026-09-29: switched LeetCode off on the
   Courses tab and the LeetCode drill (q60008) stayed up. The server stops
   serving a course that is off (course_registry.course_off, via
   kc_prefs.is_disabled), but nothing asked it for a new drill. So the
   course choice is checked here too: `studied: false` from
   /course-shares is exactly the server's course_off, and a drill tagged to a
   concept of such a course is out, the same way an out-of-focus one is.

   🔴 THE FOCUS PICKER IS GONE (Seth, 2026-09-29: "for practice focus, move it
   in to … the filter … where you check off multiple areas"). Choosing what to
   practise is now the graph toolbar's per-area switches (kg-toolbar.js), which
   write kc-prefs `enabled: false` — the same off the server already honours —
   and a concept switched off that way is out for this file too (`kcWhyOut`),
   so a drill already on screen goes when its area does.
   An account that still holds the Ray Tracing 0.1 focus is NOT converted: the
   focus also scopes ARENA's milestones (course_mix.milestones → Home's "%
   complete") and a stored ray placement's exposure (practice_targets), which
   area switches do not reproduce. The Filter shows it as its own row with a
   Turn off button (`focus()` / `clearFocus()` below) until the learner does. */
(() => {
  const RAY = "raytracing-0.1";
  let generation = 0;
  let current = null;
  let scope = null; // Set of in-focus KC ids, null = whole curriculum
  let offCourses = new Set(); // course ids with `studied: false` (course-shares)
  let courseLoads = 0;
  let qmatrix = null; // qid → target_kcs, null until loaded
  let qmatrixLoad = null;
  let replacing = false;

  const loadQmatrix = () => {
    qmatrixLoad ??= fetch("lessons/qmatrix_tags.json", { cache: "force-cache" })
      .then((res) => (res.ok ? res.json() : null))
      .then((rows) => {
        if (!rows) throw new Error("q-matrix unavailable");
        qmatrix = new Map();
        for (const [qid, tags] of Object.entries(rows)) qmatrix.set(Number(qid), tags?.target_kcs || []);
      })
      // Not cached as settled: the next enforce() tries again.
      .catch(() => { qmatrixLoad = null; });
    return qmatrixLoad;
  };

  const loadCourses = async (mine) => {
    const res = await apiFetch("/api/practice/course-shares");
    if (!res.ok) throw new Error(`course-shares ${res.status}`);
    const data = await res.json();
    if (mine !== courseLoads) return;
    offCourses = new Set((data.courses || []).filter((r) => r.studied === false).map((r) => r.course));
  };

  // course_registry.course_of: the standalone courses by prefix, else ARENA
  // (the main graph). Every delta-drills KC starts "deltadrills.".
  const courseOf = (kc) =>
    kc.startsWith("leetcode.") ? "leetcode" : kc.startsWith("deltadrills.") ? "delta-drills" : "arena";

  /* Why a drill is out, or null. Out = TAGGED to a concept of a course that is
     off, or outside the 0.1 focus. An untagged drill is left to the server (it
     refuses those under a focus anyway); refusing it here too could bounce a
     lane the q-matrix does not cover back and forth. Fails open until the
     q-matrix loads: the server is still the real filter. */
  const userOff = (kc) => {
    const row = window.getKcLattice?.()?.kcs?.[kc];
    return !!(row && row.pref && row.pref.enabled === false);
  };
  const anyUserOff = () => {
    const kcs = window.getKcLattice?.()?.kcs;
    return !!kcs && Object.values(kcs).some((row) => row && row.pref && row.pref.enabled === false);
  };
  const kcWhyOut = (kc) => {
    if (offCourses.has(courseOf(kc))) return "from a course you switched off";
    if (userOff(kc)) return "on a concept you switched off";
    if (scope && !scope.has(kc)) return "on a concept outside Ray Tracing 0.1";
    return null;
  };
  const filtering = () => !!(scope || offCourses.size || anyUserOff());
  const whyOut = (q) => {
    if (!filtering() || !qmatrix || !q) return null;
    const kcs = qmatrix.get(Number(q.question_id ?? q.id ?? q.questionId)) || [];
    return kcs.map(kcWhyOut).find(Boolean) || null;
  };
  const outOfScope = (q) => whyOut(q) !== null;

  /* A concept the learner asked to practise by name (Practice ⤢ on the graph,
     a ?lesson= link, an exercise's practice block — practice/kc-practice.js)
     pins the queue to it, and its drills are the learner's own pick: they are
     painted even when that concept is out. What a LATER filter change does to
     that pin is enforce()'s business, not renderQuestion's. */
  const pinned = () => window.__kcFocusId || null;

  /* For renderQuestion: true = this drill is out of focus and the server's
     next pick is being painted in its place. Whatever the server hands back
     is painted as-is (`replacing` skips the check), so an out-of-focus answer
     from the server can never loop; a failed fetch paints the original
     rather than leaving the prompt blank. `force` = a filter just changed,
     which outranks a pin (see enforce). */
  const redirects = (q, count, { force = false } = {}) => {
    if (replacing || (pinned() && !force) || !outOfScope(q)) return false;
    replacing = true;
    PracticeAPI.getNextQuestion()
      .then((nextQ) => renderQuestion(nextQ || q, count))
      .catch((err) => {
        console.warn("[practice-target] could not replace the out-of-focus question:", err);
        renderQuestion(q, count);
      })
      .finally(() => { replacing = false; });
    return true;
  };

  /* `changed` = the learner just changed the focus or a course. The newer
     choice wins over a concept pinned before it (the LeetCode case above came
     off a LeetCode concept's Practice ⤢), so an out pin is let go. On a page
     load nothing changed: a pin made by the link that opened the page stands. */
  const enforce = async ({ changed = false } = {}) => {
    if (!filtering() || practiceMode !== "backend") return;
    await loadQmatrix();
    if (!qmatrix) return;
    // Script-global consts, not window properties (practice/timer.js, api.js).
    const session = typeof PracticeSession !== "undefined" ? PracticeSession : null;
    const saved = window.SessionSnapshot?.read?.();
    const why = saved ? whyOut({ id: saved.questionId }) : null;
    // A paused block the learner started on a concept by name (its ladder is
    // in the snapshot) is that pick: kept across a page load, dropped only
    // when a filter changes after it.
    const picked = !!(saved?.ladder || saved?.config?.ladder);
    if (session?.hasPausedSession() && why && (changed || !picked)) {
      session.discard();
      sessionSummary.textContent =
        `Your paused session was ${why}, so it was set aside. Start a new block to keep going.`;
    }
    const onScreen = typeof PracticeAPI !== "undefined" ? PracticeAPI.currentQuestion : null;
    if (changed && pinned() && kcWhyOut(pinned())) window.KcPractice?.release?.();
    const graded = !practiceFeedbackArea?.classList.contains("hidden");
    if (graded || session?.hasPausedSession()) return;
    redirects(onScreen, practiceQuestionCount, { force: changed });
  };

  const request = async (target) => {
    const res = await apiFetch("/api/practice/practice-target", target === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ target }),
    });
    if (!res.ok) throw new Error(`practice-target ${res.status}`);
    return res.json();
  };
  const refresh = async () => {
    const mine = ++generation;
    try {
      const data = await request();
      if (mine !== generation) return;
      current = data;
      scope = data.target === RAY ? new Set(data.kcs || []) : null;
      window.dispatchEvent(new CustomEvent("delta:practice-focus-changed", { detail: { target: data.target } }));
    } catch (err) {
      console.warn("[practice-target] could not read the practice focus:", err);
    }
    recheck();
  };
  // The Filter's "Turn off" on the old focus row.
  const clearFocus = async () => {
    const mine = ++generation;
    const data = await request("all");
    if (mine !== generation) return false;
    current = data;
    scope = null;
    await window.deltaRefreshKcLattice?.();
    window.dispatchEvent(new CustomEvent("delta:adaptive-state-changed"));
    window.dispatchEvent(new CustomEvent("delta:practice-focus-changed", { detail: { target: data.target } }));
    return true;
  };
  /* The course choice changed (courses.js toggle, course-pick.js), or the
     page just learned who the learner is: re-read which courses are off, then
     drop whatever drill that leaves out. */
  const recheck = async (changed = false) => {
    const mine = ++courseLoads;
    try { await loadCourses(mine); } catch (err) { console.warn("[practice-target] could not read courses:", err); }
    // A newer recheck is in flight: only its (fresher) course list may act.
    if (mine !== courseLoads) return;
    enforce({ changed }).catch((err) => console.warn("[practice-target] focus check failed:", err));
  };
  window.PracticeTarget = {
    outOfScope, redirects, clearFocus,
    // {target, kcs} as the server last said, or null before it has.
    focus: () => (current ? { target: current.target, kcs: current.kcs || [], label: current.target === RAY ? "Ray Tracing 0.1" : null } : null),
  };
  loadQmatrix();
  window.addEventListener("delta:practice-mode-ready", refresh);
  window.addEventListener("delta:courses-changed", () => recheck(true));
  // An area or a concept switched off on the graph (kg-toolbar.js Filter,
  // kg-panel.js cog): the drill on screen may be one of them.
  window.addEventListener("delta:kc-prefs-changed", (e) => {
    if (e.detail && e.detail.enabled === false) enforce({ changed: true }).catch((err) => console.warn("[practice-target] focus check failed:", err));
  });
  if (window.DDPracticeModeReady) refresh();
})();
