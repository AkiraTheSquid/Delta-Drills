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
   renderQuestion (practice/ui.js), which every other door goes through. */
(() => {
  const picker = document.getElementById("practice-target");
  const note = document.getElementById("practice-target-note");
  if (!picker || !note) return;
  const RAY = "raytracing-0.1";
  let generation = 0;
  let current = null;
  let scope = null; // Set of in-focus KC ids, null = whole curriculum
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

  /* Out of focus = TAGGED to a concept outside it. An untagged drill is left
     to the server (it refuses those under a focus anyway); refusing it here
     too could bounce a lane the q-matrix does not cover back and forth. Fails
     open until the q-matrix loads: the server is still the real filter. */
  const outOfScope = (q) => {
    if (!scope || !qmatrix || !q) return false;
    const kcs = qmatrix.get(Number(q.question_id ?? q.id ?? q.questionId)) || [];
    return kcs.length > 0 && kcs.some((kc) => !scope.has(kc));
  };

  /* For renderQuestion: true = this drill is out of focus and the server's
     next pick is being painted in its place. Whatever the server hands back
     is painted as-is (`replacing` skips the check), so an out-of-focus answer
     from the server can never loop; a failed fetch paints the original
     rather than leaving the prompt blank. */
  const redirects = (q, count) => {
    if (replacing || !outOfScope(q)) return false;
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

  const enforce = async () => {
    if (!scope || practiceMode !== "backend") return;
    await loadQmatrix();
    if (!scope || !qmatrix) return;
    // Script-global consts, not window properties (practice/timer.js, api.js).
    const session = typeof PracticeSession !== "undefined" ? PracticeSession : null;
    const saved = window.SessionSnapshot?.read?.();
    if (session?.hasPausedSession() && saved && outOfScope({ id: saved.questionId })) {
      session.discard();
      sessionSummary.textContent =
        "Your paused session was on a concept outside Ray Tracing 0.1, so it was set aside. Start a new block to practise 0.1.";
    }
    const onScreen = typeof PracticeAPI !== "undefined" ? PracticeAPI.currentQuestion : null;
    const graded = !practiceFeedbackArea?.classList.contains("hidden");
    if (graded || session?.hasPausedSession()) return;
    redirects(onScreen, practiceQuestionCount);
  };

  const paint = (data) => {
    current = data;
    picker.value = data.target;
    const focused = data.target === RAY;
    const ids = new Set(data.kcs || []);
    scope = focused ? ids : null;
    const cy = window.deltaConceptGraphCy?.();
    cy?.nodes().forEach((n) => {
      if (focused && !ids.has(n.id())) n.style("display", "none");
      else n.removeStyle("display");
    });
    cy?.edges().forEach((e) => {
      if (focused && (!ids.has(e.source().id()) || !ids.has(e.target().id()))) e.style("display", "none");
      else e.removeStyle("display");
    });
    /* `placement_ready` is a STORED ray placement's reading (the test itself
       was retired 2026-09-26); said only when there is one. */
    const placed = data.placement_ready?.length || 0;
    note.textContent = focused
      ? `0.1 + ${ids.size} concepts including prerequisites.${placed ? ` ${placed} ready from an earlier placement.` : ""}`
      : "Practice across the curriculum.";
  };
  const request = async (target) => {
    const res = await apiFetch("/api/practice/practice-target", target === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ target }),
    });
    if (!res.ok) throw new Error("Could not load practice focus. Check your connection and sign-in.");
    return res.json();
  };
  const refresh = async () => {
    const mine = ++generation;
    try {
      const data = await request();
      if (mine !== generation) return;
      paint(data);
      picker.disabled = false;
    } catch (err) {
      if (mine === generation) { picker.disabled = true; note.textContent = err.message; }
      return;
    }
    enforce().catch((err) => console.warn("[practice-target] focus check failed:", err));
  };
  picker.addEventListener("change", async () => {
    const mine = ++generation;
    picker.disabled = true;
    try {
      const data = await request(picker.value);
      if (mine !== generation) return;
      paint(data);
      await window.deltaRefreshKcLattice?.();
      window.dispatchEvent(new CustomEvent("delta:adaptive-state-changed"));
    } catch (err) {
      if (current) picker.value = current.target;
      note.textContent = err.message;
      return;
    } finally { picker.disabled = false; }
    enforce().catch((err) => console.warn("[practice-target] focus check failed:", err));
  });
  window.PracticeTarget = { outOfScope, redirects };
  loadQmatrix();
  window.addEventListener("delta:practice-mode-ready", refresh);
  window.addEventListener("delta:practice-target-graph-ready", () => { if (current) paint(current); });
  if (window.DDPracticeModeReady) refresh();
})();
