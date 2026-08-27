/* ================================================================
   FEEDBACK PANELS — the learner's way to report what is wrong

   Two surfaces, one behaviour:

   PROBLEM — "Provide feedback" sits beside Submit (and beside the rating
   buttons after a submit, because that is when most people notice). It expands
   #problem-feedback-panel at the bottom of the left column and scrolls it into
   view. Chips choose WHAT KIND of feedback this is; the note says what
   happened; Send posts once.

   LESSON — a lesson has no submit button and no question id, so it gets its
   own panel that docks to the LEFT of the reading column and its own endpoint.
   Routing lesson prose into /problem-feedback would file it against the
   question the lesson is gating AND queue an AI rewrite of that question.

   🔴 The chips do not send. They used to: one click on "Broken" posted
   immediately, so the note box under them was decorative — whatever you typed
   arrived only if you then clicked a SECOND chip. Selection and submission are
   separate here, and `Send` is the only thing that posts.
   ================================================================ */

const DDFeedbackPanel = (() => {
  const $ = (id) => document.getElementById(id);

  const _reducedMotion = () =>
    !!window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* The panel's own scrolling ancestor — .practice-left on the practice
     screen, the document under 900px where layout.css hands scrolling back to
     the page. Resolved per call: the same panel scrolls in a pane on a desktop
     and in the window on a phone. */
  const _scrollerFor = (el) => {
    let node = el.parentElement;
    while (node && node !== document.body && node !== document.documentElement) {
      const overflowY = getComputedStyle(node).overflowY;
      if (
        (overflowY === "auto" || overflowY === "scroll") &&
        node.scrollHeight > node.clientHeight + 1
      ) {
        return node;
      }
      node = node.parentElement;
    }
    return null;
  };

  /* Already readable where it is? Then there is nothing to do.

     🔴 This is not an optimisation. Without it, opening a panel that was
     already fully on screen still ran the fallback scroll and moved the whole
     DOCUMENT — which slides the notebook in the other pane up by the same
     amount, for no reason the learner asked for. A short question is the
     common case, so that was the common case.

     A panel taller than the view can never be "fully" visible; for that one,
     having its top on screen IS revealed, or the retry loop fights itself. */
  const _isRevealed = (el, pane) => {
    const box = el.getBoundingClientRect();
    const top = pane ? pane.getBoundingClientRect().top : 0;
    const bottom = pane ? pane.getBoundingClientRect().bottom : window.innerHeight;
    if (box.height > bottom - top) return box.top >= top - 1 && box.top < bottom;
    return box.top >= top - 1 && box.bottom <= bottom + 1;
  };

  /* 🔴 A panel that was `display:none` a frame ago has no height yet, and a
     scroll past the current scrollHeight is CLAMPED, not queued — the pane
     lands at its old bottom and the panel stays below the fold. Re-aim while
     it settles (same failure the solution cell hit on resume, 2026-08-27). */
  const _reveal = (panel, retries = 5) => {
    if (!panel || !panel.getClientRects().length) return;
    const pane = _scrollerFor(panel);
    if (_isRevealed(panel, pane)) return;
    const lead = 20;
    const behavior = _reducedMotion() ? "auto" : "smooth";
    if (pane) {
      const offset = panel.getBoundingClientRect().top - pane.getBoundingClientRect().top;
      pane.scrollTo({ top: Math.max(0, pane.scrollTop + offset - lead), behavior });
    } else {
      const top = window.scrollY + panel.getBoundingClientRect().top - 90;
      window.scrollTo({ top: Math.max(0, top), behavior });
    }
    if (retries > 0) {
      setTimeout(() => {
        if (!_isRevealed(panel, pane)) _reveal(panel, retries - 1);
      }, 120);
    }
  };

  /* Clearing a note has to tell DDAutoGrow. Assigning .value fires no input
     event, so a box grown to 300px stays 300px tall after it is emptied. */
  const _clearNote = (note) => {
    if (!note) return;
    note.value = "";
    note.dispatchEvent(new Event("input", { bubbles: true }));
  };

  const _setStatus = (el, text, isError) => {
    if (!el) return;
    el.textContent = text;
    el.classList.toggle("dd-fb-status-error", !!isError);
    el.classList.toggle("hidden", !text);
  };

  /* One selected chip at a time. `aria-pressed` carries it for assistive tech;
     `.flagged` is the paint, and it is the class feedback.css already knows. */
  const _bindChips = (chips, onPick) => {
    chips.forEach((chip) => {
      chip.setAttribute("aria-pressed", "false");
      chip.addEventListener("click", () => {
        chips.forEach((other) => {
          const picked = other === chip;
          other.classList.toggle("flagged", picked);
          other.setAttribute("aria-pressed", picked ? "true" : "false");
        });
        onPick(chip.dataset.flag || "");
      });
    });
  };

  const _clearChips = (chips) => {
    chips.forEach((chip) => {
      chip.classList.remove("flagged");
      chip.setAttribute("aria-pressed", "false");
    });
  };

  /* The question this panel is about, right now. Read fresh — the pool moves
     under an in-flight send. */
  const _currentQuestionId = () => {
    if (typeof PracticeAPI === "undefined") return null;
    const q = PracticeAPI.currentQuestion;
    return q ? q.question_id : null;
  };

  // ── Problem panel ──────────────────────────────────────────────
  const problem = {
    panel: null, chips: [], note: null, status: null, sendBtn: null, tag: "",
    sending: false,
  };

  const openProblem = () => {
    if (!problem.panel) return;
    problem.panel.classList.remove("hidden");
    document.querySelectorAll('.dd-fb-trigger[data-fb-target="problem"]').forEach((btn) =>
      btn.setAttribute("aria-expanded", "true"));
    requestAnimationFrame(() => {
      _reveal(problem.panel);
      if (problem.note) problem.note.focus({ preventScroll: true });
    });
  };

  const closeProblem = () => {
    if (!problem.panel) return;
    problem.panel.classList.add("hidden");
    document.querySelectorAll('.dd-fb-trigger[data-fb-target="problem"]').forEach((btn) =>
      btn.setAttribute("aria-expanded", "false"));
  };

  const toggleProblem = () => {
    if (!problem.panel) return;
    if (problem.panel.classList.contains("hidden")) openProblem();
    else closeProblem();
  };

  /* Between questions. Not a close: the panel stays where the learner left it,
     but nothing of the last problem may ride along to the next one. */
  const resetProblem = () => {
    problem.tag = "";
    _clearChips(problem.chips);
    _clearNote(problem.note);
    _setStatus(problem.status, "", false);
    closeProblem();
  };

  const _sendProblem = async () => {
    if (problem.sending) return;
    const question = typeof PracticeAPI === "undefined" ? null : PracticeAPI.currentQuestion;
    if (!question) {
      _setStatus(problem.status, "No problem on screen to report.", true);
      return;
    }
    if (!problem.tag) {
      _setStatus(problem.status, "Pick what kind of feedback this is first.", true);
      return;
    }
    const note = problem.note ? problem.note.value.trim() : "";
    const subject = question.question_id;
    problem.sending = true;
    if (problem.sendBtn) problem.sendBtn.disabled = true;
    _setStatus(problem.status, "Sending…", false);
    if (typeof PracticeSession !== "undefined") PracticeSession.holdClock("problem-feedback-submit");
    try {
      const correct = typeof practiceProgress === "undefined"
        ? null
        : practiceProgress.lastResultCorrect;
      const result = await PracticeAPI.reportProblem(
        subject, problem.tag, note, correct,
      );
      /* 🔴 The learner can move on while the post is in flight. Clearing the
         box on the way back would then wipe what they have typed about the
         NEXT question, and the status line would describe a send they cannot
         see the subject of. If the screen moved, this completion is silent. */
      if (_currentQuestionId() !== subject) return;
      // Say which of the outcomes happened. A queued repair is picked up by
      // the local runner, which may not be running this minute.
      // queuedLocally means it never left this browser — saying "logged ✓"
      // there would be a lie; success:false means not even that worked.
      if (result && result.success === false) {
        _setStatus(problem.status, "Could not save that — this browser is blocking storage.", true);
        return;
      }
      let said = "Thanks — logged ✓";
      if (result && result.queuedLocally) {
        said = "Saved on this device — it sends when you are signed in.";
      } else if (result && result.improvementQueued) {
        said = "Thanks — logged ✓ and queued for a rewrite";
      }
      _setStatus(problem.status, said, false);
      _clearNote(problem.note);
      _clearChips(problem.chips);
      problem.tag = "";
    } catch (_) {
      _setStatus(problem.status, "Could not send that — it is saved here and will retry.", true);
    } finally {
      problem.sending = false;
      if (problem.sendBtn) problem.sendBtn.disabled = false;
      if (typeof PracticeSession !== "undefined") {
        PracticeSession.releaseClock("problem-feedback-submit");
      }
    }
  };

  /* Keyed by the CONCEPT. Walking segment to segment inside one KC keeps a
     half-written note; the title is only what the panel displays. One
     definition, because the draft-clearing rule and the in-flight-send rule
     have to agree on what "the same lesson" means. */
  const _lessonKey = (ctx) => (ctx ? String(ctx.kc || ctx.title || "") : "");

  // ── Lesson panel ───────────────────────────────────────────────
  const lesson = {
    panel: null, chips: [], note: null, status: null, sendBtn: null, toggleBtn: null,
    subject: null, tag: "", sending: false, context: null,
  };

  const _openLesson = () => {
    if (!lesson.panel) return;
    lesson.panel.classList.remove("hidden");
    document.body.classList.add("dd-lesson-feedback-open");
    if (lesson.toggleBtn) lesson.toggleBtn.setAttribute("aria-expanded", "true");
    requestAnimationFrame(() => {
      if (lesson.note) lesson.note.focus({ preventScroll: true });
    });
  };

  const closeLesson = () => {
    if (!lesson.panel) return;
    lesson.panel.classList.add("hidden");
    document.body.classList.remove("dd-lesson-feedback-open");
    if (lesson.toggleBtn) lesson.toggleBtn.setAttribute("aria-expanded", "false");
  };

  const _toggleLesson = () => {
    if (!lesson.panel) return;
    if (lesson.panel.classList.contains("hidden")) _openLesson();
    else closeLesson();
  };

  /* Called on every lesson page. The panel names the concept it is about, so
     feedback written on concept 2 can never be filed against concept 3 — and
     switching concepts clears a half-written note rather than carrying it. */
  const setLessonContext = (context) => {
    const next = context || null;
    const changed = _lessonKey(lesson.context) !== _lessonKey(next);
    lesson.context = next;
    if (lesson.subject) {
      lesson.subject.textContent = next && next.title
        ? `About: ${next.title}`
        : "About this lesson";
    }
    if (changed) {
      lesson.tag = "";
      _clearChips(lesson.chips);
      _clearNote(lesson.note);
      _setStatus(lesson.status, "", false);
    }
  };

  const _sendLesson = async () => {
    if (lesson.sending) return;
    if (!lesson.context) {
      _setStatus(lesson.status, "No lesson on screen to report.", true);
      return;
    }
    if (!lesson.tag) {
      _setStatus(lesson.status, "Pick what kind of feedback this is first.", true);
      return;
    }
    const note = lesson.note ? lesson.note.value.trim() : "";
    const subject = _lessonKey(lesson.context);
    lesson.sending = true;
    if (lesson.sendBtn) lesson.sendBtn.disabled = true;
    _setStatus(lesson.status, "Sending…", false);
    try {
      const result = await PracticeAPI.reportLesson({
        kc: lesson.context.kc || "",
        lessonTitle: lesson.context.title || "",
        questionId: lesson.context.questionId,
        tag: lesson.tag,
        note,
      });
      // Same rule as the problem panel: if the learner walked to another
      // concept while this was in flight, say nothing and clear nothing.
      if (_lessonKey(lesson.context) !== subject) return;
      if (result && result.success === false) {
        _setStatus(lesson.status, "Could not save that — this browser is blocking storage.", true);
        return;
      }
      _setStatus(
        lesson.status,
        result && result.queuedLocally
          ? "Saved on this device — it sends when you are signed in."
          : "Thanks — logged ✓",
        false,
      );
      _clearNote(lesson.note);
      _clearChips(lesson.chips);
      lesson.tag = "";
    } catch (_) {
      _setStatus(lesson.status, "Could not send that — it is saved here and will retry.", true);
    } finally {
      lesson.sending = false;
      if (lesson.sendBtn) lesson.sendBtn.disabled = false;
    }
  };

  // ── Wiring ─────────────────────────────────────────────────────
  const init = () => {
    problem.panel = $("problem-feedback-panel");
    problem.note = $("problem-feedback-note");
    problem.status = $("problem-feedback-status");
    problem.sendBtn = $("problem-feedback-send");
    problem.chips = problem.panel
      ? Array.from(problem.panel.querySelectorAll(".dd-fb-chip"))
      : [];
    _bindChips(problem.chips, (tag) => {
      problem.tag = tag;
      _setStatus(problem.status, "", false);
    });
    if (problem.sendBtn) problem.sendBtn.addEventListener("click", _sendProblem);
    const problemClose = $("problem-feedback-close");
    if (problemClose) problemClose.addEventListener("click", closeProblem);
    document.querySelectorAll('.dd-fb-trigger[data-fb-target="problem"]').forEach((btn) => {
      btn.setAttribute("aria-expanded", "false");
      btn.addEventListener("click", toggleProblem);
    });

    lesson.panel = $("lesson-feedback-panel");
    lesson.note = $("lesson-feedback-note");
    lesson.status = $("lesson-feedback-status");
    lesson.sendBtn = $("lesson-feedback-send");
    lesson.subject = $("lesson-feedback-subject");
    lesson.toggleBtn = $("lesson-feedback-toggle");
    lesson.chips = lesson.panel
      ? Array.from(lesson.panel.querySelectorAll(".dd-fb-chip"))
      : [];
    _bindChips(lesson.chips, (tag) => {
      lesson.tag = tag;
      _setStatus(lesson.status, "", false);
    });
    if (lesson.sendBtn) lesson.sendBtn.addEventListener("click", _sendLesson);
    const lessonClose = $("lesson-feedback-close");
    if (lessonClose) lessonClose.addEventListener("click", closeLesson);
    if (lesson.toggleBtn) {
      lesson.toggleBtn.setAttribute("aria-expanded", "false");
      lesson.toggleBtn.addEventListener("click", _toggleLesson);
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }

  return {
    openProblem, closeProblem, resetProblem,
    setLessonContext, closeLesson,
  };
})();

window.DDFeedbackPanel = DDFeedbackPanel;
