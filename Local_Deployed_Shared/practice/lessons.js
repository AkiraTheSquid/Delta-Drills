/* ================================================================
   LESSON GATE — first-encounter exposure guard

   Before revealing a question whose target KC is new, stop and send the
   learner to the notebook section that teaches it. The panel shows one card
   per pending concept — its name, where it lives, a link into the notebook and
   a "I've read it" button — and nothing else.

   IT DOES NOT RENDER THE LESSON. It used to (2026-07-31): concept prose and
   the worked example went into #question-text, and the example's code went
   into the right-hand editor. Both are gone, because a worked example is
   commonly several code blocks that each build on the state the previous one
   left behind, and the only place that behaves correctly is a notebook. So the
   content lives in the `dd-kp-<kc>` section of the generated Colab notebook and
   this file routes to it.

   Backend mode gets pending KCs from `question.lesson_gate`; local mode
   derives them from qmatrix tags + local exposure. Finishing a KP records
   exposure, then resumes normal question rendering.
   ================================================================ */

const LessonGate = (() => {
  let lessonsData = null;
  let qmatrix = null;
  let loadFailed = false;

  const _exposureKey = () => `${getPracticeStorageKey()}_kc_exposure`;

  const _localExposure = () => {
    try {
      return JSON.parse(localStorage.getItem(_exposureKey()) || "{}") || {};
    } catch (_) {
      return {};
    }
  };

  const _markLocalExposure = (kcs) => {
    try {
      const map = _localExposure();
      const now = new Date().toISOString();
      kcs.forEach((kc) => {
        if (!map[kc]) map[kc] = now;
      });
      localStorage.setItem(_exposureKey(), JSON.stringify(map));
    } catch (_) {}
  };

  const _markBackendExposure = async (kcs) => {
    if (practiceMode !== "backend") return;
    try {
      const res = await apiFetch("/api/practice/exposure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kcs }),
      });
      if (res.status === 401) handleExpiredToken();
    } catch (_) {
      // Lost POST only means lesson may appear again on another device.
    }
  };

  const _fetchJson = async (path) => {
    // Always revalidate lesson content — a stale disk-cached JSON silently
    // shows the learner an OLD lesson after a recompile (no visible error).
    // no-cache = conditional request; server 304s when unchanged, so cheap.
    const res = await fetch(path, { cache: "no-cache" });
    if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
    return res.json();
  };

  const _ensureLessons = async () => {
    if (lessonsData || loadFailed) return;
    try {
      lessonsData = await _fetchJson("lessons/lessons_structured.json");
    } catch (err) {
      console.warn("[lessons] content unavailable — lesson gate disabled:", err);
      loadFailed = true;
    }
  };

  const _ensureQmatrix = async () => {
    if (qmatrix || loadFailed) return;
    try {
      qmatrix = await _fetchJson("lessons/qmatrix_tags.json");
    } catch (err) {
      console.warn("[lessons] qmatrix unavailable — local lesson gate disabled:", err);
      loadFailed = true;
    }
  };

  const _findKp = (kc) => {
    if (!lessonsData) return null;
    for (const lesson of lessonsData.lessons) {
      for (const kp of lesson.kps) {
        if (kp.kc === kc) return { lesson, kp };
      }
    }
    return null;
  };

  const _pendingKcs = async (question) => {
    if (question?.diagnostic_active) return [];
    if (practiceMode === "backend") {
      return [...new Set((question?.lesson_gate || []).map((entry) => entry?.kc).filter(Boolean))];
    }
    await _ensureQmatrix();
    if (!qmatrix) return [];
    const tags = qmatrix[String(question?.question_id)];
    if (!tags?.target_kcs?.length) return [];
    const exposed = _localExposure();
    return [...new Set(tags.target_kcs.filter((kc) => !exposed[kc]))];
  };

  /* ---------- The gate ------------------------------------------------- */

  /* The lessons under the "Numpy" topic teach PyTorch — every drill was
     converted to `import torch as t` in the July dialect passes — but the topic
     string is NOT free to rename. `questions.py` builds backend subtopic keys as
     `f"{topic}: {subtopic}"`, so that word is the key every learner's stored BKT
     mastery is filed under; changing it in place orphans their history. Rename
     the LABEL only, and leave the key alone until there is a state migration. */
  const TOPIC_LABELS = { Numpy: "PyTorch tensors" };
  const _topicLabel = (topic) => TOPIC_LABELS[topic] || topic;

  /* Escapes both quote forms as well as the angle brackets, because some of
     these values land inside a double-quoted attribute (`href=`, `target=`)
     rather than in element text — a bare `"` there closes the attribute early
     and lets the rest of the value become markup. */
  const esc = (value) =>
    String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const _el = (id) => document.getElementById(id);

  /* One card per pending concept.

     This screen used to BE the lesson: it rendered the concept prose, the
     "watch out" note and the worked example into #question-text, and pushed the
     example's code into the editor in the right-hand panel. That is gone
     (2026-07-31). A worked example is regularly a sequence of code blocks that
     each depend on the state the previous one left behind, and reproducing that
     faithfully means reproducing a notebook — so the example lives in the
     notebook, and this screen's job shrank to: say which concept is next, send
     the learner to the section that teaches it, and wait for them to come back.

     What did NOT shrink is the bookkeeping. Finishing a concept still records
     exposure (locally and, in backend mode, server-side) and still credits the
     ladder's `worked` rung through LadderUI, because the rung the pending
     question is staged on depends on it. */
  const _pendingList = (kcs) => {
    const items = [];
    for (const kc of kcs) {
      const found = _findKp(kc);
      if (!found) continue;
      items.push({ kc, lesson: found.lesson, kp: found.kp });
    }
    return items;
  };

  const _cardHtml = (item, index, total) => {
    const url = window.ColabRoute ? window.ColabRoute.urlForKc(item.kc) : "";
    const counter = total > 1 ? ` · concept ${index + 1} of ${total}` : "";
    const where = url
      ? `<a class="lesson-gate-link" id="lesson-gate-link" href="${esc(url)}"
            target="${esc(window.ColabRoute.TARGET)}">Read it in the notebook ↗</a>`
      // No mapping for this KC. Say so plainly instead of showing a dead link:
      // the learner can still continue, they just get no reading first.
      : '<span class="lesson-gate-note" id="lesson-gate-link">This concept has no notebook section yet.</span>';
    return (
      '<div class="lesson-gate-card">' +
      `<div class="lesson-gate-eyebrow">New concept${esc(counter)}</div>` +
      `<h2 class="lesson-gate-title" id="lesson-gate-title" tabindex="-1">${esc(item.kp.title)}</h2>` +
      `<div class="lesson-gate-where">${esc(_topicLabel(item.lesson.topic))} · ` +
      `${esc(item.lesson.title)}</div>` +
      '<p class="lesson-gate-blurb">Read this section and run its examples in the notebook, ' +
      'then come back here.</p>' +
      `<div class="lesson-gate-actions">${where}` +
      '<button type="button" class="primary" id="lesson-gate-continue-btn">' +
      (index === total - 1 ? "I've read it — continue to the question →" : "I've read it — next concept →") +
      "</button></div></div>"
    );
  };

  /* Point the page-wide topbar at the concept this gate is about.

     The estimate is fetched rather than read off the pending question because
     the gate may cover several concepts and at most one of them is the one that
     question is staged on — labelling one concept's record with another's
     attempts would be worse than showing none. Guest mode has no backend to
     ask, so the fetch is skipped there. */
  const _showTopbar = async (item, index, total) => {
    const bar = window.ConceptTopbar;
    if (!bar) return;
    bar.show({
      kc: item.kc,
      title: item.kp.title,
      eyebrow: `${_topicLabel(item.lesson.topic)} · Concept ${index + 1} of ${total}`,
      stage: "lesson",
      estimate: null,
    });
    if (typeof apiFetch !== "function" || practiceMode !== "backend") return;
    try {
      const res = await apiFetch(
        `/api/practice/kc-estimate?kc=${encodeURIComponent(item.kc)}`,
      );
      if (!res.ok) return;
      const data = await res.json();
      // The learner may have moved on while this was in flight. Only apply the
      // estimate if the topbar is still showing the concept it belongs to.
      if (bar.activeKc() === item.kc) bar.setEstimate(data.ladder_estimate);
    } catch (err) {
      console.warn("[lessons] concept estimate unavailable:", err);
    }
  };

  const _cleanup = () => {
    document.body.classList.remove("lesson-mode");
    const host = _el("lesson-gate");
    if (host) {
      host.innerHTML = "";
      host.classList.add("hidden");
    }
  };

  const maybeShow = async (question, onDone, forceKcs = null) => {
    try {
      const kcs = forceKcs || (await _pendingKcs(question));
      if (!kcs.length) return false;
      await _ensureLessons();
      if (!lessonsData) return false;
      const items = _pendingList(kcs);
      if (!items.length) return false;

      const host = _el("lesson-gate");
      if (!host) return false;
      // The URLs come out of the generated notebook map, which is fetched
      // once. Awaiting it here means the very first gate of a session still
      // renders a working link instead of the "no notebook section" fallback.
      if (window.ColabRoute) await window.ColabRoute.load();

      document.body.classList.add("lesson-mode");
      host.classList.remove("hidden");
      let index = 0;
      let finished = false;
      // Concepts the learner actually clicked through to the end.
      const taught = [];

      const finishAll = async () => {
        if (finished) return;
        finished = true;
        // Credit the ladder BEFORE handing back. This gate stands in for the
        // worked example, so the concept comes off the `worked` rung here — and
        // the question `onDone` is about to render was staged for the rung the
        // learner has just left. LadderUI re-stages it in place, which is why
        // this is awaited rather than fired and forgotten.
        if (window.LadderUI && typeof window.LadderUI.creditTaught === "function") {
          try {
            await window.LadderUI.creditTaught(taught, question);
          } catch (err) {
            console.warn("[lessons] could not credit worked examples:", err);
          }
        }
        _cleanup();
        onDone();
      };

      const showCard = () => {
        const item = items[index];
        host.innerHTML = _cardHtml(item, index, items.length);
        window.scrollTo({ top: 0 });
        const title = _el("lesson-gate-title");
        if (title) title.focus();

        // Steer the shared Colab tab to the concept, so the reading is one
        // click away rather than a hunt through a 500-cell notebook. Only ever
        // best-effort — a browser allows window.open during a user gesture, and
        // the click that advanced to this card is one, but a popup blocker may
        // still decline. The link above is the guaranteed way through, which is
        // why it is rendered as a real anchor rather than a button.
        if (window.ColabRoute) window.ColabRoute.openKc(item.kc);

        // Not awaited: the card is already on screen and the estimate fills in
        // when it arrives. Blocking on a network call would make a slow
        // connection look like a broken gate.
        _showTopbar(item, index, items.length);

        const button = _el("lesson-gate-continue-btn");
        let advancing = false;
        button.onclick = () => {
          if (advancing || finished) return;
          advancing = true;
          button.disabled = true;
          _markLocalExposure([item.kc]);
          _markBackendExposure([item.kc]);
          // Also credits the ladder's `worked` rung — but in finishAll, not
          // here, because the response re-stages the pending question and that
          // has to land before it renders. Without any crediting at all, the
          // gate would teach a KP and the ladder would immediately re-teach the
          // identical concept: the two counters are written by different
          // endpoints and neither one implies the other.
          taught.push(item.kc);
          index++;
          if (index < items.length) showCard();
          else finishAll();
        };
      };

      showCard();
      return true;
    } catch (err) {
      console.warn("[lessons] gate error — continuing without lesson:", err);
      _cleanup();
      return false;
    }
  };

  /* `question` is optional and is NOT used to pick the lesson (`kc` does
     that) — it is the card waiting behind this screen, so finishAll can hand
     it to the ladder to be re-staged before it renders. Omit it and the
     lesson still shows; the caller is then responsible for the re-staging. */
  const showLesson = (kc, onDone = () => {}, question = null) =>
    maybeShow(question, onDone, [kc]);

  // Exposed for the single-KC ladder (kc-practice.js): it needs the KP's
  // faded/independent item lists and the lesson's subtopic_key, which are the
  // same records the gate already loads.
  const getKpEntry = async (kc) => {
    await _ensureLessons();
    return _findKp(kc);
  };

  return {
    maybeShow,
    showLesson,
    getKpEntry,
    // `renderMarkdown` and `activeQuestion` were exported here. The first was
    // this file's markdown subset, shared with the ladder so its inline worked
    // example matched the lesson screen's; the second told the Pyodide
    // bootstrap that a LESSON's example was running rather than a question.
    // Neither has a caller now — lesson prose and worked examples are notebook
    // cells, and nothing in the panel renders or runs them.
  };
})();

window.LessonGate = LessonGate;

/* ?lesson=<kc> — teach one KC, then drill it (kc-practice.js) until mastered.
   No session quota or timer: this flow ends on the mastery gate, not a count. */
(function () {
  const kc = new URLSearchParams(location.search).get("lesson");
  if (!kc) return;
  // Suppresses renderQuestion while the lesson pages are up (a late session
  // resume must not clobber them). KcPractice.start() clears it.
  window.__lessonDemoOnly = true;

  /* #question-text is gone, so the demo path's own messages go in the gate
     host — the one element on this page that still holds prose.

     `title` and `body` are set with textContent, NEVER interpolated into the
     markup. `kc` comes straight off `?lesson=` in the URL and one of these
     messages quotes it back, so building this as an HTML string would let a
     crafted link run script in the app's origin. The markup here is fixed and
     the variable parts are text. */
  const _fallbackMessage = (title, body) => {
    const host = document.getElementById("lesson-gate");
    if (!host) return;
    host.classList.remove("hidden");
    host.innerHTML =
      '<div class="lesson-gate-card">' +
      '<h2 class="lesson-gate-title"></h2><p class="lesson-gate-blurb"></p></div>';
    host.querySelector(".lesson-gate-title").textContent = title;
    host.querySelector(".lesson-gate-blurb").textContent = body;
  };

  const _beginLadder = async () => {
    try {
      const started = await window.KcPractice.start(kc);
      if (!started) {
        _fallbackMessage(
          "Lesson complete",
          "No practice problems are attached to this concept yet.",
        );
        return;
      }
      const nextQ = await PracticeAPI.getNextQuestion();
      if (!nextQ) throw new Error("no question available");
      renderQuestion(nextQ, 1);
    } catch (err) {
      console.warn("[lessons] could not start KC practice:", err);
      _fallbackMessage(
        "Lesson complete",
        "Practice problems could not be loaded. Reload to try again.",
      );
    }
  };

  const start = () => {
    const page = document.getElementById("page-practice");
    document.querySelectorAll(".tab").forEach((tab) =>
      tab.classList.toggle("active", tab.dataset.tab === "practice"));
    document.querySelectorAll(".page").forEach((candidate) =>
      candidate.classList.toggle("hidden", candidate.id !== "page-practice"));
    if (page) page.classList.remove("session-idle");
    window.LessonGate.showLesson(kc, _beginLadder).then((shown) => {
      if (!shown) {
        _fallbackMessage(
          "No lesson found",
          `No lesson found for "${kc}". Check the KC id.`,
        );
      }
    });
  };
  if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", () => setTimeout(start, 300));
  } else {
    setTimeout(start, 300);
  }
})();
