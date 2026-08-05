/* ================================================================
   LESSON GATE — first-encounter exposure guard

   Before revealing a question whose target KC is new, use the normal
   practice split as a lesson screen. Left side teaches one concept and
   explains one worked example. Right-side editor receives that example's
   complete runnable code. Running/editing it is optional. No faded exercise,
   grading, or popup appears inside the lesson.

   Backend mode gets pending KCs from `question.lesson_gate`; local mode
   derives them from qmatrix tags + local exposure. Finishing a KP records
   exposure, then resumes normal question rendering.
   ================================================================ */

const LessonGate = (() => {
  let lessonsData = null;
  let qmatrix = null;
  let loadFailed = false;
  let activeQuestion = null; // Truthy during lesson → Run uses local Pyodide.
  const DEFAULT_EDITOR = "import numpy as np\nnp.random.seed(0)\n\n# Write your solution here\n";

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

  /* ---------- Markdown subset (mirrors lessons/viewer.html) ------------ */

  const esc = (value) =>
    String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const inline = (value) =>
    esc(value)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>");

  const md = (text, { renderCode = true } = {}) => {
    if (!text) return "";
    const lines = text.split("\n");
    const out = [];
    let i = 0;
    let list = null;
    let para = [];
    let quote = [];
    const flushPara = () => {
      if (para.length) {
        out.push("<p>" + inline(para.join(" ")) + "</p>");
        para = [];
      }
    };
    const flushList = () => {
      if (list) {
        out.push("</" + list + ">");
        list = null;
      }
    };
    const flushQuote = () => {
      if (quote.length) {
        out.push("<blockquote>" + inline(quote.join(" ")) + "</blockquote>");
        quote = [];
      }
    };

    while (i < lines.length) {
      const line = lines[i];
      const fence = line.match(/^```(.*)$/);
      if (fence) {
        flushPara();
        flushList();
        flushQuote();
        const buf = [];
        i++;
        while (i < lines.length && !/^```/.test(lines[i])) {
          buf.push(lines[i]);
          i++;
        }
        i++;
        // The info string is carried into the DOM rather than dropped. It is
        // the authoring format's own runnable marker: a plain ```python fence
        // is executed by validate_lessons.py against a shared per-file
        // namespace, top to bottom, and ```python no-run is skipped. That
        // makes it exactly the right signal for which blocks get a Run button,
        // so practice/notebook.js reads it off the rendered element. Deciding
        // it here would make the one shared renderer notebook-aware, and the
        // ladder's inline example and lessons/viewer.html use it too.
        if (renderCode) {
          out.push(
            '<pre data-fence="' + esc(fence[1].trim()) + '"><code>' +
              esc(buf.join("\n")) +
              "</code></pre>",
          );
        }
        continue;
      }
      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        flushPara();
        flushList();
        flushQuote();
        out.push("<h4>" + inline(heading[2]) + "</h4>");
        i++;
        continue;
      }
      if (/^>\s?/.test(line)) {
        flushPara();
        flushList();
        quote.push(line.replace(/^>\s?/, ""));
        i++;
        continue;
      }
      const item = line.match(/^(\s*)([-*]|\d+\.)\s+(.*)$/);
      if (item) {
        flushPara();
        flushQuote();
        const kind = /\d+\./.test(item[2]) ? "ol" : "ul";
        if (list !== kind) {
          flushList();
          out.push("<" + kind + ">");
          list = kind;
        }
        let itemText = item[3];
        i++;
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*]|\d+\.)\s/.test(lines[i])) {
          itemText += " " + lines[i].trim();
          i++;
        }
        out.push("<li>" + inline(itemText) + "</li>");
        continue;
      }
      if (!line.trim()) {
        flushPara();
        flushList();
        flushQuote();
        i++;
        continue;
      }
      flushQuote();
      para.push(line.trim());
      i++;
    }
    flushPara();
    flushList();
    flushQuote();
    return out.join("\n");
  };

  const _firstPythonFence = (text) => {
    const match = String(text || "").match(/```python\s*\n([\s\S]*?)```/);
    return match ? match[1].replace(/\s+$/, "") + "\n" : "";
  };

  /* ---------- One inline screen per concept segment -------------------- */

  const _buildPages = (kcs) => {
    const pages = [];
    for (const kc of kcs) {
      const found = _findKp(kc);
      if (!found) continue;
      const { lesson, kp } = found;
      const segments = kp.segments?.length
        ? kp.segments
        : [{
            title: "",
            concept_markdown: kp.concept_markdown,
            watch_out_markdown: "",
            worked_example_markdown: kp.worked_example_markdown,
            worked_example_code: _firstPythonFence(kp.worked_example_markdown),
          }];
      segments.forEach((seg, index) => {
        pages.push({
          lesson,
          kp,
          seg,
          segCount: segments.length,
          lastOfKp: index === segments.length - 1,
        });
      });
    }
    pages.forEach((page, index) => {
      page.pageIndex = index;
      page.pageTotal = pages.length;
    });
    return pages;
  };

  /* The lessons under the "Numpy" topic teach PyTorch — every drill was
     converted to `import torch as t` in the July dialect passes — but the topic
     string is NOT free to rename. `questions.py` builds backend subtopic keys as
     `f"{topic}: {subtopic}"`, so that word is the key every learner's stored BKT
     mastery is filed under; changing it in place orphans their history. Rename
     the LABEL only, and leave the key alone until there is a state migration. */
  const TOPIC_LABELS = { Numpy: "PyTorch tensors" };
  const _topicLabel = (topic) => TOPIC_LABELS[topic] || topic;

  const _pageHtml = (page) => {
    const { kp, seg } = page;
    const pageTitle = seg.title || kp.title;
    const watchOut = seg.watch_out_markdown ||
      (page.segCount === 1 ? kp.misconceptions_markdown : "");
    const isLast = page.pageIndex === page.pageTotal - 1;
    // The concept name, the graph button and the progress counter used to be
    // built here, inside the panel. They now live in #concept-topbar, which
    // spans the page and survives this innerHTML being replaced — see
    // practice/concept-topbar.js and `_showTopbar` below.
    let html = `<h2 class="lesson-kp-title" id="lesson-title" tabindex="-1">${esc(pageTitle)}</h2>`;
    // `nb-scope` marks the regions whose ```python fences are programs rather
    // than illustrations — the same two sections validate_lessons.py executes
    // against one shared namespace. LessonNotebook turns those into cells; a
    // fence in "Watch out" is outside the scope and stays static, because CI
    // never runs it and a Run button on unrun code is a trap.
    html += '<div class="lesson-body nb-scope">' + md(seg.concept_markdown) + "</div>";
    if (watchOut) {
      html += '<div class="lesson-watch-out"><h3>Watch out</h3>' + md(watchOut) + "</div>";
    }
    // The worked example's code is rendered INLINE now, where the prose that
    // explains it is, rather than being shipped off to the editor in the other
    // panel. `renderCode: false` used to strip it precisely because it lived
    // over there; keeping the two in one column is the whole point of the
    // notebook layout, so the fences stay and LessonNotebook makes them run.
    html += '<div class="lesson-worked nb-scope"><h3>Worked example</h3>' +
      md(seg.worked_example_markdown) +
      '<p class="lesson-example-note">Run any block to see it execute. A block runs ' +
      "everything above it too, so the variables it needs already exist.</p></div>";
    html += '<div class="lesson-actions"><button type="button" class="primary" id="lesson-continue-btn">' +
      (isLast ? "Continue to the question →" : "Next concept →") +
      "</button></div>";
    return html;
  };

  /* Point the page-wide topbar at the concept this page teaches.

     The estimate is fetched rather than read off the pending question, for two
     reasons: the lesson may teach several KPs and at most one of them is the
     concept that question is staged on, and the ?lesson=<kc> entry point has no
     question at all. A failed fetch leaves the estimate blank rather than
     falling back to the staged question's number — labelling one concept's
     record with another concept's attempts would be worse than showing none.

     Guest mode has no backend to ask, so the fetch is skipped entirely there. */
  const _showTopbar = async (page) => {
    const bar = window.ConceptTopbar;
    if (!bar) return;
    const { lesson, kp, seg } = page;
    bar.show({
      kc: kp.kc,
      title: seg.title || kp.title,
      eyebrow: `${_topicLabel(lesson.topic)} · Lesson ${page.pageIndex + 1} of ${page.pageTotal}`,
      stage: "lesson",
      estimate: null,
    });
    if (typeof apiFetch !== "function" || practiceMode !== "backend") return;
    try {
      const res = await apiFetch(
        `/api/practice/kc-estimate?kc=${encodeURIComponent(kp.kc)}`,
      );
      if (!res.ok) return;
      const data = await res.json();
      // The learner may have paged on while this was in flight. Only apply the
      // estimate if the topbar is still showing the concept it belongs to.
      if (bar.activeKc() === kp.kc) bar.setEstimate(data.ladder_estimate);
    } catch (err) {
      console.warn("[lessons] concept estimate unavailable:", err);
    }
  };

  /* What the runner needs to know about the code on a lesson page.

     `primary_library` is "torch" for every lesson, including the einops ones:
     the July dialect conversion rewrote the whole course into the torch
     dialect, so an einops lesson imports einops AND torch. This field said
     "numpy" from before that conversion, and `questionIsTorch` reads it — so
     every lesson cell was routed to Pyodide, which cannot import torch, and
     every Run button on every lesson answered with a ModuleNotFoundError
     traceback. Signed-in learners included; the backend fork runner has torch
     preimported and was never asked. */
  const _runtimeContext = (page) => ({
    topic: "Lesson",
    lesson_topic: page.lesson.topic,
    primary_library: "torch",
    supports_visual_output: false,
    test_cases: [],
  });

  const _el = (id) => document.getElementById(id);

  const _cleanup = () => {
    document.body.classList.remove("lesson-mode");
    activeQuestion = null;
    const editor = _el("code-editor");
    if (editor) editor.value = DEFAULT_EDITOR;
    const out = _el("output-area");
    if (out) out.textContent = "";
  };

  const maybeShow = async (question, onDone, forceKcs = null) => {
    try {
      const kcs = forceKcs || (await _pendingKcs(question));
      if (!kcs.length) return false;
      await _ensureLessons();
      if (!lessonsData) return false;
      const pages = _buildPages(kcs);
      if (!pages.length) return false;

      const questionText = _el("question-text");
      const questionNumber = _el("question-number");
      const editor = _el("code-editor");
      const output = _el("output-area");
      if (!questionText || !editor) return false;

      document.body.classList.add("lesson-mode");
      let index = 0;
      let finished = false;
      // KPs whose worked example the learner actually read through to the end.
      const taught = [];

      const finishAll = async () => {
        if (finished) return;
        finished = true;
        // Credit the ladder BEFORE handing back. This screen WAS the worked
        // example, so the concept comes off the `worked` rung here — and the
        // question `onDone` is about to render was staged for the rung the
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

      const showPage = () => {
        const page = pages[index];
        activeQuestion = _runtimeContext(page);
        if (questionNumber) questionNumber.textContent = "Lesson";
        questionText.innerHTML = _pageHtml(page);
        // Every runnable block on the page becomes a cell, explanation blocks
        // included, and they share state top to bottom. Mounting the whole
        // page rather than `.lesson-worked` is what lets a concept be taught
        // as a sequence of things the learner can run at the point the prose
        // raises them; `no-run` fences stay static. Done after the innerHTML
        // rather than inside _pageHtml because it rewrites rendered DOM, which
        // keeps the markdown renderer shared and notebook-unaware.
        if (window.LessonNotebook) {
          window.LessonNotebook.mount(questionText);
        }
        questionText.scrollTop = 0;
        window.scrollTo({ top: 0 });
        editor.value = page.seg.worked_example_code ||
          _firstPythonFence(page.seg.worked_example_markdown) || DEFAULT_EDITOR;
        if (output) output.textContent = "";

        // Not awaited: the page is already rendered and the estimate fills in
        // when it arrives. Blocking the render on a network call would make a
        // slow connection look like a broken lesson.
        _showTopbar(page);

        const button = _el("lesson-continue-btn");
        let advancing = false;
        button.onclick = () => {
          if (advancing || finished) return;
          advancing = true;
          button.disabled = true;
          if (page.lastOfKp) {
            _markLocalExposure([page.kp.kc]);
            _markBackendExposure([page.kp.kc]);
            // Also credits the ladder's `worked` rung — but in finishAll, not
            // here, because the response re-stages the pending question and
            // that has to land before it renders. Without any crediting at
            // all, the gate would teach a KP and the ladder would immediately
            // re-teach the identical page: the two counters are written by
            // different endpoints and neither one implies the other.
            taught.push(page.kp.kc);
          }
          index++;
          if (index < pages.length) showPage();
          else finishAll();
        };
      };

      showPage();
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
    // The ladder renders the same worked-example markdown beside faded and
    // partial problems. Sharing this renderer keeps that example looking
    // identical to the one taught on the lesson screen — a second, subtly
    // different markdown subset would read as a different example.
    renderMarkdown: md,
    get activeQuestion() {
      return activeQuestion;
    },
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

  const _fallbackMessage = (html) => {
    const text = document.getElementById("question-text");
    if (text) text.innerHTML = html;
  };

  const _beginLadder = async () => {
    try {
      const started = await window.KcPractice.start(kc);
      if (!started) {
        _fallbackMessage(
          '<h2 class="lesson-kp-title">Lesson complete</h2>' +
          "<p>No practice problems are attached to this concept yet.</p>",
        );
        return;
      }
      const nextQ = await PracticeAPI.getNextQuestion();
      if (!nextQ) throw new Error("no question available");
      renderQuestion(nextQ, 1);
    } catch (err) {
      console.warn("[lessons] could not start KC practice:", err);
      _fallbackMessage(
        '<h2 class="lesson-kp-title">Lesson complete</h2>' +
        "<p>Practice problems could not be loaded. Reload to try again.</p>",
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
        const text = document.getElementById("question-text");
        if (text) text.textContent = `No lesson found for "${kc}". Check the KC id.`;
      }
    });
  };
  if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", () => setTimeout(start, 300));
  } else {
    setTimeout(start, 300);
  }
})();
