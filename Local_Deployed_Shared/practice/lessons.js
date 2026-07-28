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
        if (renderCode) out.push("<pre><code>" + esc(buf.join("\n")) + "</code></pre>");
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
    const { lesson, kp, seg } = page;
    const pageTitle = seg.title || kp.title;
    const watchOut = seg.watch_out_markdown ||
      (page.segCount === 1 ? kp.misconceptions_markdown : "");
    const isLast = page.pageIndex === page.pageTotal - 1;
    let html =
      '<div class="lesson-topbar">' +
      `<span class="lesson-topbar-topic">${esc(_topicLabel(lesson.topic))} · ${esc(pageTitle)}</span>` +
      // The lesson already knows its own KC, so unlike the practice-view button
      // this needs no q-matrix lookup — it can always land somewhere real.
      `<button type="button" class="lesson-graph-jump" data-kc="${esc(kp.kc)}" ` +
      `title="Open “${esc(kp.kc)}” in the knowledge graph">See in knowledge graph</button>` +
      `<span class="lesson-topbar-progress">Lesson ${page.pageIndex + 1} of ${page.pageTotal}</span>` +
      "</div>";
    html += `<h2 class="lesson-kp-title" id="lesson-title" tabindex="-1">${esc(pageTitle)}</h2>`;
    html += '<div class="lesson-body">' + md(seg.concept_markdown) + "</div>";
    if (watchOut) {
      html += '<div class="lesson-watch-out"><h3>Watch out</h3>' + md(watchOut) + "</div>";
    }
    html += '<div class="lesson-worked"><h3>Worked example</h3>' +
      md(seg.worked_example_markdown, { renderCode: false }) +
      '<p class="lesson-example-note">Example code is preloaded on the right. Run or edit it only if useful.</p></div>';
    html += '<div class="lesson-actions"><button type="button" class="primary" id="lesson-continue-btn">' +
      (isLast ? "Continue to the question →" : "Next concept →") +
      "</button></div>";
    return html;
  };

  const _runtimeContext = (page) => ({
    topic: "Lesson",
    lesson_topic: page.lesson.topic,
    primary_library: page.lesson.topic === "Einops" ? "einops" : "numpy",
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

      const finishAll = () => {
        if (finished) return;
        finished = true;
        _cleanup();
        onDone();
      };

      const showPage = () => {
        const page = pages[index];
        activeQuestion = _runtimeContext(page);
        if (questionNumber) questionNumber.textContent = "Lesson";
        questionText.innerHTML = _pageHtml(page);
        questionText.scrollTop = 0;
        window.scrollTo({ top: 0 });
        editor.value = page.seg.worked_example_code ||
          _firstPythonFence(page.seg.worked_example_markdown) || DEFAULT_EDITOR;
        if (output) output.textContent = "";

        const jump = questionText.querySelector(".lesson-graph-jump");
        if (jump) {
          jump.onclick = () => {
            if (typeof switchTab === "function") switchTab("knowledge-graph");
            // The graph can only size itself once its page is visible, and on a
            // first visit it still has to build — deltaFocusConceptGraphKc waits
            // for both. Same contract the practice-view button uses.
            requestAnimationFrame(() => {
              if (typeof window.deltaFocusConceptGraphKc === "function") {
                window.deltaFocusConceptGraphKc(jump.dataset.kc);
              }
            });
          };
        }

        const button = _el("lesson-continue-btn");
        let advancing = false;
        button.onclick = () => {
          if (advancing || finished) return;
          advancing = true;
          button.disabled = true;
          if (page.lastOfKp) {
            _markLocalExposure([page.kp.kc]);
            _markBackendExposure([page.kp.kc]);
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

  const showLesson = (kc, onDone = () => {}) => maybeShow(null, onDone, [kc]);

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
          '<div class="lesson-topbar"><span class="lesson-topbar-topic">Lesson complete</span></div>' +
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
        '<div class="lesson-topbar"><span class="lesson-topbar-topic">Lesson complete</span></div>' +
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
