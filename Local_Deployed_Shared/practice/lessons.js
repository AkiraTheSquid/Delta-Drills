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
  // config.js owns the text; see DEFAULT_EDITOR_CODE there for why it is torch.
  const DEFAULT_EDITOR = DEFAULT_EDITOR_CODE;

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

  /* ONE CONCEPT PER VISIT.

     A KP is not one idea — `kp-ndarray-model` teaches three, and the markdown
     has always been written that way, each concept with its own worked example
     and its own faded drill. The gate used to render all of them back to back
     and then hand over one question, which produces a learner who has read
     three things and practised one. A step is therefore a CONCEPT, not a KP:
     teach it, drill it, come back for the next one.

     `exposureKey` is what gets posted when the page is read: the KC itself for
     a single-concept KP (unchanged, and that is 32 of the 63), `<kc>#<id>` for
     one concept of a longer one. The KC's own key stays reserved for "the whole
     KP is done" — post it early and the remaining concepts are never taught. */
  const _stepFor = (kc, exposed) => {
    const segments = (_findKp(kc)?.kp?.segments || []).filter((seg) => seg.concept_id);
    const whole = { kc, segmentIndex: 0, segmentTotal: 1, exposureKey: kc };
    if (segments.length < 2) return whole;
    for (let i = 0; i < segments.length; i += 1) {
      const key = `${kc}#${segments[i].concept_id}`;
      if (!exposed[key]) {
        return { kc, segmentIndex: i, segmentTotal: segments.length, exposureKey: key };
      }
    }
    return { ...whole, segmentIndex: segments.length - 1, segmentTotal: segments.length };
  };

  /* Backend mode takes the step from the gate entry rather than recomputing it:
     the server holds the authoritative exposure map (it is per account, not per
     browser), and a second opinion derived from localStorage would re-teach a
     concept the learner finished on another device. */
  const _stepFromGate = (entry) => ({
    kc: entry.kc,
    segmentIndex: Number(entry.segment_index) || 0,
    segmentTotal: Number(entry.segment_total) || 1,
    exposureKey: entry.exposure_key || entry.kc,
  });

  const _pendingSteps = async (question) => {
    if (question?.diagnostic_active) return [];
    if (practiceMode === "backend") {
      /* 🔴 …MINUS ANYTHING THIS BROWSER HAS ALREADY SHOWN.

         `lesson_gate` is attached by the SERVER when the question is served,
         and it is a snapshot: nothing the learner does afterwards changes the
         copy riding on a question object already in this tab. The exposure
         posts made when a page is read are about the NEXT question the server
         picks.

         `timer.js:resume()` asks about exactly that object, rehydrated from
         what was persisted at pause. So pausing on a drill and pressing
         Continue re-taught, from page one, the concept the learner had just
         read on the way to that drill — the snapshot still said it was
         pending. Reproduced on prod: the local exposure map held
         `numpy.ndarray-model#s0-…` while the question's gate still listed it.

         So a page whose `exposure_key` is already in this browser's map is
         dropped. That is a SUPPRESSION on top of the server's decision, never
         a replacement for it: the server stays authoritative for what to teach
         next (it is per account, not per browser, which is why `_stepFromGate`
         exists at all), and this only declines to show the same page to the
         same browser twice. Filtered BEFORE the dedupe, so an entry that is
         dropped cannot take a later, unread entry for the same KC with it. */
      const exposed = _localExposure();
      const seen = new Set();
      return (question?.lesson_gate || [])
        .filter((entry) => entry?.kc && !exposed[entry.exposure_key || entry.kc])
        .filter((entry) => !seen.has(entry.kc) && seen.add(entry.kc))
        .map(_stepFromGate);
    }
    await _ensureQmatrix();
    if (!qmatrix) return [];
    const tags = qmatrix[String(question?.question_id)];
    if (!tags?.target_kcs?.length) return [];
    const exposed = _localExposure();
    return [...new Set(tags.target_kcs.filter((kc) => !exposed[kc]))]
      .map((kc) => _stepFor(kc, exposed));
  };

  /* ---------- Markdown subset (mirrors lessons/viewer.html) ------------ */

  const esc = (value) =>
    String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const inline = (value) =>
    esc(value)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>");

  /* `headingLevels` keeps a heading's authored depth instead of flattening it.

     Every caller but one wants the flattening. A lesson page is ONE concept
     inside a panel that already has an <h2> title, so an authored `#` there is
     a sub-heading of that title and rendering it as an <h1> would put a second
     page title under the first. The notebook view is the exception: it renders
     a whole lesson at once — lesson title, KP titles, segment titles, problem
     headers — and the depths are the only thing that says which of those
     contains which. Flattened, a 656-cell notebook is 400 identical bumps. */
  const md = (text, { renderCode = true, headingLevels = false } = {}) => {
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
        const tag = headingLevels ? `h${heading[1].length}` : "h4";
        out.push(`<${tag}>` + inline(heading[2]) + `</${tag}>`);
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

  const _buildPages = (steps) => {
    const pages = [];
    for (const step of steps) {
      const found = _findKp(step.kc);
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
      // One page — the concept this visit owes. The rest of the KP is not
      // skipped, it is DEFERRED: the gate fires again for concept 2 as soon as
      // the drill for concept 1 has been answered, because the KC's own
      // exposure key is only written once the last one is read.
      const index = Math.min(Math.max(step.segmentIndex, 0), segments.length - 1);
      pages.push({
        lesson,
        kp,
        step,
        seg: segments[index],
        segIndex: index,
        segCount: segments.length,
        lastOfKp: index === segments.length - 1,
      });
    }
    pages.forEach((page, index) => {
      page.pageIndex = index;
      page.pageTotal = pages.length;
    });
    return pages;
  };

  /* The lessons under the "Numpy" topic teach PyTorch, but the topic string is
     the key every learner's stored mastery is filed under — see
     `displayTopic` in practice/config.js, which is now the single copy of that
     map. This file used to hold its own, which is how the lesson header could
     say "PyTorch tensors" while the question header below it said "Numpy". */
  const _topicLabel = (topic) =>
    (typeof displayTopic === "function" ? displayTopic(topic) : topic);

  /* The published notebook's anchor for the concept a lesson page teaches, or
     "" when there is nowhere to send the learner.

     Empty is the ordinary answer on the normal app (no Colab edition) and the
     honest answer for a concept whose lesson was never published — both fall
     through to the full in-panel lesson below. Same escape hatch as
     `dd-no-notebook` on the question side: a rail pointing at a notebook that
     does not exist is worse than the page it replaced. */
  const _colabLessonHref = (page) => {
    const dd = window.DDColab;
    if (!dd || typeof dd.active !== "function" || !dd.active()) return "";
    if (typeof dd.hrefForKc !== "function") return "";
    // The page's own exposure key, not just its KC. A segmented KP has one
    // notebook section per concept, and this page is exactly one of them —
    // handing over the KC would open all of them and the "Concept 2 of 3" the
    // topbar is showing would name nothing on screen.
    return dd.hrefForKc(page.kp.kc, page.step && page.step.exposureKey) || "";
  };

  /* The lesson as a RAIL rather than as the lesson.

     The notebook already contains this exact prose, its runnable blocks, the
     worked example, the problem, the hints and the solution — in that order, as
     real Colab cells against a real torch runtime. Rendering a second copy in
     the panel put the lesson on the left and the work on the right, which is
     the split the Colab edition exists to remove. So on this edition the panel
     keeps what only it can know — the rung, the estimate, which page of how
     many — and hands the reading itself to the notebook.

     The continue button is unchanged in behaviour and only re-labelled: it
     still credits the `worked` rung through `finishAll`, which is what takes
     the concept off the lesson step. "I've read it" is a claim about the
     learner, and it is the same claim the full page's button was already
     making. */
  const _colabPageHtml = (page, href, isLast) =>
    `<h2 class="lesson-kp-title" id="lesson-title" tabindex="-1">` +
    `${esc(page.seg.title || page.kp.title)}</h2>` +
    '<div class="lesson-colab-card">' +
    '<p class="lesson-colab-eyebrow">📗 Read this one in the notebook</p>' +
    "<p>This is the Colab edition — the lesson is open beside you in its " +
    "notebook, where every block runs against real PyTorch. Read it through, " +
    "run whatever you want to see, then come back here.</p>" +
    `<a class="lesson-colab-open" href="${esc(href)}" target="_blank" ` +
    'rel="noopener">Open the lesson in Colab ↗</a>' +
    "</div>" +
    '<div class="lesson-actions"><button type="button" class="primary" ' +
    'id="lesson-continue-btn">' +
    (isLast ? "I've read it — give me the problem →" : "I've read it — next concept →") +
    "</button></div>";

  const _pageHtml = (page) => {
    const { kp, seg } = page;
    const pageTitle = seg.title || kp.title;
    const watchOut = seg.watch_out_markdown ||
      (page.segCount === 1 ? kp.misconceptions_markdown : "");
    const isLast = page.pageIndex === page.pageTotal - 1;
    const colabHref = _colabLessonHref(page);
    if (colabHref) return _colabPageHtml(page, colabHref, isLast);
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

  const _showTopbar = async (page) => {
    const bar = window.StageLadder;
    if (!bar) return;
    const { lesson, kp, seg } = page;
    bar.show({
      kc: kp.kc,
      title: seg.title || kp.title,
      eyebrow: page.segCount > 1
        ? `${_topicLabel(lesson.topic)} · Concept ${page.segIndex + 1} of ${page.segCount}`
        : `${_topicLabel(lesson.topic)} · Lesson`,
      stage: "lesson",
    });
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
    // The dock is a flex child of .practice-left only while the lesson owns
    // that column; leaving `dd-lesson-feedback-open` set would lay the
    // question screen out as two columns.
    if (window.DDFeedbackPanel) {
      window.DDFeedbackPanel.setLessonContext(null);
      window.DDFeedbackPanel.closeLesson();
    }
    activeQuestion = null;
    const editor = _el("code-editor");
    if (window.DeltaNotebook) window.DeltaNotebook.reset(DEFAULT_EDITOR, { addScratch: false });
    else if (editor) editor.value = DEFAULT_EDITOR;
    const out = _el("output-area");
    if (out) out.textContent = "";
  };

  const maybeShow = async (question, onDone, forceKcs = null) => {
    try {
      // Content first: a local-mode step is read out of the KP's own segment
      // list, so `_pendingSteps` cannot answer before the lessons have loaded.
      await _ensureLessons();
      if (!lessonsData) return false;
      const steps = forceKcs
        ? forceKcs.map((kc) => _stepFor(kc, _localExposure()))
        : await _pendingSteps(question);
      if (!steps.length) return false;
      const pages = _buildPages(steps);
      if (!pages.length) return false;

      const questionText = _el("question-text");
      const questionNumber = _el("question-number");
      const editor = _el("code-editor");
      const output = _el("output-area");
      if (!questionText || !editor) return false;

      document.body.classList.add("lesson-mode");

      /* 🔴 THE GATE OWNS THE SCREEN IT RENDERS INTO, and until 2026-08-27 it
         did not. Everything below draws into `#question-text`, which lives
         inside `.practice-split` — and `styles/practice/timer.css` sets
         `#page-practice.session-idle .practice-split { display: none }`. So a
         gate that fired while the practice page was IDLE rendered a whole
         lesson into a display:none box and returned TRUE, telling its caller
         the learner was reading something. They were looking at the idle dial.

         `resume()` is exactly that caller. It restores the saved question,
         asks the gate, and hands `_resumeCore` over as `onDone` — and
         `_resumeCore` is the only thing that takes `session-idle` off. So the
         one Continue button on the idle screen led to a screen that never
         changed, with a fully rendered lesson behind it and no way back into
         the session but a reload. Seth, 2026-08-27: "once I pressed the button
         to continue practice, the top bar completely disappears".

         Cleared HERE rather than in `resume()` because it is true of every
         caller: a lesson on screen is a screen, whoever asked for it. The two
         paths that already clear it (timer.js `start()`, and the `?lesson=`
         bootstrap at the foot of this file) are unaffected — removing a class
         that is not there is not an error.

         🔴 AND ONLY THIS CLASS. The gate does not switch tabs and does not
         unhide the page: `#page-practice.hidden` means the learner is reading
         something else, and yanking them out of it is not the gate's call.
         This is the point of no return — every path above it has already
         returned false — so the screen is opened exactly when a lesson is
         actually about to be drawn on it. */
      _el("page-practice")?.classList.remove("session-idle");

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
        // Reading a worked example through to the end is the one non-answer
        // thing the XP seam credits — it is the gate the learner cannot skip,
        // and finishing it should move the bar like anything else they did.
        window.dispatchEvent(new CustomEvent("delta:xp", { detail: { kind: "lesson_read" } }));
        _cleanup();
        onDone();
      };

      // The notebook index arrives over the network and a lesson can render
      // before it lands, which reads as "no notebook" and draws the full
      // in-panel lesson the Colab edition is trying not to draw. Re-render once
      // when it settles. Guarded so the re-render cannot re-arm itself, and
      // only ever fires while the same page is still on screen.
      let awaitingIndex = false;
      const _redrawWhenColabIndexLands = (page) => {
        const dd = window.DDColab;
        if (awaitingIndex || !dd || typeof dd.active !== "function" || !dd.active()) return;
        if (typeof dd.whenReady !== "function") return;
        awaitingIndex = true;
        dd.whenReady(() => {
          if (finished || pages[index] !== page) return;
          if (_colabLessonHref(page)) showPage();
        });
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
        // On the Colab edition the panel holds a rail, not cells — there is
        // nothing to mount and no worked example to load into an editor the
        // learner is not looking at. Steer the notebook to this concept the
        // same way the question side steers it (ui.js), so the reading is
        // already on screen beside them rather than one click away.
        const colabHref = _colabLessonHref(page);
        if (colabHref) {
          if (window.DDColab && typeof window.DDColab.openNotebook === "function") {
            window.DDColab.openNotebook(colabHref);
          }
        } else if (window.LessonNotebook) {
          // The KP + concept index is the kernel's session key: cells share a
          // namespace within a page, and moving to the next concept starts a
          // clean one.
          window.LessonNotebook.mount(questionText, `${page.kp.kc}#${page.segIndex}`);
          _redrawWhenColabIndexLands(page);
        }
        // Name the concept the feedback panel is about. A lesson is several
        // pages, and a note written on concept 2 must never be filed against
        // concept 3 — setLessonContext clears a half-written note when the
        // subject changes, which is the only correct thing to do with it.
        if (window.DDFeedbackPanel) {
          window.DDFeedbackPanel.setLessonContext({
            kc: page.kp.kc,
            title: page.seg.title || page.kp.title,
            questionId: question ? question.question_id : null,
          });
        }
        questionText.scrollTop = 0;
        window.scrollTo({ top: 0 });
        const lessonCode = colabHref
          ? DEFAULT_EDITOR
          : (page.seg.worked_example_code ||
            _firstPythonFence(page.seg.worked_example_markdown) || DEFAULT_EDITOR);
        if (window.DeltaNotebook) window.DeltaNotebook.reset(lessonCode, { addScratch: false });
        else editor.value = lessonCode;
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
          // The concept just read, and — only on the last one — the KP itself.
          // Writing the KC's key early is the one unrecoverable mistake here:
          // it says "taught" about concepts the learner has not seen, and the
          // gate never fires for them again.
          const keys = [page.step.exposureKey];
          if (page.lastOfKp && page.step.exposureKey !== page.kp.kc) keys.push(page.kp.kc);
          _markLocalExposure(keys);
          _markBackendExposure(keys);
          // Also credits the ladder's `worked` rung — but in finishAll, not
          // here, because the response re-stages the pending question and
          // that has to land before it renders. Without any crediting at
          // all, the gate would teach a KP and the ladder would immediately
          // re-teach the identical page: the two counters are written by
          // different endpoints and neither one implies the other.
          //
          // Credited on EVERY concept page, not just the last. Each one is a
          // worked example, and the drill waiting behind this page is the one
          // for THIS concept — it has to be re-staged onto the faded rung now
          // or the loop hands over a blank editor and calls it fading.
          taught.push(page.kp.kc);
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
