/* course-builder/course-builder.js — "Plan it here with the AI", the second
   option under the Courses tab's "Add your own course" row (Seth, 2026-09-29).

   #page-course-builder, a page with no tab of its own (courses.js opens it),
   split in two:
     LEFT   a ChatGPT-style chat (builder-chat.js) with concepts attached
            from the graph;
     RIGHT  the course's concept graph (builder-graph.js) — the Knowledge
            Graph tab's map and nothing else — or, after "View lesson" on a
            node's card, that concept's lesson, with a way back to the graph.
   A divider between them drags to trade width, the same gesture as the ARENA
   notebook's contents rail (practice/arena-notebook-nav.js: pointer capture,
   arrow keys, double-click resets), and the width is remembered. The top
   right of the right half maximises it over the whole page; the same button
   then brings the chat back.

   Built on the first time the page is shown, whatever showed it — the
   MutationObserver below watches the page's `hidden` class, so a reload that
   restores this tab builds it too, not only the Courses button. */
(function () {
  "use strict";

  const SPLIT_KEY = "dd_cb_split";
  const COURSE_KEY = "dd_cb_course";
  const DEFAULT_SPLIT = 40;               // % of the width the chat takes
  const MIN_SPLIT = 22, MAX_SPLIT = 72;
  const ALL = "";

  let page = null;
  let shell = null;
  let built = false;
  let course = "arena";
  let view = "graph";                     // "graph" | "lesson"
  let lessonKc = null;

  const $ = (sel) => shell.querySelector(sel);
  const load = (k) => { try { return localStorage.getItem(k); } catch (_) { return null; } };
  const save = (k, v) => { try { localStorage.setItem(k, v); } catch (_) { /* private mode */ } };
  const esc = (v) => String(v == null ? "" : v).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const courses = () => (window.DeltaCourseRegistry ? window.DeltaCourseRegistry.list() : []);
  const courseLabel = () => (courses().find((c) => c.id === course) || { label: "every concept" }).label;

  const markup = () => `
    <header class="cb-top">
      <button type="button" class="cb-back">← Courses</button>
      <div class="cb-top-title">
        <span class="cb-eyebrow">Add your own course</span>
        <h1>Course builder</h1>
      </div>
      <div class="cb-top-tools">
        <label class="cb-course-pick"><span>Graph</span>
          <select class="cb-course-select" aria-label="Which course's graph to show">
            ${courses().map((c) => `<option value="${esc(c.id)}">${esc(c.label)}</option>`).join("")}
            <option value="${ALL}">All concepts</option>
          </select>
        </label>
        <button type="button" class="cb-btn cb-btn-ghost cb-new-chat">New chat</button>
      </div>
    </header>
    <div class="cb-split">
      <section class="cb-chat-pane" aria-label="AI chat">
        <div class="cb-tray" hidden></div>
        <div class="cb-chat-root"></div>
      </section>
      <div class="cb-divider" role="separator" aria-orientation="vertical" tabindex="0"
           aria-label="Resize the chat and the graph" aria-valuemin="${MIN_SPLIT}" aria-valuemax="${MAX_SPLIT}">
        <span class="cb-divider-grip" aria-hidden="true"></span>
      </div>
      <section class="cb-view-pane" data-view="graph" aria-label="Course graph">
        <div class="cb-view-bar">
          <div class="cb-crumb cb-crumb-graph">
            <span class="cb-eyebrow">Course graph</span>
            <span class="cb-crumb-name"></span>
            <span class="cb-crumb-count"></span>
          </div>
          <div class="cb-crumb cb-crumb-lesson">
            <button type="button" class="cb-btn cb-btn-ghost cb-to-graph">← Back to graph</button>
            <span class="cb-crumb-name cb-lesson-name"></span>
          </div>
          <div class="cb-view-tools">
            <button type="button" class="cb-icon-btn cb-lesson-chat" title="Add this concept to the AI chat">＋ Chat</button>
            <button type="button" class="cb-icon-btn cb-fit" title="Fit the graph to the pane">Fit</button>
            <button type="button" class="cb-icon-btn cb-full" aria-pressed="false" title="Full screen">
              <span class="cb-full-on" aria-hidden="true">⤢</span><span class="cb-full-off" aria-hidden="true">⤡</span>
              <span class="cb-full-label">Full screen</span>
            </button>
          </div>
        </div>
        <div class="cb-graph-host" tabindex="0" aria-label="Concept graph. Arrow keys step through the concepts and open each one's card; Escape closes it."><p class="cb-status">Loading the graph…</p></div>
        <article class="cb-lesson-host" hidden></article>
      </section>
    </div>`;

  /* ---------------- sizing ------------------------------------------- */
  // Fill the viewport below wherever the page starts (under the topbar and
  // the guest banner), like the Concept Chat page, so neither pane scrolls
  // the window.
  const fitPage = () => {
    if (!page || page.classList.contains("hidden")) return;
    const top = page.getBoundingClientRect().top + window.scrollY;
    page.style.height = `${Math.max(420, window.innerHeight - top)}px`;
  };
  const clamp = (v) => Math.max(MIN_SPLIT, Math.min(MAX_SPLIT, v));
  const currentSplit = () => clamp(Number(load(SPLIT_KEY)) || DEFAULT_SPLIT);
  const applySplit = (pct) => {
    const v = clamp(pct);
    shell.style.setProperty("--cb-left", `${v}%`);
    $(".cb-divider").setAttribute("aria-valuenow", String(Math.round(v)));
    return v;
  };

  const wireDivider = () => {
    const split = $(".cb-split");
    const div = $(".cb-divider");
    let dragging = false;
    let applied = currentSplit();
    const pctAt = (x) => {
      const r = split.getBoundingClientRect();
      return ((x - r.left) / r.width) * 100;
    };
    div.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      dragging = true;
      applied = currentSplit();
      div.setPointerCapture(e.pointerId);
      shell.classList.add("is-dragging");
      e.preventDefault();
    });
    div.addEventListener("pointermove", (e) => {
      if (dragging) applied = applySplit(pctAt(e.clientX));
    });
    const end = (e) => {
      if (!dragging) return;
      dragging = false;
      shell.classList.remove("is-dragging");
      if (div.hasPointerCapture(e.pointerId)) div.releasePointerCapture(e.pointerId);
      save(SPLIT_KEY, String(applied));
      window.DDBuilderGraph.resize();
    };
    div.addEventListener("pointerup", end);
    div.addEventListener("pointercancel", end);
    div.addEventListener("dblclick", () => { save(SPLIT_KEY, String(DEFAULT_SPLIT)); applySplit(DEFAULT_SPLIT); });
    div.addEventListener("keydown", (e) => {
      const step = e.key === "ArrowLeft" ? -2 : e.key === "ArrowRight" ? 2 : 0;
      if (!step) return;
      e.preventDefault();
      save(SPLIT_KEY, String(applySplit(currentSplit() + step)));
    });
  };

  /* ---------------- full screen -------------------------------------- */
  const setFull = (on) => {
    shell.classList.toggle("is-full", on);
    const b = $(".cb-full");
    b.setAttribute("aria-pressed", String(on));
    b.title = on ? "Minimise — bring the chat back" : "Full screen";
    $(".cb-full-label").textContent = on ? "Minimise" : "Full screen";
  };

  /* ---------------- graph ⇄ lesson ----------------------------------- */
  const showGraph = () => {
    view = "graph";
    lessonKc = null;
    $(".cb-view-pane").dataset.view = "graph";
    $(".cb-lesson-host").hidden = true;
    $(".cb-graph-host").hidden = false;
    window.DDBuilderGraph.resize();
    $(".cb-graph-host").focus({ preventScroll: true });
  };

  const showLesson = (id) => {
    const kp = window.DDBuilderData.lesson(id);
    if (!kp) return;
    view = "lesson";
    lessonKc = id;
    const host = $(".cb-lesson-host");
    const title = (window.DDBuilderData.kc(id) || {}).title || id;
    $(".cb-lesson-name").textContent = title;
    host.innerHTML = typeof window.deltaKcLessonHtml === "function"
      ? window.deltaKcLessonHtml(kp)
      : `<h2 class="kg2-title">${esc(title)}</h2><pre>${esc(kp.concept_markdown || "")}</pre>`;
    try { window.DeltaMath?.render?.(host); } catch (_) { /* maths stays as source */ }
    host.scrollTop = 0;
    $(".cb-view-pane").dataset.view = "lesson";
    $(".cb-graph-host").hidden = true;
    host.hidden = false;
    refreshLessonChatBtn();
  };

  const refreshLessonChatBtn = () => {
    const b = $(".cb-lesson-chat");
    const inChat = !!lessonKc && window.DDBuilderChat.attached().includes(lessonKc);
    b.disabled = inChat;
    b.textContent = inChat ? "✓ In the chat" : "＋ Chat";
  };

  /* ---------------- course ------------------------------------------- */
  const showCourse = async () => {
    $(".cb-crumb-graph .cb-crumb-name").textContent = courseLabel();
    $(".cb-crumb-count").textContent = "";
    const status = $(".cb-status");
    try {
      const ok = await window.DDBuilderGraph.show(course, window.DDBuilderChat.attached());
      if (ok === null) return; // a newer pick is loading; it owns the crumb
      if (status) status.remove();
      if (!ok) throw new Error("graph library not loaded");
      const n = window.DDBuilderGraph.count();
      $(".cb-crumb-count").textContent = `${n} concept${n === 1 ? "" : "s"}`;
    } catch (err) {
      console.warn("[course-builder] graph failed", err);
      const host = $(".cb-graph-host");
      if (!host.querySelector(".cb-status")) host.insertAdjacentHTML("afterbegin", '<p class="cb-status"></p>');
      host.querySelector(".cb-status").textContent = "Couldn't load the graph. Refresh the page to try again.";
    }
  };

  /* ---------------- build -------------------------------------------- */
  const build = async () => {
    built = true;
    shell = document.createElement("div");
    shell.className = "cb-shell";
    shell.innerHTML = markup();
    page.replaceChildren(shell);
    applySplit(currentSplit());
    wireDivider();

    const stored = load(COURSE_KEY);
    if (stored !== null && (stored === ALL || courses().some((c) => c.id === stored))) course = stored;
    const select = $(".cb-course-select");
    select.value = course;
    select.addEventListener("change", () => {
      course = select.value;
      save(COURSE_KEY, course);
      if (view !== "graph") showGraph();
      showCourse();
    });

    $(".cb-back").addEventListener("click", () => {
      setFull(false);
      if (typeof switchTab === "function") switchTab("courses");
    });
    $(".cb-new-chat").addEventListener("click", () => window.DDBuilderChat.newChat());
    $(".cb-full").addEventListener("click", () => {
      setFull(!shell.classList.contains("is-full"));
      requestAnimationFrame(() => window.DDBuilderGraph.resize());
    });
    $(".cb-fit").addEventListener("click", () => window.DDBuilderGraph.fit());
    $(".cb-to-graph").addEventListener("click", showGraph);
    $(".cb-lesson-chat").addEventListener("click", () => {
      if (!lessonKc) return;
      window.DDBuilderChat.attach(lessonKc);
      window.DDBuilderGraph.markInChat(lessonKc, true);
      refreshLessonChatBtn();
    });
    shell.addEventListener("keydown", (e) => {
      if (e.key !== "Escape" || !shell.classList.contains("is-full")) return;
      if (e.target.closest && e.target.closest(".cb-card")) return; // the card's own Esc closes it first
      setFull(false);
      window.DDBuilderGraph.resize();
    });

    // A pane that changes width (drag, full screen, window) re-measures the
    // canvas, or Cytoscape keeps drawing into its old box.
    if (typeof ResizeObserver === "function") {
      new ResizeObserver(() => window.DDBuilderGraph.resize()).observe($(".cb-graph-host"));
    }

    window.DDBuilderGraph.mount($(".cb-graph-host"), {
      addToChat: (id) => { window.DDBuilderChat.attach(id); },
      viewLesson: showLesson,
    });
    window.DDBuilderChat.mount($(".cb-chat-root"), $(".cb-tray"), {
      courseLabel: () => (course === ALL ? "every concept" : courseLabel()),
      onDetach: (id) => {
        window.DDBuilderGraph.markInChat(id, false);
        if (view === "lesson") refreshLessonChatBtn();
      },
    });

    try {
      await window.DDBuilderData.load();
    } catch (err) {
      console.warn("[course-builder] data failed", err);
    }
    await showCourse();
  };

  const onShown = () => {
    fitPage();
    if (!built) build();
    // An empty canvas means the last load failed: retry it on every reopen.
    else if (!window.DDBuilderGraph.count()) showCourse();
    else requestAnimationFrame(() => window.DDBuilderGraph.resize());
  };

  const init = () => {
    page = document.getElementById("page-course-builder");
    if (!page) return;
    window.addEventListener("resize", fitPage);
    new MutationObserver(() => { if (!page.classList.contains("hidden")) onShown(); })
      .observe(page, { attributes: true, attributeFilter: ["class"] });
    if (!page.classList.contains("hidden")) onShown();
  };

  const open = () => {
    if (typeof switchTab === "function") switchTab("course-builder");
    else if (page) { page.classList.remove("hidden"); onShown(); }
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();

  window.DDCourseBuilder = { open };
})();
