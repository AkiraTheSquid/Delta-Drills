/* INSTRUCTOR REVIEW — what instructor feedback mode actually SHOWS.
 *
 * One page (#page-instructor-review), two doors (Seth, 2026-08-24: "they
 * should have the choice to choose between providing feedback through the
 * graph, and providing feedback through the questions themselves"):
 *
 *   QUESTIONS  browse the whole bank topic → subtopic → question, each one
 *              shown WITH its canonical solution — an instructor reviews
 *              content, they don't grind drills to reach it. Flags post to
 *              the existing /api/practice/problem-feedback endpoint, so an
 *              actionable flag from an allowlisted account still queues the
 *              repair runner exactly as an in-practice flag does.
 *   GRAPH      the REAL lesson graph, full-bleed under the topbar. Not a
 *              second copy of it: `.kg-container.kg2` — the Knowledge Graph
 *              tab's live Cytoscape, its lesson pane, its learner-model dock —
 *              is MOVED in here and moved back on the way out, exactly the way
 *              concept-graph/why-graph.js hosts it for the landing page's
 *              maximise. Tap a bubble and its lesson opens beside the map (the
 *              behaviour Seth asked to keep: "the interactive one that displays
 *              the lesson"); tap the arrow between two bubbles, or arm "propose
 *              a missing edge", to flag structure. Posts to
 *              /api/practice/graph-feedback (append-only log; no automated
 *              repair behind it — edge changes reshape the unlock lattice).
 *
 *              It used to draw concept-graph/graph-viz.json instead — the old
 *              ARENA 205-atom graph of the WHOLE curriculum, a force-directed
 *              blob with no lessons behind its nodes and nothing wired to it
 *              since lesson-graph.js superseded it. Reviewing sequencing there
 *              reviewed a graph the app no longer teaches from.
 *
 * Data: questions_structured.json (the bank, canonical_solution included) is
 * fetched lazily on the first door click, never at boot — learners pay nothing
 * for a surface they never open. The graph door fetches nothing at all now;
 * lesson-graph.js owns that load and builds on first use.
 *
 * Submission plumbing mirrors practice/api.js reportProblem: best-effort via
 * window.apiFetch (app.js), and on ANY failure — offline, 401, apiFetch not
 * there — the entry falls back to a localStorage queue so nothing is lost.
 * The question flags reuse the SAME queue key practice uses
 * ("problem_feedback_queue"); graph flags get their own ("graph_feedback_
 * queue") because their shape is different.
 *
 * Leaving the mode leaves the page: instructor-mode.js dispatches
 * dd-instructor-mode-changed, and if the flag goes OFF while this page is
 * showing we click the hidden #ir-exit-goto proxy (data-goto-tab="practice")
 * — switchTab is a top-level const in app.js, deliberately not on window, and
 * proxy buttons are the sanctioned way in. */
(() => {
  "use strict";

  const page = document.getElementById("page-instructor-review");
  if (!page) return;

  const el = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");

  /* ── view switching inside the page ─────────────────────────────── */
  const views = { chooser: el("ir-chooser"), questions: el("ir-questions"), graph: el("ir-graph") };
  const backBtn = el("ir-back");
  const show = (name) => {
    /* Leaving the graph view HANDS THE KNOWLEDGE GRAPH BACK. It is not ours —
       it is borrowed out of #page-knowledge-graph — so every exit from this
       view has to go through here, or that tab renders empty. (releaseKg is
       declared below; nothing calls show() before the module finishes.) */
    if (name !== "graph") releaseKg();
    /* The graph view is `position: fixed` under the topbar, so the page it
       left behind would still scroll under it — and a wheel gesture that
       misses the canvas would scroll the chooser instead of panning the map. */
    document.body.classList.toggle("ir-kg-open", name === "graph");
    Object.entries(views).forEach(([k, v]) => v && v.classList.toggle("hidden", k !== name));
    if (backBtn) backBtn.classList.toggle("hidden", name === "chooser");
  };
  if (backBtn) backBtn.addEventListener("click", () => show("chooser"));

  /* ── shared submit plumbing ─────────────────────────────────────── */
  const queueLocally = (key, entry) => {
    try {
      const q = JSON.parse(localStorage.getItem(key) || "[]");
      q.push({ ...entry, timestamp: new Date().toISOString() });
      localStorage.setItem(key, JSON.stringify(q));
    } catch (_) { /* storage blocked — best effort only */ }
  };
  const post = async (path, body, queueKey) => {
    const fetcher = window.apiFetch;
    if (typeof fetcher === "function") {
      try {
        const res = await fetcher(path, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (res.ok) return { sent: true };
      } catch (_) { /* fall through to the queue */ }
    }
    queueLocally(queueKey, body);
    return { sent: false };
  };
  const flashStatus = (node, sent) => {
    if (!node) return;
    node.textContent = sent ? "sent ✓" : "saved locally ✓";
    node.classList.remove("hidden");
  };

  /* ══ QUESTIONS ═══════════════════════════════════════════════════ */
  const QUESTION_TAGS = [
    ["broken", "🚩 Broken"],
    ["unclear", "😕 Unclear"],
    ["wrong_image", "🖼 Wrong image"],
    ["good", "👍 Good"],
  ];
  let bank = null; // [{id, curriculum, exercise}]
  let activeTopic = null;

  const loadBank = async () => {
    if (bank) return bank;
    const res = await fetch("questions_structured.json");
    bank = await res.json();
    return bank;
  };

  const questionCard = (q) => {
    const c = q.curriculum || {};
    const ex = q.exercise || {};
    const art = document.createElement("article");
    art.className = "ir-q";
    art.innerHTML = `
      <header class="ir-q-head">
        <span class="ir-q-id">#${esc(q.id)}</span>
        <span class="ir-q-diff">${esc(c.difficulty_label || "?")} · ${esc(c.difficulty_score ?? "?")}</span>
        <span class="ir-q-sub">${esc(c.subtopic || "")}</span>
      </header>
      <div class="ir-q-text">${esc(ex.question_text || "")}</div>
      <details class="ir-q-details"><summary>Starter code</summary><pre>${esc(ex.starter_code || "—")}</pre></details>
      <details class="ir-q-details" open><summary>Canonical solution</summary><pre>${esc(ex.canonical_solution || "—")}</pre></details>
      <details class="ir-q-details"><summary>Expected output</summary><pre>${esc(ex.expected_output || "—")}</pre></details>
      <div class="ir-q-flags">
        ${QUESTION_TAGS.map(([tag, label]) => `<button type="button" class="ghost ir-flag" data-tag="${tag}">${label}</button>`).join("")}
        <span class="ir-q-status hidden"></span>
      </div>
      <textarea class="ir-q-note" data-autogrow rows="4" maxlength="5000"
        placeholder="Optional note — what's wrong, or what's good, about this problem…"></textarea>`;
    const note = art.querySelector(".ir-q-note");
    const status = art.querySelector(".ir-q-status");
    art.querySelectorAll(".ir-flag").forEach((btn) => {
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        const { sent } = await post(
          "/api/practice/problem-feedback",
          { question_id: q.id, tag: btn.dataset.tag, note: note.value.trim(), correct: null },
          "problem_feedback_queue"
        );
        btn.disabled = false;
        flashStatus(status, sent);
      });
    });
    return art;
  };

  const renderSubtopic = (subKey) => {
    const list = el("ir-qlist");
    list.innerHTML = "";
    bank
      .filter((q) => (q.curriculum || {}).subtopic_key === subKey)
      .forEach((q) => list.appendChild(questionCard(q)));
  };

  const renderTopic = (topic) => {
    activeTopic = topic;
    el("ir-topics").querySelectorAll(".ir-pill").forEach((p) =>
      p.classList.toggle("ir-pill--on", p.dataset.topic === topic));
    const subs = new Map(); // subtopic_key -> {label, n}
    bank.forEach((q) => {
      const c = q.curriculum || {};
      if (c.topic !== topic) return;
      const cur = subs.get(c.subtopic_key) || { label: c.subtopic, n: 0 };
      cur.n += 1;
      subs.set(c.subtopic_key, cur);
    });
    const box = el("ir-subtopics");
    box.innerHTML = "";
    subs.forEach(({ label, n }, key) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "ir-pill ir-pill--sub";
      b.dataset.sub = key;
      b.textContent = `${label} (${n})`;
      b.addEventListener("click", () => {
        box.querySelectorAll(".ir-pill").forEach((p) => p.classList.toggle("ir-pill--on", p === b));
        renderSubtopic(key);
      });
      box.appendChild(b);
    });
    el("ir-qlist").innerHTML =
      '<p class="ir-hint">Pick a subtopic to list its questions.</p>';
  };

  const openQuestions = async () => {
    show("questions");
    try {
      await loadBank();
    } catch (_) {
      el("ir-qlist").innerHTML = '<p class="ir-hint">Couldn’t load the question bank.</p>';
      return;
    }
    const topicsBox = el("ir-topics");
    if (!topicsBox.childElementCount) {
      const counts = new Map();
      bank.forEach((q) => {
        const t = (q.curriculum || {}).topic || "Other";
        counts.set(t, (counts.get(t) || 0) + 1);
      });
      counts.forEach((n, topic) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "ir-pill";
        b.dataset.topic = topic;
        b.textContent = `${topic} (${n})`;
        b.addEventListener("click", () => renderTopic(topic));
        topicsBox.appendChild(b);
      });
    }
    if (!activeTopic) el("ir-qlist").innerHTML = '<p class="ir-hint">Pick a topic above.</p>';
  };

  /* ══ GRAPH ═══════════════════════════════════════════════════════
     The lesson graph, hosted and EDITABLE. `.kg-container.kg2` is the
     Knowledge Graph tab's live element — one Cytoscape instance, one lesson
     pane, one learner-model dock — and it is MOVED into #ir-kg-frame and moved
     back when the instructor leaves, the way concept-graph/why-graph.js
     already borrows it for the landing page's maximise. Moving beats copying
     for the same reason it does there: every behaviour of the real graph
     arrives for free and cannot drift, because it IS the real graph.

     This file owns the BORROWING — finding the element, taking it only from
     home, putting it back, and keeping the two ends of that honest. What an
     instructor then does to the graph belongs to instructor-graph-edit.js:
     selection, the right-hand inspector, edge direction, deleting, the ✛
     handle, and the proposal ledger. The split is the same one the page
     already makes elsewhere — this file is the surface, that file is the
     interaction — and it keeps a single owner for the rule that matters most
     here, that every edit is reversible on the way out. */
  const KG_SELECTOR = ".kg-container.kg2";

  let cy = null;            // lesson-graph.js's instance, borrowed
  let kgHome = null;        // where to put the Knowledge Graph back
  const editor = () => window.DDGraphEdit || null;

  const setPanelHint = (t) => { const h = el("ir-panel-hint"); if (h) h.textContent = t; };
  const setStatus = (t) => {
    const s = el("ir-graph-status");
    if (!s) return;
    if (t) { s.textContent = t; s.classList.remove("hidden"); } else s.classList.add("hidden");
  };

  /* The toolbar's flag button names the concept the ✛ is currently on, so an
     instructor can see WHICH one a flag is about to be filed against. */
  const onFocus = (name) => {
    const btn = el("ir-flag-concept");
    if (!btn) return;
    btn.disabled = !name;
    btn.textContent = name ? `Inspect “${name}”` : "Inspect this concept";
  };

  /* lesson-graph.js builds on first use and answers null until it has, so the
     editor is attached the same way that file's own deltaFocusConceptGraphKc
     waits for it: poll briefly, then give up rather than leak a timer. */
  const attachEditor = () => {
    if (!editor()) { setStatus("The graph editor didn't load."); return; }
    if (editor().isActive()) return;
    let tries = 0;
    const tick = () => {
      // The instructor can leave inside the poll window; attaching after that
      // would arm a ✛ handle on the learner's own Knowledge Graph tab.
      if (!kgHome) return;
      const c = typeof window.deltaConceptGraphCy === "function" ? window.deltaConceptGraphCy() : null;
      if (!c) {
        if (tries++ < 80) setTimeout(tick, 120);
        else setStatus("The lesson graph didn't finish loading.");
        return;
      }
      cy = c;
      /* attach() answers false when it cannot find the overlay layer inside the
         borrowed graph — it builds the inspector and the ✛ handle in there.
         Left unchecked, the toolbar would keep its buttons and none of them
         would do anything, which is the failure that looks like a working
         screen. */
      if (!editor().attach({ cy, frame: el("ir-kg-frame"), post: postGraph, onFocus })) {
        setStatus("The graph loaded, but its editing layer didn't — reload the page.");
        return;
      }
      setStatus("");
    };
    tick();
  };

  const postGraph = (body) => post("/api/practice/graph-feedback", body, "graph_feedback_queue");

  const hostKg = () => {
    const frame = el("ir-kg-frame");
    if (!frame) return;
    /* openGraph defers this by a frame, and a frame is long enough to leave:
       back to the chooser, or straight to another tab. Releasing first is a
       no-op when nothing is hosted, so without this the stale callback would
       walk in afterwards and park the graph inside a hidden view — the
       Knowledge Graph tab then renders empty with nothing left to blame. */
    if (page.classList.contains("hidden") || views.graph.classList.contains("hidden")) return;
    if (kgHome) { if (cy) { cy.resize(); cy.fit(undefined, 36); } return; }
    const kg = document.querySelector(KG_SELECTOR);
    if (!kg) { setStatus("Couldn't find the lesson graph."); return; }
    /* why-graph.js borrows the very same element for the landing page's
       maximise, and each host remembers where it took it FROM. Two hosts at
       once and whichever releases second finds nothing to put back, so the
       Knowledge Graph tab ends up permanently empty. Only take it from home. */
    if (!kg.closest("#page-knowledge-graph")) {
      setStatus("The lesson map is open on another screen — close that one first.");
      return;
    }
    kgHome = { parent: kg.parentNode, next: kg.nextSibling };
    frame.appendChild(kg);
    frame.classList.add("is-hosting-kg");
    // Builds on first use; on later ones it just resizes and refits.
    if (typeof window.deltaInitConceptGraph === "function") window.deltaInitConceptGraph();
    setStatus("");
    attachEditor();
  };

  const releaseKg = () => {
    /* Before the early return: the scroll lock rides the VIEW, and the view can
       be open while nothing is hosted (the graph was borrowed elsewhere). */
    document.body.classList.remove("ir-kg-open");
    if (!kgHome) return;
    /* Detach FIRST. It reverts every staged proposal, so the element handed
       back is the one that was borrowed — an instructor's deleted edge must
       never reach the learner's own tab, and their ✛ handle must not either.
       The ledger itself survives, and is replayed on the way back in. */
    if (editor()) editor().detach();
    onFocus(null);
    const frame = el("ir-kg-frame");
    const kg = frame && frame.querySelector(KG_SELECTOR);
    // insertBefore(node, null) appends, which is the right answer when the
    // graph was the last child of its page.
    if (kg) kgHome.parent.insertBefore(kg, kgHome.next);
    kgHome = null;
    if (frame) frame.classList.remove("is-hosting-kg");
    /* Cytoscape caches its renderer's box, and the box it last measured was
       this frame's. Home is a different size, so the graph has to re-measure
       or it draws at the instructor screen's dimensions inside the tab.
       Only when the tab is actually on screen: a hidden container measures 0,
       and writing that is worse than leaving the stale number — app.js calls
       deltaInitConceptGraph() again when the tab next opens. Today the
       ordering saves us (this observer runs before switchTab's own refit
       frame), and that is exactly the kind of luck worth not depending on. */
    if (kg && kg.offsetParent !== null && typeof window.deltaInitConceptGraph === "function") {
      window.deltaInitConceptGraph();
    }
  };

  const openGraph = () => {
    show("graph");
    /* Short on purpose. The practice session's clock notch hangs ~25px below
       the topbar in the middle of this strip (styles/practice/notch-menu.css),
       so a sentence that reaches the centre reads under it. The full
       instructions are the inspector's empty state, which has the room. */
    setPanelHint("Tap an arrow to inspect it.");
    // Cytoscape can't size a display:none container — host only now, once the
    // pane is actually visible (the same deferral lesson-graph.js documents).
    requestAnimationFrame(hostKg);
  };

  el("ir-missing-edge").addEventListener("click", () => {
    if (editor()) editor().armMissingEdge();
  });
  el("ir-flag-concept").addEventListener("click", () => {
    if (editor()) editor().inspectFocused();
  });
  el("ir-kg-exit").addEventListener("click", () => show("chooser"));

  /* ── doors + mode exit ──────────────────────────────────────────── */
  el("ir-door-questions").addEventListener("click", openQuestions);
  el("ir-door-graph").addEventListener("click", openGraph);

  document.addEventListener("dd-instructor-mode-changed", (e) => {
    const on = !!(e.detail && e.detail.on);
    if (!on && !page.classList.contains("hidden")) {
      releaseKg();
      const exit = el("ir-exit-goto");
      if (exit) exit.click();
    }
  });

  /* app.js's switchTab hides this page by toggling `hidden` on it and fires no
     event, so the one reliable signal that the instructor navigated away is the
     class itself. Without this, opening the graph door and then clicking
     "Knowledge Graph" in the nav lands on an empty tab — the graph is still
     parked in a hidden page. Watch the box, not the events. */
  if (typeof MutationObserver === "function") {
    new MutationObserver(() => {
      if (page.classList.contains("hidden")) releaseKg();
    }).observe(page, { attributes: true, attributeFilter: ["class"] });
  }
})();
