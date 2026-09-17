/* ================================================================
   COLAB.JS — the content script. Everything that has to touch Colab's DOM.

   Three jobs, in the order the panel needs them:

     1. find a cell        — by our own `metadata.id` anchor on generated
                             notebooks, by heading text on ARENA's (which are
                             nbformat 4.2 with zero ids, so Colab regenerates
                             DOM ids on every load and anchors are useless
                             there);
     2. make it visible    — a cell inside a COLLAPSED section scrolls to the
                             section header instead, silently: the scroll
                             lands and nothing moves. Expanding the parent
                             first is what turns that no-op into a landing;
     3. read a cell's output — so the panel can pre-select a self-grade button
                             when the notebook shipped an assertion.

   Colab specifics this relies on, all verified against a live notebook:
     * the real scroll container is `colab-scroller#notebook-main`, NOT window
     * cells are `div.cell`; the active one carries `.focused`
     * section toggles are `md-icon-button.header-section-toggle`, and their
       `aria-label` reads "Expand"/"Collapse" depending on current state
   ================================================================ */

(() => {
  const SCROLLER = "colab-scroller#notebook-main";
  const CELL = "div.cell";
  const TOGGLE = "md-icon-button.header-section-toggle";

  const scroller = () => document.querySelector(SCROLLER);

  const cells = () => Array.from(document.querySelectorAll(CELL));

  /** Is `el` actually inside the visible part of the scroll container? */
  const isVisible = (el) => {
    const host = scroller();
    if (!host || !el) return false;
    const a = el.getBoundingClientRect();
    const b = host.getBoundingClientRect();
    return a.bottom > b.top && a.top < b.bottom && a.height > 0;
  };

  /**
   * Locate a cell.
   *
   * `anchor` is the nbformat `metadata.id` we generate (Colab renders it as
   * the DOM id `cell-<id>`), and is exact when present. `text` is the
   * fallback: match a cell whose rendered text contains the string. Text
   * matching is what makes the same panel work on ARENA's un-id'd notebooks.
   */
  const findCell = ({ anchor, text }) => {
    if (anchor) {
      const byId =
        document.getElementById(`cell-${anchor}`) ||
        document.getElementById(anchor);
      if (byId) return byId.closest(CELL) || byId;
      // Generated notebooks also carry the anchor as an HTML comment inside
      // the markdown source, which survives even if Colab drops the id.
      const commented = cells().find((c) =>
        (c.textContent || "").includes(`dd:${anchor}`),
      );
      if (commented) return commented;
    }
    if (text) {
      const needle = text.trim().toLowerCase();
      return (
        cells().find((c) =>
          (c.textContent || "").trim().toLowerCase().includes(needle),
        ) || null
      );
    }
    return null;
  };

  /**
   * Expand every collapsed section above `cell`.
   *
   * Walk backwards from the target through the cell list and click each
   * toggle whose label says "Expand". Walking the whole prefix rather than
   * just the nearest one matters for nested sections: an inner section can be
   * expanded while the outer one that contains it is still closed, in which
   * case expanding only the inner changes nothing on screen.
   */
  const expandAbove = (cell) => {
    const all = cells();
    const idx = all.indexOf(cell);
    if (idx < 0) return 0;
    let clicked = 0;
    for (let i = idx; i >= 0; i--) {
      const toggle = all[i].querySelector(TOGGLE);
      if (!toggle) continue;
      const label = (toggle.getAttribute("aria-label") || "").trim();
      if (/^expand/i.test(label)) {
        toggle.click();
        clicked++;
      }
    }
    return clicked;
  };

  const scrollToCell = async (cell) => {
    cell.scrollIntoView({ behavior: "smooth", block: "center" });
    // Colab animates section expansion; give it a beat before reporting back,
    // otherwise `isVisible` measures the pre-animation geometry.
    await new Promise((r) => setTimeout(r, 450));
  };

  const goto = async ({ anchor, text }) => {
    let cell = findCell({ anchor, text });
    if (!cell) return { ok: false, reason: "not-found" };

    const expanded = expandAbove(cell);
    if (expanded) {
      // Expanding re-renders the notebook, so the node we captured may be
      // detached. Re-resolve before scrolling.
      await new Promise((r) => setTimeout(r, 250));
      cell = findCell({ anchor, text }) || cell;
    }

    await scrollToCell(cell);
    if (!isVisible(cell)) {
      // One retry: a lazily-mounted editor can change heights mid-scroll.
      await scrollToCell(cell);
    }
    return { ok: true, expanded, visible: isVisible(cell) };
  };

  /**
   * Read the rendered output of a cell, for the assertion pre-select.
   *
   * Returns the text plus a coarse verdict. `errored` is the only thing we
   * treat as authoritative — a traceback means the cell did not pass. A clean
   * run is NOT treated as a pass, because most cells print something whether
   * or not the answer was right; that judgement stays with the student.
   */
  const readOutput = ({ anchor, text }) => {
    const cell = findCell({ anchor, text });
    if (!cell) return { ok: false, reason: "not-found" };
    const outputs = cell.querySelector("colab-static-output-renderer, .output");
    const body = outputs ? outputs.textContent || "" : "";
    const errored = /Traceback \(most recent call last\)|^\w*Error:|AssertionError/m.test(
      body,
    );
    return {
      ok: true,
      ran: Boolean(body.trim()),
      errored,
      text: body.slice(0, 4000),
    };
  };

  /**
   * Which notebook is this, and is it loaded enough to navigate?
   *
   * The panel asks before every jump, because the tutor picks weakest-first
   * across all subtopics and one lesson is one subtopic — so the next problem
   * routinely lives in a different notebook than the open one. Without this
   * the panel would scroll for a cell that is not here and report "not found",
   * which reads as a bug rather than as "you need the other notebook".
   *
   * nbformat metadata is not exposed to the DOM, so identity is read off the
   * two things that are: the generated first-cell anchor, and the
   * `DD_LESSON_ID` assignment in the setup cell. The second is the reliable
   * one — it is plain rendered text, so it survives Colab dropping cell ids
   * and it is visible before the whole notebook has mounted.
   */
  const identify = () => {
    const n = cells().length;
    let lessonId = null;

    const anchored = document.querySelector('[id^="cell-dd-lesson-"]');
    if (anchored) lessonId = anchored.id.slice("cell-dd-lesson-".length);

    if (!lessonId) {
      const body = document.body ? document.body.textContent || "" : "";
      const marker = body.match(/dd:dd-lesson-([A-Za-z0-9_-]+)/);
      const setup = body.match(/DD_LESSON_ID\s*=\s*["']([A-Za-z0-9_-]+)["']/);
      lessonId = (marker && marker[1]) || (setup && setup[1]) || null;
    }

    return {
      ok: true,
      // `ready` is about the DOM, not about identity: a notebook that is still
      // mounting has no cells, and reporting "wrong notebook" then would send
      // the panel into a pointless re-navigation loop.
      ready: n > 0,
      lessonId,
      // Fragmentless: Colab appends #scrollTo=… as you move around, and the
      // panel stores this to reopen the notebook later.
      url: location.href.split("#")[0],
      cells: n,
    };
  };

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (!msg || !msg.type || !msg.type.startsWith("dd:")) return undefined;
    try {
      switch (msg.type) {
        case "dd:ping":
          sendResponse({ ok: true, cells: cells().length });
          return false;
        case "dd:identify":
          sendResponse(identify());
          return false;
        case "dd:goto":
          goto(msg).then(sendResponse);
          return true; // async
        case "dd:read-output":
          sendResponse(readOutput(msg));
          return false;
        case "dd:reveal-solution":
          // Owned by colab_focus.js (it holds the cell tagging), routed here so
          // the panel has ONE message surface to talk to. Both files run in the
          // same isolated world, and this listener only fires long after they
          // have both loaded.
          sendResponse(
            window.__ddFocus
              ? window.__ddFocus.reveal(msg.problem)
              : { ok: false, reason: "focus-not-loaded" },
          );
          return false;
        default:
          sendResponse({ ok: false, reason: "unknown-message" });
          return false;
      }
    } catch (err) {
      sendResponse({ ok: false, reason: String(err) });
      return false;
    }
  });
})();
