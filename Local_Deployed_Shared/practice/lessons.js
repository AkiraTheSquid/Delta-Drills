/* ================================================================
   LESSON GATE — first-encounter exposure guard (Pass 2)

   Before a practice question whose target KC the learner has never been
   taught is revealed, show the introducing knowledge-point lesson
   (concept prose → worked example → faded practice → misconceptions),
   then record the exposure and continue to the question.

   Gate sources:
     backend mode — /next-question responses carry `lesson_gate`
       (computed server-side from the user's kc_exposure state); on
       Continue we POST /api/practice/exposure.
     local/supabase mode — computed client-side from
       lessons/qmatrix_tags.json + a localStorage exposure map.

   Content always renders from the static lessons/lessons_structured.json
   (same artifact the review viewer uses). Any load/render failure
   disables the gate for the session — practice must never block on it.
   ================================================================ */

const LessonGate = (() => {
  let lessonsData = null;      // lessons_structured.json (lazy, cached)
  let qmatrix = null;          // qmatrix_tags.json (lazy, cached; local mode only)
  let loadFailed = false;
  let overlay = null;
  let previousFocus = null;

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
      // Exposure is monotone and cheap to re-earn; a lost POST just means
      // the same lesson may show once more on another device.
    }
  };

  const _fetchJson = async (path) => {
    const res = await fetch(path);
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

  // Pending KCs for this question, in gate order. Backend mode trusts the
  // server-computed lesson_gate; local mode derives it from the Q-matrix.
  const _pendingKcs = async (question) => {
    if (question?.diagnostic_active) return [];
    if (practiceMode === "backend") {
      return [...new Set((question?.lesson_gate || []).map((e) => e?.kc).filter(Boolean))];
    }
    await _ensureQmatrix();
    if (!qmatrix) return [];
    const tags = qmatrix[String(question?.question_id)];
    if (!tags?.target_kcs?.length) return [];
    const exposed = _localExposure();
    return [...new Set(tags.target_kcs.filter((kc) => !exposed[kc]))];
  };

  /* ---------- rendering (markdown subset — mirrors lessons/viewer.html) */

  const esc = (s) =>
    String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const inline = (s) =>
    esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>");

  const md = (text) => {
    if (!text) return "";
    const lines = text.split("\n");
    const out = [];
    let i = 0, list = null, para = [], quote = [];
    const flushPara = () => { if (para.length) { out.push("<p>" + inline(para.join(" ")) + "</p>"); para = []; } };
    const flushList = () => { if (list) { out.push("</" + list + ">"); list = null; } };
    const flushQuote = () => { if (quote.length) { out.push("<blockquote>" + inline(quote.join(" ")) + "</blockquote>"); quote = []; } };
    while (i < lines.length) {
      const line = lines[i];
      const fence = line.match(/^```(.*)$/);
      if (fence) {
        flushPara(); flushList(); flushQuote();
        const info = fence[1].trim();
        const buf = [];
        i++;
        while (i < lines.length && !/^```/.test(lines[i])) { buf.push(lines[i]); i++; }
        i++;
        const cls = info.includes("starter") ? ' class="lesson-starter"' : "";
        const codeHtml = "<pre" + cls + "><code>" + esc(buf.join("\n")) + "</code></pre>";
        if (info.includes("solution")) {
          out.push('<details class="lesson-solution"><summary>Show solution</summary>' + codeHtml + "</details>");
        } else {
          out.push(codeHtml);
        }
        continue;
      }
      const h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) { flushPara(); flushList(); flushQuote(); out.push("<h4>" + inline(h[2]) + "</h4>"); i++; continue; }
      if (/^>\s?/.test(line)) { flushPara(); flushList(); quote.push(line.replace(/^>\s?/, "")); i++; continue; }
      const li = line.match(/^(\s*)([-*]|\d+\.)\s+(.*)$/);
      if (li) {
        flushPara(); flushQuote();
        const kind = /\d+\./.test(li[2]) ? "ol" : "ul";
        if (list !== kind) { flushList(); out.push("<" + kind + ">"); list = kind; }
        let item = li[3]; i++;
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*]|\d+\.)\s/.test(lines[i])) {
          item += " " + lines[i].trim(); i++;
        }
        out.push("<li>" + inline(item) + "</li>");
        continue;
      }
      if (!line.trim()) { flushPara(); flushList(); flushQuote(); i++; continue; }
      flushQuote();
      para.push(line.trim());
      i++;
    }
    flushPara(); flushList(); flushQuote();
    return out.join("\n");
  };

  const _kpHtml = (lesson, kp, index, total) => {
    let html = `<div class="lesson-gate-context">New concept ${total > 1 ? `(${index + 1} of ${total}) ` : ""}· ${esc(lesson.topic)} · ${esc(lesson.title)}</div>`;
    html += `<h2 class="lesson-gate-kp-title" id="lesson-gate-title" tabindex="-1">${esc(kp.title)}</h2>`;
    html += `<h3>Concept</h3>` + md(kp.concept_markdown);
    html += `<h3>Worked example</h3>` + md(kp.worked_example_markdown);
    if (kp.faded_items?.length) {
      html += "<h3>Try it — faded practice</h3>";
      for (const f of kp.faded_items) {
        html += '<div class="lesson-gate-item">';
        if (f.prompt) html += "<p>" + inline(f.prompt) + "</p>";
        html += '<pre class="lesson-starter"><code>' + esc(f.starter_code) + "</code></pre>";
        html += '<details class="lesson-solution"><summary>Show solution</summary><pre><code>' + esc(f.solution) + "</code></pre></details>";
        html += "</div>";
      }
    }
    if (kp.misconceptions_markdown) {
      html += '<h3>Watch out</h3><div class="lesson-gate-misconceptions">' + md(kp.misconceptions_markdown) + "</div>";
    }
    return html;
  };

  const _ensureOverlay = () => {
    if (overlay) return overlay;
    overlay = document.createElement("div");
    overlay.id = "lesson-gate-overlay";
    overlay.className = "lesson-gate-overlay hidden";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-labelledby", "lesson-gate-title");
    overlay.setAttribute("aria-describedby", "lesson-gate-body");
    overlay.setAttribute("aria-hidden", "true");
    overlay.innerHTML =
      '<div class="lesson-gate-panel">' +
      '<div class="lesson-gate-body" id="lesson-gate-body" role="document"></div>' +
      '<div class="lesson-gate-actions">' +
      '<button type="button" class="primary" id="lesson-gate-continue"></button>' +
      "</div></div>";
    overlay.addEventListener("keydown", (event) => {
      if (overlay.classList.contains("hidden")) return;
      // Lesson is a required first-encounter gate. Escape must not silently
      // skip it; keep focus inside the modal instead.
      if (event.key === "Escape") {
        event.preventDefault();
        overlay.querySelector("#lesson-gate-title")?.focus({ preventScroll: true });
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = [...overlay.querySelectorAll(
        'button:not([disabled]), summary, a[href], input:not([disabled]), [tabindex]:not([tabindex="-1"])',
      )].filter((el) => !el.hidden);
      if (!focusable.length) {
        event.preventDefault();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (!focusable.includes(document.activeElement)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    });
    document.body.appendChild(overlay);
    return overlay;
  };

  const _hide = (restoreFocus = false) => {
    if (overlay) {
      overlay.classList.add("hidden");
      overlay.setAttribute("aria-hidden", "true");
    }
    document.body.classList.remove("lesson-gate-open");
    if (restoreFocus && previousFocus?.focus) {
      previousFocus.focus({ preventScroll: true });
    }
    previousFocus = null;
  };

  /**
   * Show the introducing lesson(s) for this question's unexposed target
   * KCs, if any. Returns true when the gate took over — the caller must
   * NOT render the question; `onDone` runs after the learner finishes
   * (exposure recorded). Returns false to proceed normally.
   */
  const maybeShow = async (question, onDone) => {
    try {
      const kcs = await _pendingKcs(question);
      if (!kcs.length) return false;
      await _ensureLessons();
      if (!lessonsData) return false;
      const pages = kcs.map(_findKp).filter(Boolean);
      if (!pages.length) return false;

      const el = _ensureOverlay();
      const body = el.querySelector("#lesson-gate-body");
      const btn = el.querySelector("#lesson-gate-continue");
      let index = 0;
      let advanceLocked = false;
      let finished = false;

      previousFocus = document.activeElement;

      const showPage = () => {
        const { lesson, kp } = pages[index];
        body.innerHTML = _kpHtml(lesson, kp, index, pages.length);
        btn.textContent =
          index < pages.length - 1 ? "Got it — next concept" : "Got it — start the question";
        body.scrollTop = 0;
        el.classList.remove("hidden");
        el.setAttribute("aria-hidden", "false");
        document.body.classList.add("lesson-gate-open");
        el.querySelector("#lesson-gate-title")?.focus({ preventScroll: true });
      };

      btn.onclick = () => {
        if (advanceLocked || finished) return;
        advanceLocked = true;
        btn.disabled = true;
        const kc = pages[index].kp.kc;
        _markLocalExposure([kc]);
        _markBackendExposure([kc]); // fire-and-forget; never blocks the UI
        index++;
        if (index < pages.length) {
          showPage();
          // Keep disabled across browser-generated double-click pair so one
          // physical action cannot skip the next concept.
          setTimeout(() => {
            if (finished) return;
            advanceLocked = false;
            btn.disabled = false;
          }, 300);
        } else {
          finished = true;
          _hide();
          onDone();
        }
      };

      showPage();
      return true;
    } catch (err) {
      console.warn("[lessons] gate error — continuing without lesson:", err);
      _hide(true);
      return false;
    }
  };

  return { maybeShow };
})();

window.LessonGate = LessonGate;
