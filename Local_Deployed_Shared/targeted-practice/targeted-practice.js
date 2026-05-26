/* ================================================================
   TARGETED PRACTICE — search + select ARENA exercises to drill

   Two-mode UI inside the same #page-targeted-practice slot:

     1. SEARCH mode (default)
        - Search card on top, selected card below.
        - Each selected item shows a readiness bar (same .stats-bar
          style as the Predicted-scores table). Readiness comes from
          window.computeArenaReadiness against the matching ARENA
          problem (looked up by notebookPath in ARENA_STAGE1_PROBLEMS).
        - Submit kicks off REVIEW mode.

     2. REVIEW mode (post-submit)
        - Search card is hidden.
        - Each selected item bar animates from the snapshotted
          "before" readiness → a synthetic "after" readiness, with the
          same blue-fill + green/red delta visuals as the arena-unlock
          interstitial. The "after" is a placeholder bump for now —
          there is no real backend that grades a targeted practice
          session, so we synthesize a plausible gain so the layout is
          reviewable end-to-end.
        - Per item: an Open-in-Colab anchor (always). When the after-
          score crosses the Ready threshold, a "Ready ✓" badge appears
          to highlight which exercises the student is now cleared for.
        - At the bottom: a Back-to-search button that fully resets
          (drops the selection, returns to search mode).

   No persistence — the selected list and the before/after snapshots
   are in-memory only. When the real backend "start a focused queue"
   endpoint lands, replace the synthetic "after" with the real
   post-session readiness scores.

   Globals it expects:
     - window.ARENA_EXERCISES_BY_NOTEBOOK   (arena/exercises.js)
     - window.ARENA_PREREQS_TEMP_EXERCISES  (stats/predicted-prereqs-temp.js)
     - window.ARENA_PREREQS_TEMP_NOTEBOOK_PATH
     - window.ARENA_STAGE1_PROBLEMS         (arena/manifest.js)
     - window.computeArenaReadiness         (arena/manifest.js)
     - colabUpstreamHref                    (stats/predicted-links.js)
   ================================================================ */

(function () {
  const root = document.getElementById("page-targeted-practice");
  if (!root) return;

  const searchCard = root.querySelector(".tp-search-card");
  const searchInput = document.getElementById("tp-search-input");
  const clearBtn = document.getElementById("tp-search-clear");
  const resultsList = document.getElementById("tp-results-list");
  const resultsHint = document.getElementById("tp-results-hint");
  const selectedTitleEl = document.getElementById("tp-selected-title");
  const selectedList = document.getElementById("tp-selected-list");
  const selectedCountEl = document.getElementById("tp-selected-count");
  const selectedEmptyEl = document.getElementById("tp-selected-empty");
  const submitBtn = document.getElementById("tp-submit-btn");
  const backBtn = document.getElementById("tp-back-btn");
  const banner = document.getElementById("tp-banner");
  const bannerMeta = document.getElementById("tp-banner-meta");
  const bannerEndBtn = document.getElementById("tp-banner-end");

  // Ready threshold — exercises whose synthetic after-score lands ≥ this
  // get the "Ready ✓" badge. Matches the rough cutoff `labelForScore`
  // uses elsewhere ("Likely ready" ≥ 70).
  const READY_THRESHOLD = 70;

  // ---- Catalog (search source) ----

  // Parse the notebook path into a short human label.
  // e.g. content/.../chapter0_fundamentals/exercises/part1_ray_tracing/0.1_Ray_Tracing_exercises.ipynb
  //   →  "0.1 Ray Tracing"
  const labelFromNotebookPath = (path) => {
    const file = String(path).split("/").pop() || "";
    const stem = file.replace(/_exercises?\.ipynb$/i, "").replace(/\.ipynb$/i, "");
    const m = stem.match(/^(\d+\.\d+)[_\s]+(.+)$/);
    if (m) return `${m[1]} ${m[2].replace(/_/g, " ")}`;
    return stem.replace(/_/g, " ");
  };

  // Stable id per exercise — used as the dedupe key in the selected list.
  const exerciseId = (notebookPath, anchor, title) =>
    `${notebookPath}#${anchor || title}`;

  const buildCatalog = () => {
    const reg = window.ARENA_EXERCISES_BY_NOTEBOOK || {};
    const tempExercises = Array.isArray(window.ARENA_PREREQS_TEMP_EXERCISES)
      ? window.ARENA_PREREQS_TEMP_EXERCISES
      : [];
    const tempNotebookPath = window.ARENA_PREREQS_TEMP_NOTEBOOK_PATH || "";
    const out = [];
    for (const [notebookPath, exercises] of Object.entries(reg)) {
      const sub = labelFromNotebookPath(notebookPath);
      if (Array.isArray(exercises) && exercises.length > 0) {
        for (const ex of exercises) {
          out.push({ id: exerciseId(notebookPath, ex.anchor, ex.title), title: ex.title, sub, notebookPath, anchor: ex.anchor || "" });
        }
        continue;
      }
      // Fallback for the 0.0 prereqs notebook (registry is [] there).
      if (notebookPath === tempNotebookPath && tempExercises.length > 0) {
        for (const ex of tempExercises) {
          out.push({ id: exerciseId(notebookPath, ex.anchor, ex.title), title: ex.title, sub, notebookPath, anchor: ex.anchor || "" });
        }
      }
    }
    // Procedural drills (window.DRILLS_CATALOG) ride in on the same surface.
    // Each drill carries its own subtopics + targetSeconds so the launch path
    // (ArenaUnlock.showFor below) can POST to the new arena-rating EWMA flow
    // without falling through to the legacy ARENA_PREREQS_TEMP map.
    const drills = Array.isArray(window.DRILLS_CATALOG) ? window.DRILLS_CATALOG : [];
    for (const d of drills) {
      out.push({
        id: d.id,
        title: d.title,
        sub: d.sub,
        notebookPath: d.notebookPath,
        anchor: "",
        subtopics: d.subtopics,
        targetSeconds: d.targetSeconds,
        isDrill: true,
        isComposite: d.isComposite === true,
        compositeAtomIds: d.compositeAtomIds || null,
        arenaPart: d.arenaPart || null,
      });
    }
    return out;
  };

  let _catalog = null;
  const catalog = () => (_catalog ||= buildCatalog());

  // ---- Readiness lookup (per ARENA problem, shared by all exercises
  //      in the same notebook) ----

  let _problemsByPath = null;
  const problemsByPath = () => {
    if (_problemsByPath) return _problemsByPath;
    const map = new Map();
    (window.ARENA_STAGE1_PROBLEMS || []).forEach((p) => {
      if (p && p.notebookPath) map.set(p.notebookPath, p);
    });
    _problemsByPath = map;
    return map;
  };

  const readinessForNotebook = (notebookPath) => {
    const problem = problemsByPath().get(notebookPath);
    if (!problem) return null;
    const fallback = Number(problem.readinessScore) || 0;
    if (typeof window.computeArenaReadiness !== "function") return fallback;
    const s = window.computeArenaReadiness(problem.skillWeights, fallback);
    return Number.isFinite(s) ? s : fallback;
  };

  const colabHrefFor = (notebookPath) => {
    if (typeof colabUpstreamHref === "function") return colabUpstreamHref(notebookPath);
    return "#";
  };

  // ---- State ----

  // Selected items: Map<id, {id, title, sub, notebookPath, anchor}>.
  const selected = new Map();
  // Snapshots taken at Submit time so the bar animation has an "old"
  // value even if the catalog readiness recomputes mid-animation.
  // Map<id, number 0-100>.
  const beforeReadiness = new Map();
  // Synthetic after-scores produced by simulateAfter() at Submit time.
  // Stored so re-render in review mode keeps the same numbers.
  const afterReadiness = new Map();

  let isReviewMode = false;
  // sessionActive = student clicked Submit (banner up, practice tab active).
  // Distinct from isReviewMode (which only kicks in after End-targeted-practice
  // takes them back here and runs the bar animation).
  let sessionActive = false;

  // Synthetic "after the student practiced this" readiness — bigger
  // jump when starting lower, with a tiny jitter so two bars in a row
  // don't move in lockstep. Real backend will replace this once the
  // queue endpoint lands.
  const simulateAfter = (before) => {
    const base = Number.isFinite(before) ? before : 0;
    const gap = Math.max(0, 100 - base);
    const gain = Math.max(8, gap * 0.30) + (Math.random() * 6 - 3);
    return Math.max(0, Math.min(100, Math.round((base + gain) * 10) / 10));
  };

  // ---- Search rendering ----

  const escapeHtml = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const highlight = (text, needle) => {
    if (!needle) return escapeHtml(text);
    const esc = escapeHtml(text);
    const re = new RegExp(needle.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "ig");
    return esc.replace(re, (m) => `<mark>${m}</mark>`);
  };

  const renderResults = (query) => {
    const q = query.trim();
    resultsList.innerHTML = "";
    if (q.length < 2) {
      resultsHint.textContent = "Type at least 2 characters to search.";
      resultsHint.style.display = "";
      return;
    }
    const needle = q.toLowerCase();
    const matches = catalog()
      .filter((ex) => ex.title.toLowerCase().includes(needle))
      .slice(0, 40);
    if (matches.length === 0) {
      resultsHint.textContent = `No exercises matched "${q}".`;
      resultsHint.style.display = "";
      return;
    }
    resultsHint.style.display = "none";
    const frag = document.createDocumentFragment();
    for (const ex of matches) {
      const li = document.createElement("li");
      li.className = "tp-result-item";
      if (selected.has(ex.id)) li.classList.add("tp-result-selected");
      li.dataset.id = ex.id;
      li.innerHTML = `
        <input type="checkbox" class="tp-result-checkbox" ${selected.has(ex.id) ? "checked" : ""} aria-label="Add to practice queue" />
        <div class="tp-result-text">
          <div class="tp-result-title">${highlight(ex.title, q)}</div>
          <div class="tp-result-sub">${escapeHtml(ex.sub)}</div>
        </div>
      `;
      // Click anywhere on the row toggles selection.
      li.addEventListener("click", (e) => {
        if (e.target.classList.contains("tp-result-checkbox")) return;
        toggleSelect(ex);
      });
      li.querySelector(".tp-result-checkbox").addEventListener("click", (e) => {
        e.stopPropagation();
        toggleSelect(ex);
      });
      frag.appendChild(li);
    }
    resultsList.appendChild(frag);
  };

  // ---- Selected-list rendering ----

  // One readiness bar markup used by both render modes. In search mode
  // the bar is static (just a width). In review mode the bar starts at
  // `width:0%` and animateRow() drives it to the after value.
  const renderBarRow = (pct) => {
    const v = Number.isFinite(pct) ? pct : 0;
    return `
      <div class="tp-bar-row stats-bar">
        <div class="stats-bar-track tp-bar-track">
          <div class="stats-bar-fill tp-bar-fill" style="width:${v}%"></div>
          <div class="target-difficulty-delta tp-bar-delta hidden"></div>
          <div class="target-difficulty-marker tp-bar-marker-old hidden"><div class="target-difficulty-line"></div></div>
          <div class="target-difficulty-marker tp-bar-marker-new hidden"><div class="target-difficulty-line"></div></div>
        </div>
        <span class="stats-bar-value tp-bar-value">${Number.isFinite(pct) ? `${v.toFixed(0)}%` : "—"}</span>
      </div>
    `;
  };

  const renderSelectedItem = (ex) => {
    const li = document.createElement("li");
    li.className = "tp-selected-item";
    li.dataset.id = ex.id;

    const before = beforeReadiness.has(ex.id) ? beforeReadiness.get(ex.id) : readinessForNotebook(ex.notebookPath);
    const after = afterReadiness.get(ex.id);
    const isReady = isReviewMode && Number.isFinite(after) && after >= READY_THRESHOLD;
    if (isReady) li.classList.add("tp-selected-ready");

    // Search mode → static current readiness. Review mode → start the
    // bar at `before` and let animateRow walk it to `after`.
    const initialPct = isReviewMode ? before : readinessForNotebook(ex.notebookPath);

    const removeBtn = isReviewMode
      ? ""
      : `<button type="button" class="tp-selected-remove" title="Remove from list">Remove</button>`;

    const actions = isReviewMode
      ? `
        <div class="tp-selected-item-actions">
          ${isReady ? `<span class="tp-ready-badge">Ready ✓</span>` : `<span class="tp-not-ready-hint">Below ready threshold — practice anyway to feed the mastery model.</span>`}
          <button type="button" class="primary tp-action-practice">Practice this problem ▶</button>
        </div>
      `
      : "";

    li.innerHTML = `
      <div class="tp-selected-item-row">
        <div class="tp-selected-item-text">
          <div class="tp-selected-item-title">${escapeHtml(ex.title)}</div>
          <div class="tp-selected-item-sub">${escapeHtml(ex.sub)}</div>
        </div>
        ${removeBtn}
      </div>
      ${renderBarRow(initialPct)}
      ${actions}
    `;

    if (!isReviewMode) {
      li.querySelector(".tp-selected-remove").addEventListener("click", () => {
        selected.delete(ex.id);
        renderSelected();
        const row = resultsList.querySelector(`.tp-result-item[data-id="${CSS.escape(ex.id)}"]`);
        if (row) {
          row.classList.remove("tp-result-selected");
          const cb = row.querySelector(".tp-result-checkbox");
          if (cb) cb.checked = false;
        }
      });
    } else {
      // "Practice this problem" → hand off to the ArenaUnlock interstitial
      // on the Practice tab so the student gets the timer, hint/answer
      // scaffolding, Colab button, and 4-option self-rating that feeds the
      // mastery pipeline. Continue on the interstitial returns here.
      const practiceBtn = li.querySelector(".tp-action-practice");
      if (practiceBtn) {
        practiceBtn.addEventListener("click", () => {
          if (!window.ArenaUnlock || typeof window.ArenaUnlock.showFor !== "function") {
            console.warn("[targeted-practice] ArenaUnlock.showFor unavailable");
            return;
          }
          window.ArenaUnlock.showFor(
            {
              title: ex.title,
              notebookPath: ex.notebookPath,
              anchor: ex.anchor,
              // Procedural drills: pass through subtopics + targetSeconds + isDrill (+ composite metadata)
              // so the card hits the new arena-rating EWMA pipeline against the
              // drill's atom subtopic, and the timer/Cleared line render correctly.
              ...(ex.isDrill ? { subtopics: ex.subtopics, targetSeconds: ex.targetSeconds, isDrill: true, isComposite: ex.isComposite, compositeAtomIds: ex.compositeAtomIds, arenaPart: ex.arenaPart } : {}),
            },
            () => {
              if (typeof switchTab === "function") switchTab("targeted-practice");
            }
          );
        });
      }
    }

    return li;
  };

  const renderSelected = () => {
    selectedList.innerHTML = "";
    selectedCountEl.textContent = String(selected.size);
    selectedEmptyEl.classList.toggle("hidden", selected.size > 0);
    if (!isReviewMode) {
      submitBtn.disabled = selected.size === 0;
    }
    if (selected.size === 0) return;
    const frag = document.createDocumentFragment();
    for (const ex of selected.values()) {
      frag.appendChild(renderSelectedItem(ex));
    }
    selectedList.appendChild(frag);
  };

  const toggleSelect = (ex) => {
    if (isReviewMode) return; // selection is locked after Submit
    if (selected.has(ex.id)) selected.delete(ex.id);
    else selected.set(ex.id, ex);
    renderSelected();
    const row = resultsList.querySelector(`.tp-result-item[data-id="${CSS.escape(ex.id)}"]`);
    if (row) {
      const isSel = selected.has(ex.id);
      row.classList.toggle("tp-result-selected", isSel);
      const cb = row.querySelector(".tp-result-checkbox");
      if (cb) cb.checked = isSel;
    }
  };

  // ---- Review-mode animation ----

  // Parameterized port of practice/arena-unlock.js#_animateBarRow.
  // Walks `fill.width`, `markerNew.left`, and the up/down delta bar from
  // `oldPct` to `newPct` over `duration` ms.
  const animateRow = (li, oldPct, newPct) => {
    const fill = li.querySelector(".tp-bar-fill");
    const delta = li.querySelector(".tp-bar-delta");
    const markerOld = li.querySelector(".tp-bar-marker-old");
    const markerNew = li.querySelector(".tp-bar-marker-new");
    const valueEl = li.querySelector(".tp-bar-value");
    if (!fill || !delta || !markerOld || !markerNew || !valueEl) return;

    const o = Math.max(0, Math.min(100, Number.isFinite(oldPct) ? oldPct : 0));
    const n = Math.max(0, Math.min(100, Number.isFinite(newPct) ? newPct : o));
    const isFlat = Math.abs(n - o) < 0.05;
    const isUp = n > o;

    fill.style.width = o + "%";
    markerOld.style.left = o + "%";
    markerOld.classList.remove("hidden");
    markerNew.style.left = o + "%";
    markerNew.classList.remove("hidden");
    delta.classList.toggle("up", isUp && !isFlat);
    delta.classList.toggle("down", !isUp && !isFlat);
    if (!isFlat) delta.classList.remove("hidden");
    valueEl.textContent = `${o.toFixed(0)}% → ${o.toFixed(0)}%`;

    const start = performance.now();
    const duration = 900;
    const tick = (now) => {
      const progress = Math.min((now - start) / duration, 1);
      const v = o + (n - o) * progress;
      fill.style.width = v + "%";
      markerNew.style.left = v + "%";
      const left = Math.min(o, v);
      const width = Math.abs(v - o);
      delta.style.left = left + "%";
      delta.style.width = width + "%";
      valueEl.textContent = `${o.toFixed(0)}% → ${v.toFixed(0)}%`;
      if (progress < 1) {
        requestAnimationFrame(tick);
        return;
      }
      fill.style.width = n + "%";
      markerNew.style.left = n + "%";
      valueEl.textContent = `${o.toFixed(0)}% → ${n.toFixed(0)}%`;
      if (isFlat) {
        delta.classList.add("hidden");
        delta.style.width = "0%";
      }
    };
    requestAnimationFrame(tick);
  };

  const enterReviewMode = () => {
    // beforeReadiness was snapshotted at startSession() so it reflects
    // pre-practice scores. If we got here without a session (e.g. legacy
    // path), fall back to the current readiness as "before".
    afterReadiness.clear();
    selected.forEach((ex, id) => {
      if (!beforeReadiness.has(id)) {
        beforeReadiness.set(id, readinessForNotebook(ex.notebookPath) ?? 0);
      }
      afterReadiness.set(id, simulateAfter(beforeReadiness.get(id)));
    });

    isReviewMode = true;
    root.classList.add("tp-review-mode");
    if (searchCard) searchCard.classList.add("hidden");
    selectedTitleEl.textContent = "Practice results";
    submitBtn.classList.add("hidden");
    backBtn.classList.remove("hidden");
    renderSelected();

    // Stagger the bar animations so they cascade in instead of all
    // moving at once — matches the arena-unlock vibe.
    let idx = 0;
    selectedList.querySelectorAll(".tp-selected-item").forEach((li) => {
      const id = li.dataset.id;
      const before = beforeReadiness.get(id) ?? 0;
      const after = afterReadiness.get(id) ?? before;
      setTimeout(() => animateRow(li, before, after), 120 * idx);
      idx += 1;
    });
  };

  const resetToSearch = () => {
    isReviewMode = false;
    sessionActive = false;
    if (banner) banner.classList.add("hidden");
    selected.clear();
    beforeReadiness.clear();
    afterReadiness.clear();
    root.classList.remove("tp-review-mode");
    if (searchCard) searchCard.classList.remove("hidden");
    selectedTitleEl.textContent = "Selected to practice";
    submitBtn.classList.remove("hidden");
    submitBtn.disabled = true;
    backBtn.classList.add("hidden");
    searchInput.value = "";
    renderResults("");
    renderSelected();
    searchInput.focus();
  };

  // ---- Wire up events ----

  let debounceHandle = null;
  searchInput.addEventListener("input", () => {
    if (debounceHandle != null) window.clearTimeout(debounceHandle);
    debounceHandle = window.setTimeout(() => renderResults(searchInput.value), 80);
  });

  clearBtn.addEventListener("click", () => {
    searchInput.value = "";
    renderResults("");
    searchInput.focus();
  });

  // Submit = "I'm starting a targeted practice session." Snapshot the
  // before-scores now (so they reflect readiness at session start, not
  // at end-of-session when the cache may have shifted), show the global
  // banner, and jump to the regular Practice tab — that's where the
  // actual drilling happens. The score-increase animation lives behind
  // the banner's End-targeted-practice button.
  const startSession = () => {
    if (selected.size === 0) return;
    sessionActive = true;
    beforeReadiness.clear();
    selected.forEach((ex, id) => {
      beforeReadiness.set(id, readinessForNotebook(ex.notebookPath) ?? 0);
    });
    if (bannerMeta) {
      const n = selected.size;
      bannerMeta.textContent = `— drilling ${n} exercise${n === 1 ? "" : "s"}`;
    }
    if (banner) banner.classList.remove("hidden");
    if (typeof switchTab === "function") switchTab("practice");

    // Actually launch the selected items via ArenaUnlock.showFor. Without
    // this, the Practice tab would keep whatever drill the recommender
    // last auto-fired — so the "Submit to start practicing" button would
    // appear to do nothing (or open the wrong drill).
    const queue = Array.from(selected.values());
    const launchNext = () => {
      const ex = queue.shift();
      if (!ex) return;
      if (!window.ArenaUnlock || typeof window.ArenaUnlock.showFor !== "function") {
        console.warn("[targeted-practice] ArenaUnlock.showFor unavailable; cannot launch");
        return;
      }
      window.ArenaUnlock.showFor(
        {
          title: ex.title,
          notebookPath: ex.notebookPath,
          anchor: ex.anchor,
          // Procedural drills: pass through subtopics + targetSeconds + isDrill
          // (+ composite metadata) so the card hits the arena-rating EWMA
          // pipeline and renders the composite banner when applicable.
          ...(ex.isDrill ? { subtopics: ex.subtopics, targetSeconds: ex.targetSeconds, isDrill: true, isComposite: ex.isComposite, compositeAtomIds: ex.compositeAtomIds, arenaPart: ex.arenaPart } : {}),
        },
        launchNext
      );
    };
    launchNext();
  };

  // End-targeted-practice (banner button) = "I'm done practicing in real
  // life." Hide the banner, swap back to the Targeted Practice tab, then
  // run the before→after animation on the selected list.
  const endSession = () => {
    sessionActive = false;
    if (banner) banner.classList.add("hidden");
    if (typeof switchTab === "function") switchTab("targeted-practice");
    enterReviewMode();
  };

  submitBtn.addEventListener("click", () => {
    if (isReviewMode || selected.size === 0) return;
    startSession();
  });

  if (bannerEndBtn) bannerEndBtn.addEventListener("click", endSession);

  backBtn.addEventListener("click", () => {
    resetToSearch();
  });

  renderSelected();
})();
