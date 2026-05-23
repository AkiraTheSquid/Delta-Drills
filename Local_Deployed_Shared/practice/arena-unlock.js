/* ================================================================
   PRACTICE — ARENA UNLOCK PAGE (in-tab view swap)

   When the student clicks "Next problem", this module checks whether
   any ARENA exercise just crossed its per-subtopic prereq thresholds.
   If so, it SWAPS the practice tab's view: hides .practice-container
   (the normal question + editor layout) and reveals #arena-unlock-page
   (this module's view) in the same slot inside #page-practice.

   The header / tabs / footer stay visible the whole time because they
   live outside #page-practice. The unlock view is not a modal, popup,
   or overlay — it IS the page for the moment. Continue → swap back to
   .practice-container and load the next Delta Drills question.

   The unlock view contains:
     - Exercise title (e.g. "(1) Column-stacking")
     - "Cleared: ..." recap of the prereq subtopics that unlocked it
     - Heading code-block — exact text auto-copied to clipboard when
       the student clicks Open in Colab, so Ctrl+F+V lands them on the
       right cell inside the notebook
     - Show hint / Show answer — scaffolding buttons, real content TBD
     - Open in Colab ↗ — opens the upstream / forked ARENA notebook
     - 4-option self-rating → score deltas + Continue:
         (1) Pick one of 4 outcome buttons (each encodes {correct, feedback}
             via data-attrs — see index.html). Mapping:
               * Solved in time, no help    → correct=true,  a_lot
               * Solved in time, with hint  → correct=true,  somewhat
               * Solved, over target time   → correct=true,  not_much
               * Looked up solution         → correct=false, a_lot
         (2) Score deltas render, Continue fires the load-next callback.
     - Stuck-hint banner — appears once the timer crosses target so the
       student knows it's fine to give up and look at the solution

   Module boundaries (own files, easy to rip out when the real concept
   graph backend ships and replaces the TEMP data scaffold):
     - DOM mount: <div id="arena-unlock-page"> inside #page-practice,
       sibling of .practice-container (see index.html).
     - Styles: practice/arena-unlock.css (its own file).
     - Controller: this file.
     - Data: window.ARENA_PREREQS_TEMP_* from predicted-prereqs-temp.js.

   Public API:
     window.ArenaUnlock.tryShow(onContinue)  → Promise<bool>
       Async. Refreshes per-subtopic scores from the backend first
       (so a Submit that just nudged a subtopic over its gate fires
       the unlock on the SAME click, not next time). Returns true if
       it consumed the Next-problem click (the view is now swapped
       to the unlock page and onContinue will be called when Continue
       is clicked); false if there is nothing to unlock and the
       caller should load the next question immediately. Callers must
       `await` it.

     window.ArenaUnlock.refreshScores() → Promise<void>
       Hits /api/practice/subtopics and populates window.__arenaSubtopicsCache
       (used by getArenaPrereqSubtopicScore in backend mode where
       adaptiveStateJson is null). Idempotent / cheap.

   Loaded BEFORE practice/events.js so events.js#nextProblemBtn can
   reference window.ArenaUnlock from its click handler.
   ================================================================ */

(function () {
  const unlockPage = document.getElementById("arena-unlock-page");
  const card = document.getElementById("arena-unlock-card");
  const practiceContainer = document.querySelector(".practice-container");
  if (!unlockPage || !card || !practiceContainer) return;

  const titleEl = document.getElementById("arena-unlock-title");
  const whyEl = document.getElementById("arena-unlock-why");
  const headingEl = document.getElementById("arena-unlock-heading");
  const hintBtn = document.getElementById("arena-unlock-hint-btn");
  const answerBtn = document.getElementById("arena-unlock-answer-btn");
  const colabBtn = document.getElementById("arena-unlock-colab-btn");
  const continueBtn = document.getElementById("arena-unlock-continue-btn");
  const placeholderEl = document.getElementById("arena-unlock-placeholder");
  const stuckHintEl = document.getElementById("arena-unlock-stuck-hint");
  const stageChoice = document.getElementById("arena-unlock-stage-choice");
  const stageResult = document.getElementById("arena-unlock-stage-result");
  const choiceButtons = document.querySelectorAll(".arena-unlock-choice-btn");
  const resultListEl = document.getElementById("arena-unlock-result-list");

  let currentEx = null;
  let onContinueCallback = null;
  let beforeScores = {};          // {subtopicKey: p0to1} snapshot at showCard time

  // Match the strip used elsewhere — Jupyter Book / Colab render markdown
  // backticks as plain text, so Ctrl+F for the raw title with backticks
  // would miss. Strip them in the clipboard payload AND in the displayed
  // heading code block (so the student sees the same text Colab does).
  const stripBackticks = (text) => String(text || "").replace(/`/g, "");

  // Build the "why you're ready" recap line: list each prereq's current
  // score with its target so the student knows which gates they cleared.
  const renderWhyMet = (exTitle) => {
    if (typeof window.getArenaPrereqsForExercise !== "function") return "";
    const prereqs = window.getArenaPrereqsForExercise(exTitle);
    if (!prereqs.length) return "";
    const parts = prereqs.map((p) => {
      const sc = (typeof window.getArenaPrereqSubtopicScore === "function")
        ? window.getArenaPrereqSubtopicScore(p.topic, p.subtopic)
        : null;
      const scoreLabel = (sc == null) ? "—" : `${Math.round(sc)}%`;
      return `${p.topic}/${p.subtopic} ${scoreLabel} (≥ ${p.minPct}%)`;
    });
    return `Cleared: ${parts.join(" · ")}`;
  };

  // Pick the Colab notebook for the current unlock view. If `currentEx`
  // carries its own notebookPath (the Targeted Practice "Practice this
  // problem" entry path supplies this), use that. Otherwise fall back to
  // the temp 0.0 prereqs notebook used by the regular auto-unlock flow.
  const colabHrefForUnlock = () => {
    const path = currentEx?.notebookPath || window.ARENA_PREREQS_TEMP_NOTEBOOK_PATH;
    if (typeof colabUpstreamHref === "function" && path) return colabUpstreamHref(path);
    return "#";
  };

  // Compose the full "Topic: Subtopic" key the backend uses (matches the
  // logic in predicted-prereqs-temp.js so we hit the same cache entries).
  const _composeSubtopicKey = (topic, subtopic) => {
    const t = String(topic || "").trim();
    const s = String(subtopic || "").trim();
    if (s.startsWith(`${t}:`)) return s;
    return t ? `${t}: ${s}` : s;
  };

  const _prereqKeysForExercise = (exTitle) => {
    if (typeof window.getArenaPrereqsForExercise !== "function") return [];
    const prereqs = window.getArenaPrereqsForExercise(exTitle) || [];
    return Array.from(new Set(prereqs.map((p) => _composeSubtopicKey(p.topic, p.subtopic))));
  };

  const _snapshotBeforeScores = (exTitle) => {
    const cache = window.__arenaSubtopicsCache || {};
    const out = {};
    _prereqKeysForExercise(exTitle).forEach((key) => {
      const entry = cache[key];
      out[key] = (entry && Number.isFinite(entry.p)) ? entry.p : null; // 0-1
    });
    return out;
  };

  const setStage = (name /* "choice" | "result" */) => {
    const map = { choice: stageChoice, result: stageResult };
    Object.entries(map).forEach(([k, el]) => {
      if (!el) return;
      el.classList.toggle("hidden", k !== name);
    });
  };

  const showCard = (ex) => {
    currentEx = ex;
    beforeScores = _snapshotBeforeScores(ex.title);
    const clean = stripBackticks(ex.title);
    titleEl.textContent = ex.title;
    headingEl.textContent = clean;
    whyEl.textContent = renderWhyMet(ex.title);
    colabBtn.href = colabHrefForUnlock();
    colabBtn.setAttribute("data-copy-key", clean);
    placeholderEl.classList.add("hidden");
    placeholderEl.textContent = "";
    if (stuckHintEl) stuckHintEl.classList.add("hidden");
    if (resultListEl) resultListEl.innerHTML = "";
    setStage("choice");
    // Re-enable buttons (a previous unlock might have left them disabled).
    choiceButtons.forEach((b) => { b.disabled = false; });
    // Reset + seed the stopwatch for this exercise. targetSeconds comes
    // from the per-exercise definition in predicted-prereqs-temp.js.
    if (window.ArenaUnlockTimer?.resetForExercise) {
      window.ArenaUnlockTimer.resetForExercise(ex.targetSeconds);
    }
    // Wire the timer's one-shot over-target callback. Two behaviors:
    //   - If "auto-submit as wrong" toggle is on: click the Looked-up-solution
    //     choice immediately (skips the stuck-hint flash since the result
    //     stage is about to render anyway).
    //   - Otherwise: surface the stuck-hint banner at the top of the card.
    if (window.ArenaUnlockTimer?.onOver) {
      window.ArenaUnlockTimer.onOver(() => {
        if (window.ArenaUnlockTimer?.isAutoSubmitWrong?.()) {
          document.getElementById("arena-unlock-choice-bad")?.click();
          return;
        }
        if (stuckHintEl) stuckHintEl.classList.remove("hidden");
      });
    }
    // VIEW SWAP — hide the normal practice layout, show the unlock view
    // in the same #page-practice content slot.
    practiceContainer.classList.add("hidden");
    unlockPage.classList.remove("hidden");
    // Focus the first choice button so keyboard users can rate fast.
    setTimeout(() => choiceButtons[0]?.focus({ preventScroll: true }), 0);
  };

  // POST the student's manual rating (correct + chosen difficulty) so every
  // prereq subtopic for the completed exercise gets bumped in the adaptive
  // state. Returns the parsed `updated` array on success, [] otherwise.
  const postArenaRating = async (ex, { feedback, correct }) => {
    if (!ex || typeof apiFetch !== "function") return [];
    if (typeof authToken !== "undefined" && !authToken) return [];
    const subtopics = _prereqKeysForExercise(ex.title);
    if (!subtopics.length) return [];
    const rating = window.ArenaUnlockTimer?.getRating?.() || {};
    const body = {
      exercise_title: ex.title,
      subtopics,
      feedback,
      correct: !!correct,
      elapsed_seconds: Number.isFinite(rating.elapsedSeconds) ? rating.elapsedSeconds : null,
      target_seconds: Number.isFinite(rating.targetSeconds) ? rating.targetSeconds : null,
    };
    try {
      const res = await apiFetch("/api/practice/arena-rating", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        console.warn("[ArenaUnlock] arena-rating failed:", res.status, await res.text().catch(() => ""));
        return [];
      }
      const data = await res.json().catch(() => ({}));
      // Refresh cached scores so the next unlock check sees the bump.
      if (window.ArenaUnlock?.refreshScores) window.ArenaUnlock.refreshScores().catch(() => {});
      return Array.isArray(data?.updated) ? data.updated : [];
    } catch (err) {
      console.warn("[ArenaUnlock] arena-rating error:", err);
      return [];
    }
  };

  // Build one animated accuracy bar per prereq subtopic. Re-uses the exact same
  // DOM scaffold + CSS classes as the singleton EWMA bar in the regular practice
  // flow (.ewma-accuracy / .stats-bar-fill / .target-difficulty-delta with .up/.down),
  // so the visuals match: blue base fill, green grow on improvement, red shrink on
  // regression, white marker lines for old + new positions.
  const _buildBarRow = (subtopicKey) => {
    const shell = document.createElement("div");
    shell.className = "arena-unlock-bar-row ewma-accuracy";
    shell.innerHTML = `
      <div class="bar-header-row">
        <div class="ewma-accuracy-label"></div>
        <div class="ewma-accuracy-value">—</div>
      </div>
      <div class="stats-bar-track ewma-accuracy-track">
        <div class="stats-bar-fill ewma-accuracy-fill" style="width:0%"></div>
        <div class="target-difficulty-delta hidden"></div>
        <div class="target-difficulty-marker arena-unlock-bar-marker-old"><div class="target-difficulty-line"></div></div>
        <div class="target-difficulty-marker hidden arena-unlock-bar-marker-new"><div class="target-difficulty-line"></div></div>
      </div>
    `;
    const refs = {
      label: shell.querySelector(".ewma-accuracy-label"),
      value: shell.querySelector(".ewma-accuracy-value"),
      fill: shell.querySelector(".ewma-accuracy-fill"),
      delta: shell.querySelector(".target-difficulty-delta"),
      markerOld: shell.querySelector(".arena-unlock-bar-marker-old"),
      markerNew: shell.querySelector(".arena-unlock-bar-marker-new"),
    };
    refs.label.textContent = subtopicKey;
    return { shell, refs };
  };

  // Parameterized version of practice/bars.js#showEwmaAccuracy. Takes a refs
  // bag instead of singleton DOM globals so we can have one per subtopic.
  const _animateBarRow = (refs, pBefore, pAfter) => {
    if (!Number.isFinite(pAfter)) {
      // Backend didn't return an update for this subtopic — show whatever we
      // had pre-click (or empty) without animating.
      if (Number.isFinite(pBefore)) {
        const pct = Math.round(pBefore * 1000) / 10;
        refs.fill.style.width = pct + "%";
        refs.value.textContent = pct.toFixed(1) + "%";
        refs.markerOld.style.left = pct + "%";
      } else {
        refs.value.textContent = "—";
      }
      return;
    }
    const newPct = Math.round(pAfter * 1000) / 10;
    const oldPct = Number.isFinite(pBefore) ? Math.round(pBefore * 1000) / 10 : 0;
    const isFlat = Math.abs(newPct - oldPct) < 0.01;
    const isUp = newPct > oldPct;

    refs.fill.style.width = oldPct + "%";
    refs.markerOld.style.left = oldPct + "%";
    refs.markerNew.classList.remove("hidden");
    refs.markerNew.style.left = oldPct + "%";
    refs.delta.classList.toggle("up", isUp && !isFlat);
    refs.delta.classList.toggle("down", !isUp && !isFlat);
    if (!isFlat) refs.delta.classList.remove("hidden");
    refs.value.textContent = `Old ${oldPct.toFixed(1)}% → New ${oldPct.toFixed(1)}%`;

    const start = performance.now();
    const duration = 900;
    const tick = (now) => {
      const progress = Math.min((now - start) / duration, 1);
      const value = oldPct + (newPct - oldPct) * progress;
      refs.fill.style.width = value + "%";
      refs.markerNew.style.left = value + "%";
      const left = Math.min(oldPct, value);
      const width = Math.abs(value - oldPct);
      refs.delta.style.left = left + "%";
      refs.delta.style.width = width + "%";
      refs.value.textContent = `Old ${oldPct.toFixed(1)}% → New ${value.toFixed(1)}%`;
      if (progress < 1) {
        requestAnimationFrame(tick);
        return;
      }
      refs.fill.style.width = newPct + "%";
      refs.markerNew.style.left = newPct + "%";
      refs.value.textContent = `Old ${oldPct.toFixed(1)}% → New ${newPct.toFixed(1)}%`;
      if (isFlat) {
        refs.delta.classList.add("hidden");
        refs.delta.style.width = "0%";
      }
    };
    requestAnimationFrame(tick);
  };

  const renderResults = (beforeMap, updated) => {
    if (!resultListEl) return;
    resultListEl.innerHTML = "";
    const byKey = {};
    (updated || []).forEach((u) => { if (u && u.subtopic) byKey[u.subtopic] = u; });
    const keys = Object.keys(beforeMap);
    if (!keys.length) {
      resultListEl.textContent = "No prereq subtopics to update.";
      return;
    }
    keys.forEach((key, idx) => {
      const upd = byKey[key];
      const before = beforeMap[key]; // 0-1 or null
      const after = upd && Number.isFinite(upd.p_after) ? upd.p_after : null;
      const { shell, refs } = _buildBarRow(key);
      resultListEl.appendChild(shell);
      // Stagger slightly so bars cascade in instead of all moving in sync.
      setTimeout(() => _animateBarRow(refs, before, after), 120 * idx);
    });
  };

  const hideCard = () => {
    // VIEW SWAP back — restore the practice question UI in the same slot.
    unlockPage.classList.add("hidden");
    practiceContainer.classList.remove("hidden");
  };

  hintBtn.addEventListener("click", () => {
    placeholderEl.textContent = "Hints are not wired up yet — this button is scaffolding so the real hint content can drop in later without UI rework.";
    placeholderEl.classList.remove("hidden");
  });

  answerBtn.addEventListener("click", () => {
    placeholderEl.textContent = "Answer reveal is not wired up yet — this button is scaffolding so the real answer-reveal can drop in later without UI rework.";
    placeholderEl.classList.remove("hidden");
  });

  // Auto-copy the exercise heading to the clipboard the moment the
  // student clicks Open in Colab. Same pattern as the Predicted-scores
  // table Colab pill. Anchor still navigates in the new tab.
  colabBtn.addEventListener("click", () => {
    const text = stripBackticks(currentEx?.title || "");
    if (navigator.clipboard?.writeText) navigator.clipboard.writeText(text).catch(() => {});
  });

  // Stage 1: 4-option self-rating. Each button encodes {correct, feedback}
  // via data-attrs (see index.html). One click → POST → render deltas → Continue.
  choiceButtons.forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (!currentEx) return;
      const correct = btn.dataset.correct === "true";
      const feedback = btn.dataset.feedback;
      choiceButtons.forEach((b) => { b.disabled = true; });
      const updated = await postArenaRating(currentEx, { feedback, correct });
      renderResults(beforeScores, updated);
      setStage("result");
      setTimeout(() => continueBtn.focus({ preventScroll: true }), 0);
    });
  });

  // Stage 2: Continue → mark shown (unless this was a Targeted Practice
  // manual launch — those exercises should remain re-practicable), swap
  // back to the practice view, fire callback.
  continueBtn.addEventListener("click", () => {
    if (currentEx && !currentEx._targetedPractice && typeof window.markArenaExerciseShown === "function") {
      window.markArenaExerciseShown(currentEx.title);
    }
    hideCard();
    const cb = onContinueCallback;
    onContinueCallback = null;
    currentEx = null;
    if (typeof cb === "function") cb();
  });

  // Pulls per-subtopic scores from the backend and stashes them as a map
  // keyed by full "Topic: Subtopic" name on window.__arenaSubtopicsCache.
  // No-op (with a quiet warn) if apiFetch / authToken aren't ready —
  // happens on first load before the user signs in.
  let _refreshInFlight = null;
  const refreshScores = async () => {
    if (_refreshInFlight) return _refreshInFlight;
    _refreshInFlight = (async () => {
      try {
        if (typeof apiFetch !== "function") return;
        if (typeof authToken !== "undefined" && !authToken) return;
        const res = await apiFetch("/api/practice/subtopics");
        if (!res.ok) return;
        const items = await res.json();
        if (!Array.isArray(items)) return;
        const map = {};
        items.forEach((it) => {
          if (it && typeof it.subtopic === "string") map[it.subtopic] = it;
        });
        window.__arenaSubtopicsCache = map;
      } catch (_) { /* network errors are non-fatal — gate stays closed */ }
      finally { _refreshInFlight = null; }
    })();
    return _refreshInFlight;
  };

  // Warm the cache once the auth token shows up. The auth flow may sign
  // the user in after this script runs (e.g. token in localStorage but
  // not yet loaded), so retry a few times rather than giving up.
  const _warmCacheWhenReady = (() => {
    let tries = 0;
    const tick = () => {
      tries += 1;
      if (typeof authToken !== "undefined" && authToken) {
        refreshScores();
        return;
      }
      if (tries < 20) setTimeout(tick, 500);
    };
    tick();
  });
  _warmCacheWhenReady();

  window.ArenaUnlock = {
    refreshScores,
    async tryShow(onContinue) {
      // Refresh first so a just-submitted answer that bumped a subtopic
      // over its gate is reflected on THIS click, not the next one.
      try { await refreshScores(); } catch (_) {}
      if (typeof window.getNextUnshownUnlockedArenaExercise !== "function") return false;
      const ex = window.getNextUnshownUnlockedArenaExercise();
      if (!ex) return false;
      onContinueCallback = onContinue;
      showCard(ex);
      return true;
    },
    // Manual launch — used by the Targeted Practice "Practice this problem"
    // button. Switches the active tab to Practice (the unlock-page lives
    // inside #page-practice), then renders the unlock view for the given
    // exercise. The ex object is expected to carry at least { title,
    // notebookPath, anchor } and optionally targetSeconds. Continue fires
    // onContinue (the caller typically returns to its own tab).
    showFor(ex, onContinue) {
      if (!ex || !ex.title) return false;
      // Refresh prereq cache opportunistically — exercises that happen to
      // be in the prereq scaffold will then get a real before→after delta
      // when the student self-rates.
      refreshScores().catch(() => {});
      if (typeof switchTab === "function") switchTab("practice");
      onContinueCallback = onContinue;
      showCard({ ...ex, _targetedPractice: true });
      return true;
    },
  };
})();
