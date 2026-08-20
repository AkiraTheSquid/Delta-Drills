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
  const bannerEl = card.querySelector(".arena-unlock-banner");
  const subEl = card.querySelector(".arena-unlock-sub");
  const headingLabelEl = card.querySelector(".arena-unlock-heading-label");

  // Snapshot the default ARENA copy so we can restore it after a drill card
  // (the DOM markup in arena-unlock-dom.js still ships ARENA-specific text).
  const DEFAULT_BANNER = bannerEl?.textContent || "";
  const DEFAULT_SUB = subEl?.textContent || "";
  const DEFAULT_HEADING_LABEL = headingLabelEl?.textContent || "";

  const DRILL_BANNER = "🛠️ Hands-on drill — practice it in Colab";
  const DRILL_SUB = "A short coding exercise in a real Colab notebook. Open it, work through it, then come back and rate how it went.";
  const COMPOSITE_BANNER = "🧩 Combined drill — a few skills at once";
  const COMPOSITE_SUB = "One exercise that uses 2–3 related skills together, the way you'll need them in real ARENA tasks. Finishing it updates all of them at once.";
  const DRILL_HEADING_LABEL = "Section heading inside the notebook — copy it, then press Ctrl+F in Colab to jump straight to the right cell:";
  // Honest one-liners that DEFINE the worked/faded tiers inline (the tester
  // flagged "faded version" / "completion beacon" as undefined jargon).
  const WORKED_SUB = "A worked example — study material, not a graded problem. Read it, run each cell, and make sure you can follow the reasoning. Nothing to submit.";
  const FADED_SUB = "A faded example — most of the code is already written. Fill in the one blanked step, run it to check, then rate how it went.";
  const headingBlockEl = document.getElementById("arena-unlock-heading-block");
  const copyBtn = document.getElementById("arena-unlock-copy-btn");
  const ratingPromptEl = card.querySelector(".arena-unlock-rating-prompt");
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
  let currentTierPath = null;     // ERE: tiered notebook path overriding ex.notebookPath
  let currentTierKind = "full";   // "worked" | "faded" | "full"

  // Match the strip used elsewhere — Jupyter Book / Colab render markdown
  // backticks as plain text, so Ctrl+F for the raw title with backticks
  // would miss. Strip them in the clipboard payload AND in the displayed
  // heading code block (so the student sees the same text Colab does).
  const stripBackticks = (text) => String(text || "").replace(/`/g, "");

  // Build the "why you're ready" recap line: list each prereq's current
  // score with its target so the student knows which gates they cleared.
  // For procedural drills (ex.isDrill / ex.subtopics) we instead show
  // which subtopic(s) the drill is going to feed — drills BUILD prereqs
  // rather than consume them, so there's nothing "cleared" to recap.
  const renderWhyMet = (ex) => {
    if (ex?.isDrill && Array.isArray(ex.subtopics) && ex.subtopics.length) {
      return `Targets: ${ex.subtopics.join(" · ")}`;
    }
    if (typeof window.getArenaPrereqsForExercise !== "function") return "";
    const prereqs = window.getArenaPrereqsForExercise(ex?.title || "");
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
  // ERE adaptive tier (Expertise Reversal Effect). For a single-atom drill,
  // pick the notebook tier by the learner's current BKT posterior for the atom:
  //   mastery < 0.40 → worked example (study-only, no beacon)
  //   0.40–0.75      → faded example  (completion problem)
  //   ≥ 0.75         → full drill     (the existing ex.notebookPath)
  // Composites stay full (integration practice is inherently expert-tier).
  // Returns {path, kind} or null to keep the full drill. window.__ereTiers is
  // the per-atom manifest (ere-tiers-manifest.js).
  const _ereTierPick = (ex) => {
    const tiers = window.__ereTiers;
    if (!tiers || !ex || ex.isComposite || !ex.atomId) return null;
    const t = tiers[ex.atomId];
    if (!t) return null;
    const m = window.__atomMastery || {};
    const mastery = Number.isFinite(m[ex.atomId]) ? m[ex.atomId] : 0.10;
    let pool = null, kind = "full";
    if (mastery < 0.40 && t.worked && t.worked.length) { pool = t.worked; kind = "worked"; }
    else if (mastery < 0.75 && t.faded && t.faded.length) { pool = t.faded; kind = "faded"; }
    if (!pool || !pool.length) return null;
    // Deterministic per-exercise pick: ex1 → variant 0, ex2 → variant 1, …
    // (was a global rotating counter, which made the SAME drill click land on
    // a different tier notebook every time — so the card title/heading no
    // longer matched the notebook that opened. Tester hit exactly this.)
    // Stable indexing lets showCard label the card from the notebook we open.
    const exIdx = Number.isFinite(ex.exerciseIndex) ? ex.exerciseIndex : 1;
    const idx = (((exIdx - 1) % pool.length) + pool.length) % pool.length;
    return { path: pool[idx], kind };
  };

  // Humanize a tier-notebook filename into a card title that MATCHES what the
  // learner sees when the notebook opens (tier notebooks open at the top, so
  // there is no separate Ctrl+F target). The worked/faded notebooks are their
  // own scaffold ladder — NOT 1:1 with the full-exercise titles — so we must
  // label the card from the file we actually open, not from ex.title.
  // ".../ere/faded-03-fill-reverse-edge-dispatch.ipynb" → "Faded example 3 · fill reverse edge dispatch".
  const _tierLabelFromPath = (path, kind) => {
    const file = String(path || "").split("/").pop().replace(/\.solution\.ipynb$|\.ipynb$/i, "");
    const kindWord = kind === "worked" ? "Worked example" : kind === "faded" ? "Faded example" : "Example";
    const m = file.match(/^(worked|faded)-(\d+)-(.+)$/i);
    if (!m) return kindWord;
    const slug = m[3].replace(/-/g, " ").trim();
    return `${kindWord} ${parseInt(m[2], 10)} · ${slug}`;
  };

  // The effective notebook path for the current card: the ERE tier override if
  // one was picked in showCard, else the exercise's own path / temp prereqs nb.
  const _effPath = () =>
    currentTierPath || currentEx?.notebookPath || window.ARENA_PREREQS_TEMP_NOTEBOOK_PATH;

  const colabHrefForUnlock = () => {
    const path = _effPath();
    if (typeof colabUpstreamHref === "function" && path) return colabUpstreamHref(path);
    return "#";
  };

  // Colab href for the SOLUTION notebook of the current drill. Each drill has
  // a generated `<name>.solution.ipynb` sibling (answer filled into the stub
  // so it runs top-to-bottom); window.__drillSolutionPaths lists which drills
  // have one. Falls back to the problem notebook (it carries the collapsed
  // solution) when no generated solution exists.
  const solutionHrefForUnlock = () => {
    const path = _effPath();
    if (path && typeof colabUpstreamHref === "function") {
      // Worked tier IS already the solution — open it as-is.
      if (currentTierKind === "worked") return colabUpstreamHref(path);
      // Faded tier always has a generated `<name>.solution.ipynb` sibling.
      if (currentTierKind === "faded") {
        return colabUpstreamHref(path.replace(/\.ipynb$/, ".solution.ipynb"));
      }
      const have = window.__drillSolutionPaths;
      if (have && typeof have.has === "function" && have.has(path)) {
        return colabUpstreamHref(path.replace(/\.ipynb$/, ".solution.ipynb"));
      }
    }
    return colabHrefForUnlock();
  };

  // Compose the full "Topic: Subtopic" key the backend uses (matches the
  // logic in predicted-prereqs-temp.js so we hit the same cache entries).
  const _composeSubtopicKey = (topic, subtopic) => {
    const t = String(topic || "").trim();
    const s = String(subtopic || "").trim();
    if (s.startsWith(`${t}:`)) return s;
    return t ? `${t}: ${s}` : s;
  };

  // Resolve the subtopics the arena-rating POST should bump for this ex.
  // Drills carry their own ex.subtopics (the atom's targeted subtopic from
  // the new algo). ARENA exercises fall back to the legacy prereq map.
  const _prereqKeysForExercise = (ex) => {
    if (ex && Array.isArray(ex.subtopics) && ex.subtopics.length) {
      return Array.from(new Set(ex.subtopics.map((s) => String(s).trim()).filter(Boolean)));
    }
    const exTitle = typeof ex === "string" ? ex : ex?.title;
    if (typeof window.getArenaPrereqsForExercise !== "function") return [];
    const prereqs = window.getArenaPrereqsForExercise(exTitle) || [];
    return Array.from(new Set(prereqs.map((p) => _composeSubtopicKey(p.topic, p.subtopic))));
  };

  // Resolve the atom ids a completed exercise practiced, for the BKT update.
  // Composite drills: compositeAtomIds (full list). Single drills: atomId.
  // ARENA exercises currently carry no atom tags on the live object → [].
  const _atomIdsForExercise = (ex) => {
    if (!ex) return [];
    if (Array.isArray(ex.compositeAtomIds) && ex.compositeAtomIds.length) {
      return Array.from(new Set(ex.compositeAtomIds.map((a) => String(a).trim()).filter(Boolean)));
    }
    if (ex.atomId) return [String(ex.atomId).trim()].filter(Boolean);
    if (Array.isArray(ex.atomIds) && ex.atomIds.length) {
      return Array.from(new Set(ex.atomIds.map((a) => String(a).trim()).filter(Boolean)));
    }
    return [];
  };

  // Snapshot the per-atom BKT posteriors (0-1) for every atom this exercise
  // practices, taken BEFORE the rating POST so renderResults can animate the
  // before→after delta. p_init (0.10) when the atom has no prior. Atom-level
  // (not subtopic) because BKT mastery is the actual estimate the rating moves.
  const _snapshotBeforeAtoms = (ex) => {
    const mastery = window.__atomMastery || {};
    const out = {};
    _atomIdsForExercise(ex).forEach((atom) => {
      out[atom] = Number.isFinite(mastery[atom]) ? mastery[atom] : 0.10;
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

  // Set one self-rating button's text + meaning. `skip:true` makes it a no-rating
  // "skip" action (used for study material). `hidden:true` removes it from view.
  const _setChoice = (id, opts) => {
    const btn = document.getElementById(id);
    if (!btn) return;
    btn.classList.toggle("hidden", !!opts.hidden);
    if (opts.hidden) return;
    const mark = btn.querySelector(".arena-unlock-choice-mark");
    const main = btn.querySelector(".arena-unlock-choice-main");
    const sub = btn.querySelector(".arena-unlock-choice-sub");
    if (mark) mark.textContent = opts.mark;
    if (main) main.textContent = opts.main;
    if (sub) sub.textContent = opts.sub;
    if (opts.skip) {
      btn.dataset.skip = "true";
      btn.removeAttribute("data-correct");
      btn.removeAttribute("data-feedback");
    } else {
      delete btn.dataset.skip;
      btn.dataset.correct = String(!!opts.correct);
      btn.dataset.feedback = opts.feedback;
    }
  };

  // Reconfigure the 4 self-rating buttons for the current tier. Worked-tier
  // cards are STUDY MATERIAL, not problems to solve — so they get read-through
  // completion states ("read it — understood" / "still fuzzy" / "skip") instead
  // of "solved in target time". (How these feed mastery is the engine track's
  // call; we just emit the rating signal via the same arena-rating POST.)
  const _configureChoices = (kind) => {
    if (kind === "worked") {
      if (ratingPromptEl) ratingPromptEl.textContent = "Study material — when you've read through it:";
      _setChoice("arena-unlock-choice-best", { mark: "✓", main: "Read it — understood", sub: "ready to apply it", correct: true, feedback: "somewhat" });
      _setChoice("arena-unlock-choice-good", { mark: "~", main: "Read it — still fuzzy", sub: "didn't fully get it", correct: false, feedback: "not_much" });
      _setChoice("arena-unlock-choice-okay", { hidden: true });
      _setChoice("arena-unlock-choice-bad", { mark: "↦", main: "Skip for now", sub: "come back later", skip: true });
      return;
    }
    if (ratingPromptEl) ratingPromptEl.textContent = "How did you do on this exercise?";
    _setChoice("arena-unlock-choice-best", { mark: "✓", main: "Solved in target time", sub: "no help", correct: true, feedback: "a_lot" });
    _setChoice("arena-unlock-choice-good", { mark: "✓", main: "Solved in target time", sub: "with a hint", correct: true, feedback: "somewhat" });
    _setChoice("arena-unlock-choice-okay", { mark: "✓", main: "Solved correctly", sub: "over target time", correct: true, feedback: "not_much" });
    _setChoice("arena-unlock-choice-bad", { mark: "✗", main: "Looked up", sub: "the solution", correct: false, feedback: "a_lot" });
  };

  const showCard = (ex) => {
    currentEx = ex;
    // ERE: pick the worked/faded/full tier for this atom by current mastery.
    const tier = _ereTierPick(ex);
    currentTierPath = tier ? tier.path : null;
    currentTierKind = tier ? tier.kind : "full";
    beforeScores = _snapshotBeforeAtoms(ex);
    // Flip the static ARENA copy when this is a drill card. Drills aren't
    // "unlocked by clearing prereqs"; they BUILD prereqs through hands-on
    // Colab work. Restore defaults on ARENA exercises so the same DOM
    // serves both flows.
    if (bannerEl) bannerEl.textContent = ex.isComposite ? COMPOSITE_BANNER : (ex.isDrill ? DRILL_BANNER : DEFAULT_BANNER);
    if (subEl) {
      if (currentTierKind === "worked") subEl.textContent = WORKED_SUB;
      else if (currentTierKind === "faded") subEl.textContent = FADED_SUB;
      else subEl.textContent = ex.isComposite ? COMPOSITE_SUB : (ex.isDrill ? DRILL_SUB : DEFAULT_SUB);
    }
    // Worked tier = study material → read-through completion states; else the
    // normal outcome self-rating. (P1.4)
    _configureChoices(currentTierKind);
    // Card title + Ctrl+F heading. The worked/faded tiers open a DIFFERENT
    // notebook than the full exercise (their own scaffold ladder), so we must
    // label the card from the notebook we actually open — otherwise the card
    // says "ex2: dispatch…" while Colab opens "faded example 3", which is
    // exactly the mismatch the tester reported. Tier notebooks open at the top,
    // so there is no Ctrl+F target: hide the heading block and copy nothing.
    // Full drills keep ex.heading as the Ctrl+F target.
    const tiered = currentTierKind === "worked" || currentTierKind === "faded";
    const ctrlFText = tiered ? "" : stripBackticks(ex.heading || ex.title);
    titleEl.textContent = tiered ? _tierLabelFromPath(currentTierPath, currentTierKind) : ex.title;
    if (headingBlockEl) headingBlockEl.classList.toggle("hidden", tiered || !ctrlFText);
    if (headingLabelEl) headingLabelEl.textContent = ex.isDrill ? DRILL_HEADING_LABEL : DEFAULT_HEADING_LABEL;
    headingEl.textContent = ctrlFText;
    whyEl.textContent = renderWhyMet(ex);
    colabBtn.href = colabHrefForUnlock();
    colabBtn.setAttribute("data-copy-key", ctrlFText);
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
  // state. Returns the parsed response object ({updated, atom_mastery}) on
  // success, {} otherwise.
  const postArenaRating = async (ex, { feedback, correct }) => {
    if (!ex || typeof apiFetch !== "function") return {};
    if (typeof authToken !== "undefined" && !authToken) return {};
    const subtopics = _prereqKeysForExercise(ex);
    if (!subtopics.length) return {};
    const rating = window.ArenaUnlockTimer?.getRating?.() || {};
    const body = {
      exercise_title: ex.title,
      subtopics,
      // Atom ids this exercise practices — drives the per-atom BKT update +
      // encompassing FIRe credit on the backend. Composite drills carry the
      // full list; single-atom drills carry one. ARENA exercises without atom
      // tags send []; they still bump the EWMA area readout via `subtopics`.
      atom_ids: _atomIdsForExercise(ex),
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
        return {};
      }
      const data = await res.json().catch(() => ({}));
      // Refresh cached scores so the next unlock check sees the bump.
      if (window.ArenaUnlock?.refreshScores) window.ArenaUnlock.refreshScores().catch(() => {});
      // Re-hydrate adaptiveStateJson so the fresh per-atom BKT posteriors
      // (atom_mastery) land in the readiness bridge. refreshScores only warms
      // the subtopic gate cache; it does NOT pull atom_mastery.
      if (typeof loadBackendAdaptiveState === "function") loadBackendAdaptiveState().catch(() => {});
      return data || {};
    } catch (err) {
      console.warn("[ArenaUnlock] arena-rating error:", err);
      return {};
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

  // Started as a parameterized copy of practice/bars.js#showEwmaAccuracy: a
  // refs bag instead of singleton DOM globals, so there can be one per
  // subtopic. That original is gone — the practice screen's accuracy bar was
  // deleted when it went to one stage ladder — and this is now the only
  // animated accuracy row in the app. It stays because it says which subtopic
  // each row is for, which is what the single-bar version never could.
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

  // Render one animated bar per ATOM the exercise practiced, before→after the
  // BKT update. `beforeMap` is {atomId: p0to1} snapshotted pre-POST; `afterMap`
  // is the response's atom_mastery {atomId: newPosterior0to1} — which also
  // includes FIRe-credited encompassed atoms not in beforeMap (shown from
  // p_init). This is the per-atom mastery estimate, the thing a rating moves.
  const renderResults = (beforeMap, afterMap) => {
    if (!resultListEl) return;
    resultListEl.innerHTML = "";
    const after = afterMap && typeof afterMap === "object" ? afterMap : {};
    // Union of directly-practiced atoms (beforeMap) + FIRe-credited atoms (only
    // in after) so encompassing credit is visible too.
    const keys = Array.from(new Set([...Object.keys(beforeMap || {}), ...Object.keys(after)]));
    if (!keys.length) {
      resultListEl.textContent = "No atoms to update for this exercise.";
      return;
    }
    keys.forEach((atomId, idx) => {
      const before = Number.isFinite(beforeMap?.[atomId]) ? beforeMap[atomId] : 0.10;
      const aft = Number.isFinite(after[atomId]) ? after[atomId] : null;
      const { shell, refs } = _buildBarRow(atomId);
      resultListEl.appendChild(shell);
      // Stagger slightly so bars cascade in instead of all moving in sync.
      setTimeout(() => _animateBarRow(refs, before, aft), 120 * idx);
    });
  };

  const hideCard = () => {
    // VIEW SWAP back — restore the practice question UI in the same slot.
    unlockPage.classList.add("hidden");
    practiceContainer.classList.remove("hidden");
  };

  hintBtn.addEventListener("click", () => {
    // Per-drill nudge hint, keyed by notebookPath (window.__drillHints, loaded
    // from drill-hints-manifest.js). Falls back gracefully if absent.
    const hints = window.__drillHints || {};
    const hint = currentEx && currentEx.notebookPath ? hints[currentEx.notebookPath] : "";
    placeholderEl.textContent = hint || "No hint available for this exercise yet.";
    placeholderEl.classList.remove("hidden");
  });

  answerBtn.addEventListener("click", () => {
    // Open the solution notebook (answer typed in, runs top-to-bottom) in a
    // new tab — routed to the student's Delta-Drills fork via colabUpstreamHref.
    const href = solutionHrefForUnlock();
    if (href && href !== "#") {
      window.open(href, "_blank", "noopener");
    } else {
      placeholderEl.textContent = "No solution notebook available for this exercise yet.";
      placeholderEl.classList.remove("hidden");
    }
  });

  // Explicit "Copy" button beside the displayed heading. The tester disliked
  // Open-in-Colab silently copying to his clipboard without asking, so nothing
  // is copied unless he clicks Copy. Only full-drill cards show the heading
  // block (tiered notebooks open at the top — see showCard). Open-in-Colab now
  // just navigates.
  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      const text = headingEl.textContent || "";
      if (!text || !navigator.clipboard?.writeText) return;
      navigator.clipboard.writeText(text)
        .then(() => {
          copyBtn.textContent = "✓ Copied";
          setTimeout(() => { copyBtn.textContent = "📋 Copy"; }, 1500);
        })
        .catch(() => {});
    });
  }

  // Mark the exercise shown (unless it was a Targeted-Practice manual launch —
  // those stay re-practicable), swap back to the practice view, and fire the
  // continue callback. Shared by the Continue button and the "Skip for now"
  // study-material action.
  const _finishAndContinue = () => {
    if (currentEx && !currentEx._targetedPractice) {
      if (currentEx.isDrill && typeof window.markDrillShown === "function") {
        window.markDrillShown(currentEx.id);
      } else if (typeof window.markArenaExerciseShown === "function") {
        window.markArenaExerciseShown(currentEx.title);
      }
    }
    hideCard();
    const cb = onContinueCallback;
    onContinueCallback = null;
    currentEx = null;
    if (typeof cb === "function") cb();
  };

  // Stage 1: self-rating. Each button encodes {correct, feedback} via data-attrs
  // (or data-skip for the study-material "Skip for now"). One click → POST →
  // render deltas → Continue (or, for skip, straight to the next question).
  choiceButtons.forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (!currentEx) return;
      if (btn.dataset.skip === "true") {
        _finishAndContinue();
        return;
      }
      const correct = btn.dataset.correct === "true";
      const feedback = btn.dataset.feedback;
      choiceButtons.forEach((b) => { b.disabled = true; });
      const data = await postArenaRating(currentEx, { feedback, correct });
      renderResults(beforeScores, data?.atom_mastery || {});
      setStage("result");
      setTimeout(() => continueBtn.focus({ preventScroll: true }), 0);
    });
  });

  // Stage 2: Continue → finish + load the next question.
  continueBtn.addEventListener("click", _finishAndContinue);

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
      // 1. ARENA legacy unlocks (callum's ARENA_3.0 notebooks).
      if (typeof window.getNextUnshownUnlockedArenaExercise === "function") {
        const ex = window.getNextUnshownUnlockedArenaExercise();
        if (ex) {
          onContinueCallback = onContinue;
          showCard(ex);
          return true;
        }
      }
      // 2. Procedural drills (new-algo, Delta-Drills repo). Drills BUILD
      // subtopic mastery; we surface each once when its EWMA crosses the
      // drill's unlockMinPct so the student moves from text-bank reps into
      // a real applied notebook.
      if (typeof window.getNextUnshownUnlockedDrill === "function") {
        const drill = window.getNextUnshownUnlockedDrill();
        if (drill) {
          onContinueCallback = onContinue;
          showCard(drill);
          return true;
        }
      }
      return false;
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
