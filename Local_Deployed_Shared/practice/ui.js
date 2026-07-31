/* ================================================================
   PRACTICE UI — rendering + feedback widgets
   ================================================================ */

function isCalibrationQuestion(q) {
  if (!q) return false;
  if (typeof q.is_cold_start === "boolean") return q.is_cold_start;
  const overrideN = Number.isFinite(q.subtopic_n) ? q.subtopic_n : undefined;
  return isColdStart(q.subtopic, overrideN);
}

// A question is "torch" if the bank tags it so, or its starter/solution imports
// torch.
function questionIsTorch(q) {
  if (!q) return false;
  if (q.primary_library === "torch") return true;
  const blob = `${q.starter_code || ""}\n${q.solution_code || ""}`;
  return /(^|\n)\s*(import\s+torch\b|from\s+torch[\s.])/.test(blob);
}

/* `torchNeedsColab` was here. It answered "does this question need Colab
   because Pyodide cannot import torch?" — a question with only one answer now
   (everything is worked in Colab, whatever it imports), so the routing it fed
   is unconditional and lives in `renderColabCard` below. `questionIsTorch`
   above survives: practice/pyodide-boot.js still uses it to keep a LESSON's
   torch example off Pyodide and on the backend fork runner. */

function _escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Render a question prompt as separated prose + code instead of one wall of
// text, and let KaTeX render the $…$ math (it was showing raw before). Fenced
// ```code``` blocks become a styled <pre>; inline `code` becomes <code>; math
// is left for KaTeX. Returns nothing — writes into #question-text.
function renderQuestionBody(q) {
  const raw = (q && q.question_text) || "";
  // Pull fenced code blocks out first so we don't KaTeX/escape-mangle them.
  const parts = [];
  const fence = /```[a-zA-Z0-9]*\n?([\s\S]*?)```/g;
  let last = 0;
  let m;
  while ((m = fence.exec(raw)) !== null) {
    if (m.index > last) parts.push({ type: "prose", text: raw.slice(last, m.index) });
    parts.push({ type: "code", text: m[1].replace(/\n$/, "") });
    last = fence.lastIndex;
  }
  if (last < raw.length) parts.push({ type: "prose", text: raw.slice(last) });

  const html = parts.map((p) => {
    if (p.type === "code") {
      return `<pre class="question-code-block"><code>${_escapeHtml(p.text)}</code></pre>`;
    }
    // Escape HTML, then turn inline `backtick` spans into <code>. Math ($…$) is
    // left untouched for KaTeX auto-render below. Blank lines split paragraphs
    // (each its own div — .question-prose has no pre-wrap, so raw \n collapses).
    return p.text
      .split(/\n{2,}/)
      .map((para) => para.trim())
      .filter(Boolean)
      .map((para) => {
        const escaped = _escapeHtml(para).replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`);
        return `<div class="question-prose">${escaped}</div>`;
      })
      .join("");
  }).join("");

  questionText.innerHTML = html || _escapeHtml(raw);

  // Render the math. auto-render.min.js loads `defer`, so it's usually ready by
  // the time a question renders; guard in case it isn't.
  if (typeof window.renderMathInElement === "function") {
    try {
      window.renderMathInElement(questionText, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true },
        ],
        throwOnError: false,
      });
    } catch (_) { /* malformed LaTeX — leave the raw text */ }
  }
}

/* THE COLAB CARD — where this problem gets worked.

   Every question routes to a notebook now. `applyTorchRouting` used to sit
   here and did the opposite job: it decided whether ONE library's questions
   needed Colab and, when they did, hid the editor, the hints and the submit
   controls. There is no editor left to hide and no per-library decision left
   to make, so what remains is: name the notebook, point at the exact cell.

   Silent when there is no mapping. 424 of the 499 bank questions are anchored
   in a notebook; a question that is not (a placement probe from a parked
   chapter, say) still renders and is still answerable — it just has nowhere to
   send you, and claiming otherwise with a dead link would be worse. */
function renderColabCard(q) {
  if (!colabCard) return;
  var url = window.ColabRoute ? window.ColabRoute.urlFor(q) : "";
  var lesson = window.ColabRoute ? window.ColabRoute.lessonFor(q) : null;
  colabCard.classList.toggle("hidden", !url);
  hideColabNote();
  // CLEAR the link when there is no route, do not just hide it. `showColabNote`
  // un-hides this card to report an error, and a stale href left behind by the
  // previous question would then be offered as if it were this problem's
  // notebook — the worst possible wrong answer, because it looks right.
  if (colabOpenLink) {
    colabOpenLink.href = url || "#";
    colabOpenLink.classList.toggle("hidden", !url);
  }
  if (colabCardLabel) {
    colabCardLabel.textContent = !url
      ? ""
      : (lesson && lesson.title ? lesson.title : "Notebook");
  }
}

/* Is there anywhere to work this problem?
   Used by the answer timer: running out of time only means "was not able to run
   the result" if there was a result to run. */
function questionHasColabRoute(q) {
  return !!(window.ColabRoute && window.ColabRoute.urlFor(q));
}

/* Steer the one Colab tab to this problem.

   Only ever best-effort — see the header of practice/colab-route.js. A browser
   permits window.open during a user gesture; the click that advanced to this
   question is one, an expiring answer timer is not. When it does not open we
   say so rather than leaving the learner waiting for a tab, because the link
   right next to the note is a working way through. */
function openColabForQuestion(q) {
  if (!window.ColabRoute || !colabCard || colabCard.classList.contains("hidden")) return;
  // ONLY when the problem is actually in front of the learner.
  //
  // `initPractice` renders a question during boot so the page has something to
  // show the moment a session starts. That render is invisible — the practice
  // page may not even be the open tab, and `session-idle` hides the question
  // behind the setup panel — and it used to fire this, so merely loading the
  // app hijacked a tab to a notebook for a problem nobody had been asked yet.
  // Both conditions are needed: `session-idle` alone misses the boot render on
  // another tab, and visibility alone misses the setup screen.
  var page = document.getElementById("page-practice");
  if (!page || page.classList.contains("hidden") || page.classList.contains("session-idle")) {
    return;
  }
  if (!window.ColabRoute.open(q)) {
    showColabNote("Couldn't open the notebook automatically — use the link above.");
  }
}

function showColabNote(message) {
  if (!colabCardNote) return;
  // The card itself may be hidden (unmapped question); the note still has to be
  // readable, since this is now the page's only error surface. `renderColabCard`
  // has already emptied the link and label in that case, so revealing the card
  // here shows the message alone rather than a previous problem's notebook.
  if (colabCard) colabCard.classList.remove("hidden");
  colabCardNote.textContent = message;
  colabCardNote.classList.remove("hidden");
}

function hideColabNote() {
  if (!colabCardNote) return;
  colabCardNote.textContent = "";
  colabCardNote.classList.add("hidden");
}

function stableQuestionId(q) {
  const raw = q?.question_id ?? q?.id;
  return Number.isFinite(raw) ? String(raw) : String(raw || "");
}

function renderQuestionIdChip(q) {
  if (!questionIdChip) return;
  const id = stableQuestionId(q);
  if (!id) {
    questionIdChip.textContent = "ID --";
    questionIdChip.disabled = true;
    questionIdChip.removeAttribute("data-question-id");
    return;
  }
  questionIdChip.disabled = false;
  questionIdChip.dataset.questionId = id;
  questionIdChip.textContent = `ID ${id}`;
  questionIdChip.title = `Copy stable problem ID: ${id}`;
  questionIdChip.onclick = async () => {
    const value = `question_id=${id}`;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(value);
      }
      questionIdChip.textContent = "Copied";
      window.setTimeout(() => {
        if (questionIdChip.dataset.questionId === id) questionIdChip.textContent = `ID ${id}`;
      }, 900);
    } catch (_) {
      questionIdChip.textContent = `ID ${id}`;
    }
  };
}

function renderQuestion(q, count) {
  // Single-lesson demo (?lesson=): keep the previewed lesson on screen — a
  // late session-resume render must not clobber it.
  if (window.__lessonDemoOnly) return;
  if (curatedExcludedIds.has(q.question_id)) {
    PracticeAPI.getNextQuestion().then((nextQ) => renderQuestion(nextQ, count));
    return;
  }
  if (staleGaussianQuestion(q)) {
    PracticeAPI.getNextQuestion().then((nextQ) => renderQuestion(nextQ, count));
    return;
  }
  practiceQuestionCount = count;
  questionNumber.textContent = "Question " + practiceQuestionCount;
  renderQuestionBody(q);
  // Names the concept under test and, on the scaffolded rungs, puts the worked
  // example back on screen beside the problem. Must run AFTER renderQuestionBody
  // — that call replaces #question-text wholesale.
  if (window.LadderUI) window.LadderUI.decorate(q);
  renderQuestionImports(q);
  renderQuestionVisual(q);
  // The starter code is no longer copied into an editor — it is already the
  // first line of the problem's cell in the notebook, which is where the
  // learner types. Rendering it here too would be a second copy nobody edits.
  subtopicLabel.textContent = q.topic ? `${q.topic}: ${q.subtopic}` : q.subtopic;
  difficultyLabel.textContent = "Difficulty: " + q.difficulty + " / 100";
  renderQuestionIdChip(q);
  if (typeof updateGraphJump === "function") updateGraphJump(q);
  questionMetaTop.classList.add("hidden");

  // Placement diagnostic badge (backend ALEKS-style placement) takes priority
  // over the legacy per-skill cold-start badge (offline staircase only).
  const overrideN = Number.isFinite(q.subtopic_n) ? q.subtopic_n : undefined;
  const coldStart = isCalibrationQuestion(q);
  const csIndex = Number.isFinite(q.subtopic_n) ? q.subtopic_n + 1 : coldStartIndex(q.subtopic, overrideN);
  if (q.diagnostic_active) {
    coldStartLabel.textContent =
      `Placement diagnostic — question ${q.diagnostic_probe_index} of ≤${q.diagnostic_budget} · exploring ${q.diagnostic_area || q.topic}`;
    if (coldStartNote) {
      coldStartNote.textContent =
        "A short adaptive placement: each question is picked to tell us the most about your level, hopping across topics instead of grinding one area. Answer what you can — and hit “I don't know yet” when something is clearly above you; that's a fast, honest signal (no penalty). It ends automatically once your level is pinned down, and regular practice starts exactly there.";
    }
    coldStartBadge.classList.remove("hidden");
  } else if (coldStart && csIndex) {
    // Calibration is PER SKILL, not global — say so, or "1 of 3" on overall
    // Question 8 reads as a stuck counter (tester hit exactly this).
    coldStartLabel.textContent = `Calibrating “${q.subtopic}” — ${csIndex} of 3`;
    if (coldStartNote) {
      // Copy must match reality: selection is adaptive from question 1
      // (wrong → easier, right → harder), NOT "3 questions at fixed
      // difficulties" (that described the old fixed-ramp system, removed).
      coldStartNote.textContent =
        `The first few questions in each skill probe your level — difficulty adapts from your answers (easier after a miss, harder after a hit). This counter restarts whenever a new skill (like “${q.subtopic}”) first comes up, and the accuracy bar stays hidden until it finishes.`;
    }
    coldStartBadge.classList.remove("hidden");
  } else {
    coldStartBadge.classList.add("hidden");
  }
  // "I don't know yet" only exists during placement — in normal practice the
  // Skip button (nothing recorded) covers that need.
  if (typeof practiceDontKnowBtn !== "undefined" && practiceDontKnowBtn) {
    practiceDontKnowBtn.classList.toggle("hidden", !q.diagnostic_active);
    practiceDontKnowBtn.disabled = false;
  }
  setTargetDifficultyInitial(getTargetDifficultyForQuestion(q));
  solutionCode.textContent = q.solution_code;
  setupQuestionAids(q);
  overrideRow.classList.add("hidden");

  // Reset to pre-report state
  practiceSubmitArea.classList.remove("hidden");
  selfReportNoBtn.disabled = false;
  selfReportYesBtn.disabled = false;
  practiceFeedbackArea.classList.add("hidden");
  practiceFeedbackArea.classList.remove("checking");
  ewmaAccuracy.classList.add("hidden");
  ewmaAccuracyFill.style.width = "0%";
  showFeedbackButtons();
  resetMissedFactRow();
  questionMetaTop.classList.add("hidden");
  // Point at the notebook cell this problem lives in, then steer the Colab tab
  // there. Rendering first matters: the auto-open reads the card's visibility
  // to decide whether there is anywhere to go.
  renderColabCard(q);
  openColabForQuestion(q);

  // Set up accuracy bar initial state (mirrors setTargetDifficultyInitial).
  // Backend mode: use p_current from the question response (adaptiveStateJson is null).
  // Pyodide mode: read from the adaptive state JSON.
  ewmaAccuracyPBefore = Number.isFinite(q.p_current)
    ? q.p_current
    : getEwmaFromAdaptiveState(q.subtopic);
  if (coldStart || q.diagnostic_active) {
    showEwmaAccuracyCalibration(q.subtopic);
  } else {
    showEwmaAccuracyInitial(ewmaAccuracyPBefore, q.subtopic);
  }

  // Reset AI explanation
  aiExplanationSection.classList.add("hidden");
  aiExplanationText.textContent = "";

  // Rigid session: every rendered question starts a fresh strict answer
  // countdown (no-op while no session is running).
  PracticeSession.onQuestionRendered();

  const pending = practiceProgress.pendingFeedback;
  if (pending) {
    if (pending.questionId === q.question_id) {
      applyPendingFeedbackState(pending);
    } else {
      practiceProgress.pendingFeedback = null;
      savePracticeProgress(practiceProgress);
    }
  }
}

// Configure the Show Hint aid for the current question and reset its reveal
// state. Hint comes straight off the question payload. (In-browser bank
// questions reveal their solution code after submit; the Colab solution link
// lives on the procedural-drill cards, not the practice tab.)
function setupQuestionAids(q) {
  const hint = q && typeof q.hint === "string" ? q.hint.trim() : "";
  if (hintText) hintText.textContent = hint;
  if (showHintBtn) showHintBtn.classList.toggle("hidden", !hint);
  if (hintSection) hintSection.classList.add("hidden");
}

function getTargetDifficultyForQuestion(q) {
  if (q && Number.isFinite(q.target_difficulty)) return q.target_difficulty;
  const fromState = getTargetDifficultyFromAdaptiveState(q?.subtopic);
  if (Number.isFinite(fromState)) return fromState;
  return Number.isFinite(q?.difficulty) ? q.difficulty : 0;
}

function showFeedbackButtons() {
  feedbackButtons.forEach((btn) => {
    btn.classList.remove("hidden");
    btn.classList.remove("feedback-btn--pressed");
    btn.disabled = false;
  });
  nextProblemBtn.classList.add("hidden");
}

function showNextProblemButton() {
  feedbackButtons.forEach((btn) => btn.classList.add("hidden"));
  nextProblemBtn.classList.remove("hidden");
}

function parseStarterImports(source) {
  if (!source || typeof source !== "string") return [];
  const imports = [];
  const seen = new Set();
  for (const line of source.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const importMatch = trimmed.match(/^import\s+(.+)$/);
    if (importMatch) {
      for (const part of importMatch[1].split(",").map((p) => p.trim()).filter(Boolean)) {
        const item = `import ${part}`;
        if (!seen.has(item)) {
          seen.add(item);
          imports.push(item);
        }
      }
      continue;
    }
    const fromMatch = trimmed.match(/^from\s+([A-Za-z0-9_.]+)\s+import\s+(.+)$/);
    if (fromMatch) {
      for (const part of fromMatch[2].split(",").map((p) => p.trim()).filter(Boolean)) {
        const item = `from ${fromMatch[1]} import ${part}`;
        if (!seen.has(item)) {
          seen.add(item);
          imports.push(item);
        }
      }
    }
  }
  return imports;
}

async function loadNotebookArrayPreview(dataUrl, tempFilePath) {
  const pyodide = await initPyodide();
  if (!pyodide) throw new Error("Python runtime unavailable.");
  const response = await fetch(dataUrl, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${dataUrl} (${response.status})`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  pyodide.FS.writeFile(tempFilePath, bytes);
  const payload = await pyodide.runPythonAsync(`
import json
import numpy as np
json.dumps(np.load(${JSON.stringify(tempFilePath)}).tolist())
`);
  return JSON.parse(payload);
}

let questionHelperRenderSeq = 0;

async function renderQuestionImports(q) {
  if (!questionImports || !questionImportsList) return;
  const renderSeq = ++questionHelperRenderSeq;
  questionImports.classList.add("hidden");
  questionImportsList.innerHTML = "";
  const helperItems = await getNotebookHelperItems(q);
  const visibleItems = helperItems.filter((item) => {
    if (item.kind === "arena-array") return true;
    const code = (item.code || "").trim();
    const label = (item.label || "").trim();
    return !!item.context && code !== label;
  });
  if (renderSeq !== questionHelperRenderSeq) return;
  if (!visibleItems.length) {
    questionImports.classList.add("hidden");
    return;
  }
  for (const item of visibleItems) {
    const itemWrap = document.createElement("div");
    itemWrap.className = "question-import-item";

    const pill = document.createElement("button");
    pill.type = "button";
    pill.className = "question-import";
    pill.textContent = item.label;
    pill.setAttribute("aria-expanded", "false");

    const detail = document.createElement("div");
    detail.className = "question-import-detail hidden";
    const note = document.createElement("div");
    note.className = "question-import-detail-note";
    note.textContent = item.note;
    detail.appendChild(note);
    if (item.context) {
      const context = document.createElement("div");
      context.className = "question-import-detail-context";
      context.textContent = item.context;
      detail.appendChild(context);
    }

    let previewState = null;
    if (item.kind === "arena-array" && item.dataUrl) {
      const preview = document.createElement("div");
      preview.className = "question-import-detail-preview";
      const previewNote = document.createElement("div");
      previewNote.className = "question-import-detail-preview-note";
      previewNote.textContent = `Source data: ${item.dataUrl}`;
      const previewCanvas = document.createElement("canvas");
      previewCanvas.className = "question-import-detail-preview-canvas hidden";
      const previewText = document.createElement("pre");
      previewText.className = "question-import-detail-preview-json hidden";
      preview.appendChild(previewNote);
      preview.appendChild(previewCanvas);
      preview.appendChild(previewText);
      detail.appendChild(preview);
      previewState = {
        loaded: false,
        loading: false,
        previewNote,
        previewCanvas,
        previewText,
      };
    }

    const shouldShowCode = item.kind === "arena-array" || (item.code && item.code.trim() !== item.label.trim());
    if (shouldShowCode) {
      const pre = document.createElement("pre");
      pre.className = "question-import-detail-code";
      const code = document.createElement("code");
      code.textContent = item.code;
      pre.appendChild(code);
      detail.appendChild(pre);
    }

    pill.addEventListener("click", () => {
      const expanded = !detail.classList.contains("hidden");
      detail.classList.toggle("hidden", expanded);
      pill.setAttribute("aria-expanded", String(!expanded));
      itemWrap.classList.toggle("expanded", !expanded);

      if (!expanded && previewState && !previewState.loaded && !previewState.loading) {
        previewState.loading = true;
        previewState.previewNote.textContent = "Loading actual notebook array...";
        loadNotebookArrayPreview(
          item.dataUrl,
          `/tmp/delta-drills-helper-${renderSeq}-${Math.random().toString(36).slice(2)}.npy`
        )
          .then((arrayData) => {
            if (renderSeq !== questionHelperRenderSeq) return;
            previewState.loaded = true;
            previewState.previewNote.textContent = "Loaded from notebook data file.";
            window.renderDeltaArrayToCanvas(previewState.previewCanvas, arrayData);
            previewState.previewCanvas.classList.remove("hidden");
            const rawJson = JSON.stringify(arrayData, null, 2);
            previewState.previewText.textContent =
              rawJson.length > 16000 ? rawJson.slice(0, 16000) + "\n...\n(truncated)" : rawJson;
            previewState.previewText.classList.remove("hidden");
          })
          .catch((err) => {
            if (renderSeq !== questionHelperRenderSeq) return;
            previewState.previewNote.textContent =
              "Failed to load notebook array: " + (err.message || String(err));
          });
      }
    });

    itemWrap.appendChild(pill);
    itemWrap.appendChild(detail);
    questionImportsList.appendChild(itemWrap);
  }
  questionImports.classList.remove("hidden");
}

function shortSubtopicName(subtopic) {
  if (!subtopic) return subtopic;
  const colon = subtopic.indexOf(": ");
  return colon >= 0 ? subtopic.slice(colon + 2) : subtopic;
}

function applyPendingFeedbackState(pending) {
  practiceSubmitArea.classList.add("hidden");
  practiceFeedbackArea.classList.remove("hidden");
  applyResult(!!pending.correct);
  questionMetaTop.classList.remove("hidden");
  overrideRow.classList.add("hidden");
  showNextProblemButton();
  setTargetDifficultyFinal(pending.oldTarget, pending.newTarget);
  if (Number.isFinite(pending.pAfter)) {
    setEwmaAccuracyFinal(pending.pBefore, pending.pAfter, pending.subtopic);
  }
}

// Apply correct/incorrect result to the feedback area UI.
function applyResult(correct) {
  resultBadge.textContent = correct ? "Correct" : "Incorrect";
  resultBadge.className = "result-badge " + (correct ? "correct" : "incorrect");
  overrideRow.classList.toggle("hidden", correct);
  practiceFeedbackArea.classList.remove("checking");
  questionMetaTop.classList.remove("hidden");
  // Buttons map to the engine's not_much / somewhat / a_lot, but difficulty is
  // driven by your mastery — so these are a felt-difficulty self-report with
  // "About right" as the clear default (see the helper line + .feedback-btn--default).
  if (correct) {
    feedbackPrompt.textContent = "Nice work. How did that feel?";
    feedbackButtons.forEach((btn, i) => {
      btn.textContent = ["About right", "A little easy", "Way too easy"][i];
    });
  } else {
    feedbackPrompt.textContent = "No worries. How did that feel?";
    feedbackButtons.forEach((btn, i) => {
      btn.textContent = ["About right", "A little hard", "Way too hard"][i];
    });
  }
  // "I missed one concrete thing" only makes sense after a wrong answer.
  if (missedFactRow) missedFactRow.classList.toggle("hidden", correct);
}

// Failed-test-case breakdown — shows WHICH cases failed with expected vs got,
// so a wrong grade is evidence, not a verdict (tester: "it needs to show all
// the cases that it tested and where it actually failed"). Renders into a
// dynamic <div> right under the result badge; hidden on correct/next.
function renderFailedTests(result, question) {
  let block = document.getElementById("failed-tests-block");
  if (!result || result.correct || !Array.isArray(result.failed_tests) || !result.failed_tests.length) {
    if (block) block.classList.add("hidden");
    return;
  }
  if (!block) {
    block = document.createElement("div");
    block.id = "failed-tests-block";
    block.className = "failed-tests-block";
    const anchor = document.getElementById("feedback-prompt");
    if (!anchor || !anchor.parentNode) return;
    anchor.parentNode.insertBefore(block, anchor);
  }
  const total = (question && Array.isArray(question.test_cases) && question.test_cases.length) || result.failed_tests.length;
  const id = stableQuestionId(question);
  const clip = (s) => {
    const t = String(s == null ? "" : s);
    return t.length > 220 ? t.slice(0, 220) + "…" : t;
  };
  const rows = result.failed_tests.map((t, i) => {
    if (t.error) return `✗ failing case ${i + 1}: error — ${_escapeHtml(clip(t.error))}`;
    return `✗ failing case ${i + 1}:\n    expected: ${_escapeHtml(clip(t.expected))}\n    got:      ${_escapeHtml(clip(t.actual))}`;
  });
  block.innerHTML =
    `<div class="failed-tests-title">Failed ${result.failed_tests.length} of ${total} test case${total === 1 ? "" : "s"}${id ? ` · ID ${_escapeHtml(id)}` : ""}</div>` +
    `<pre class="failed-tests-body">${rows.join("\n")}</pre>`;
  block.classList.remove("hidden");
}

function hideFailedTests() {
  const block = document.getElementById("failed-tests-block");
  if (block) block.classList.add("hidden");
}

// Clear the "missed one concrete thing" affordance between questions.
function resetMissedFactRow() {
  if (missedFactRow) missedFactRow.classList.add("hidden");
  if (missedFactStatus) missedFactStatus.classList.add("hidden");
  if (missedFactBtn) missedFactBtn.classList.remove("flagged");
}
