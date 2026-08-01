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

// Colab routing applies ONLY where the runner can't grade torch. Backend mode
// grades torch in-process now (fork runner, torch preimported at boot), so
// torch questions there use the normal editor flow. Offline/Supabase practice
// runs on Pyodide, which cannot import torch at all → Colab.
// The Colab edition's answer for this question, or "" — either because this is
// the normal app, or because the question's lesson has no published notebook.
// See practice/colab_mode.js; `window.DDColab` is absent on any page that does
// not load it, so every call site must tolerate that.
function colabForkHref(q) {
  const dd = window.DDColab;
  return dd && dd.active() ? dd.hrefFor(q) : "";
}

function torchNeedsColab(q) {
  // Routing a torch drill to Colab hides the WHOLE right panel — editor, aids,
  // submit. That was survivable when torch drills were a rare fork, but the
  // bank is now 448/448 torch, so this condition fires on every question the
  // moment the runtime is not the backend, and it strands the learner on a
  // prompt with no controls and no way forward.
  //
  // Only route away when there is somewhere to route TO. With no notebook the
  // old behaviour was a dead end, not a fallback.
  if (!questionIsTorch(q)) return false;
  // The Colab edition routes on PURPOSE, so it ignores practiceMode — the point
  // of that deploy is that the notebook is where you work, whether or not the
  // backend could have graded it here. It still routes only where a notebook
  // exists: the 112 questions whose lessons have none stay on the editor, which
  // is the same "somewhere to route TO" rule as below.
  if (colabForkHref(q)) return true;
  if (practiceMode === "backend") return false;
  return !!(q && (q.problem_notebook_path || q.solution_notebook_path));
}

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

// BINARY interface per question: either the Colab card (offline torch) or the
// full editor flow (everything else) — never both half-shown at once (a torch
// card next to a live editor read as contradictory; tester feedback).
function applyTorchRouting(q) {
  const colabRoute = torchNeedsColab(q);
  // The Colab edition strips the page down to a tutor rail — no prompt, no
  // editor, no worked example, because all three are in the notebook and a
  // second copy beside it is what made the panel feel like the normal app with
  // a badge on it. `dd-no-notebook` is the escape hatch: the ~75 questions
  // whose lesson was never published have nowhere to go, so they get the full
  // page back rather than an empty rail. Styling in
  // styles/practice/colab-edition.css.
  if (window.DDColab && window.DDColab.active()) {
    document.documentElement.classList.toggle("dd-no-notebook", !colabRoute);
  }
  if (torchColabNotice) torchColabNotice.classList.toggle("hidden", !colabRoute);
  // The whole right editor panel + hint aids + submit/skip swap out together.
  const rightPanel = document.querySelector(".practice-right");
  if (rightPanel) rightPanel.classList.toggle("hidden", colabRoute);
  const aids = document.getElementById("practice-aids");
  if (aids) aids.classList.toggle("hidden", colabRoute);
  practiceSubmitArea.classList.toggle("hidden", colabRoute);
  if (!colabRoute) return;
  const toHref = (p) => (p && typeof colabUpstreamHref === "function") ? colabUpstreamHref(p) : "";
  // Primary "Open in Colab" → the PROBLEM notebook (starter, no answer). Fall
  // back to the solution notebook only if no problem notebook exists.
  if (torchColabLink) {
    // The bank's own notebook paths first (all empty today, but they are the
    // per-question answer and outrank a lesson-level one), then the Colab
    // edition's lesson notebook.
    const href = toHref(q && (q.problem_notebook_path || q.solution_notebook_path))
      || colabForkHref(q);
    torchColabLink.classList.toggle("hidden", !href);
    if (href) torchColabLink.href = href;
    // Framed by the extension: steer the tab beside us instead of waiting for a
    // click. No-op everywhere else — see DDColab.openNotebook.
    if (href && window.DDColab && typeof window.DDColab.openNotebook === "function") {
      window.DDColab.openNotebook(href);
    }
  }
  // Separate "Show solution" → the worked-answer notebook, only when we have a
  // distinct problem notebook (otherwise the primary link already IS the solution).
  if (torchSolutionLink) {
    const solHref = toHref(q && q.solution_notebook_path);
    const showSolution = !!(solHref && q && q.problem_notebook_path);
    torchSolutionLink.hidden = !showSolution;
    if (showSolution) torchSolutionLink.href = solHref;
  }
}

// The notebook index arrives over the network, and the first question can render
// before it lands — which would resolve to "no notebook" and leave the learner on
// an editor that cannot import torch. Re-run the routing for whatever is on
// screen once the index settles. Fires immediately on the normal app.
if (window.DDColab && window.DDColab.active()) {
  window.DDColab.whenReady(() => {
    const current = (typeof practiceProgress === "object" && practiceProgress)
      ? practiceProgress.currentQuestion
      : null;
    if (current) applyTorchRouting(current);
  });
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
  codeEditor.value =
    q.starter_code ||
    "import numpy as np\nnp.random.seed(0)\n\n# Write your solution here\n";
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

  // Reset to pre-submit state
  practiceSubmitArea.classList.remove("hidden");
  practiceSubmitBtn.disabled = false;
  practiceFeedbackArea.classList.add("hidden");
  practiceFeedbackArea.classList.remove("checking");
  ewmaAccuracy.classList.add("hidden");
  ewmaAccuracyFill.style.width = "0%";
  showFeedbackButtons();
  resetMissedFactRow();
  questionMetaTop.classList.add("hidden");
  // Torch drills swap the submit flow for the Colab-routing notice (must run
  // AFTER the submit area is un-hidden above so it can re-hide it for torch).
  applyTorchRouting(q);

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
