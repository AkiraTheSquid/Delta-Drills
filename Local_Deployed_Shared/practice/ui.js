/* ================================================================
   PRACTICE UI — rendering + feedback widgets
   ================================================================ */

function isCalibrationQuestion(q) {
  if (!q) return false;
  if (typeof q.is_cold_start === "boolean") return q.is_cold_start;
  const overrideN = Number.isFinite(q.subtopic_n) ? q.subtopic_n : undefined;
  return isColdStart(q.subtopic, overrideN);
}

function renderQuestion(q, count) {
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
  questionText.textContent = q.question_text;
  renderQuestionImports(q);
  renderQuestionVisual(q);
  codeEditor.value =
    q.starter_code ||
    "import numpy as np\nnp.random.seed(0)\n\n# Write your solution here\n";
  subtopicLabel.textContent = q.topic ? `${q.topic}: ${q.subtopic}` : q.subtopic;
  difficultyLabel.textContent = "Difficulty: " + q.difficulty + " / 100";
  questionMetaTop.classList.add("hidden");

  // Cold-start calibration badge
  const overrideN = Number.isFinite(q.subtopic_n) ? q.subtopic_n : undefined;
  const coldStart = isCalibrationQuestion(q);
  const csIndex = Number.isFinite(q.subtopic_n) ? q.subtopic_n + 1 : coldStartIndex(q.subtopic, overrideN);
  if (coldStart && csIndex) {
    coldStartLabel.textContent = `Calibrating — ${csIndex} of 3`;
    if (coldStartNote) {
      coldStartNote.textContent =
        "First 3 questions use fixed difficulties to calibrate your level. The next difficulty is preset during calibration, so the usual accuracy bar is hidden until calibration finishes.";
    }
    coldStartBadge.classList.remove("hidden");
  } else {
    coldStartBadge.classList.add("hidden");
  }
  setTargetDifficultyInitial(getTargetDifficultyForQuestion(q));
  solutionCode.textContent = q.solution_code;
  overrideRow.classList.add("hidden");

  // Reset to pre-submit state
  practiceSubmitArea.classList.remove("hidden");
  practiceSubmitBtn.disabled = false;
  practiceFeedbackArea.classList.add("hidden");
  practiceFeedbackArea.classList.remove("checking");
  ewmaAccuracy.classList.add("hidden");
  ewmaAccuracyFill.style.width = "0%";
  showFeedbackButtons();
  questionMetaTop.classList.add("hidden");

  // Set up accuracy bar initial state (mirrors setTargetDifficultyInitial).
  // Backend mode: use p_current from the question response (adaptiveStateJson is null).
  // Pyodide mode: read from the adaptive state JSON.
  ewmaAccuracyPBefore = Number.isFinite(q.p_current)
    ? q.p_current
    : getEwmaFromAdaptiveState(q.subtopic);
  if (coldStart) {
    showEwmaAccuracyCalibration(q.subtopic);
  } else {
    showEwmaAccuracyInitial(ewmaAccuracyPBefore, q.subtopic);
  }

  // Reset AI explanation
  aiExplanationSection.classList.add("hidden");
  aiExplanationText.textContent = "";

  // Reset timer for next question if timed mode is on
  if (timedModeToggle.checked) {
    startTimer();
  }

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

function getTargetDifficultyForQuestion(q) {
  if (q && Number.isFinite(q.target_difficulty)) return q.target_difficulty;
  const fromState = getTargetDifficultyFromAdaptiveState(q?.subtopic);
  if (Number.isFinite(fromState)) return fromState;
  return Number.isFinite(q?.difficulty) ? q.difficulty : 0;
}

function showFeedbackButtons() {
  feedbackButtons.forEach((btn) => btn.classList.remove("hidden"));
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
  if (correct) {
    feedbackPrompt.textContent = "Nailed it! How hard should we go next?";
    feedbackButtons.forEach((btn, i) => {
      btn.textContent = ["Inch it up", "Rev the engine", "Full throttle"][i];
    });
  } else {
    feedbackPrompt.textContent = "Tough one. How much should we dial it back?";
    feedbackButtons.forEach((btn, i) => {
      btn.textContent = ["Just a hair easier", "Take the edge off", "Back to basics"][i];
    });
  }
}
