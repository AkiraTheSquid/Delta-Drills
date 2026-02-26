/* ================================================================
   PRACTICE.JS — Practice tab: questions, timer, feedback, Pyodide
   ================================================================
   Three practice modes:
     'backend'  — admin on localhost, uses local FastAPI endpoints
     'supabase' — non-admin / deployed, Pyodide engine + Supabase storage
     'local'    — fallback, Pyodide engine + localStorage

   API Contract (backend mode only):
     GET  /api/practice/next-question
          -> { question_id, question_text, subtopic, difficulty, expected_output }
     POST /api/practice/submit
          body: { question_id, user_code }
          -> { correct: bool, actual_output, expected_output }
     POST /api/practice/feedback
          body: { question_id, feedback: "not_much" | "somewhat" | "a_lot" }
   ================================================================ */

// --- Hardcoded fallback pool (used when questions.json fails to load) ---

const practiceQuestionPool = [
  {
    question_id: "q001",
    question_text: "Create a 5x5 matrix with values 1,2,3,4,5 on the diagonal. Use np.diag().",
    subtopic: "Array creation",
    difficulty: 50,
    expected_output: "[[1 0 0 0 0]\n [0 2 0 0 0]\n [0 0 3 0 0]\n [0 0 0 4 0]\n [0 0 0 0 5]]",
    solution_code: "Z = np.diag(1+np.arange(5)); print(Z)",
  },
  {
    question_id: "q002",
    question_text: "Create a 8x8 matrix and fill it with a checkerboard pattern of 0s and 1s.",
    subtopic: "Indexing and selection",
    difficulty: 24,
    expected_output: "[[0 1 0 1 0 1 0 1]\n [1 0 1 0 1 0 1 0]\n [0 1 0 1 0 1 0 1]\n [1 0 1 0 1 0 1 0]\n [0 1 0 1 0 1 0 1]\n [1 0 1 0 1 0 1 0]\n [0 1 0 1 0 1 0 1]\n [1 0 1 0 1 0 1 0]]",
    solution_code: "Z = np.zeros((8,8),dtype=int)\nZ[1::2,::2] = 1\nZ[::2,1::2] = 1\nprint(Z)",
  },
  {
    question_id: "q003",
    question_text: "Create a 3x3 identity matrix using np.eye().",
    subtopic: "Array creation",
    difficulty: 20,
    expected_output: "[[1. 0. 0.]\n [0. 1. 0.]\n [0. 0. 1.]]",
    solution_code: "print(np.eye(3))",
  },
  {
    question_id: "q004",
    question_text: "Find the row-wise argmax of a 5x5 random integer matrix (seed=42, range 0-9).",
    subtopic: "Vectorization and broadcasting",
    difficulty: 24,
    expected_output: "[0 4 2 0 3]",
    solution_code: "np.random.seed(42)\nZ = np.random.randint(0,10,(5,5))\nprint(Z.argmax(axis=1))",
  },
];
let practiceQuestionIndex = 0;

// --- Auth helpers ---

/**
 * Call when a backend API returns 401. Clears the stale token and
 * switches this session to local mode so the page stays usable.
 */
function handleExpiredToken() {
  console.warn("[practice] Token expired or invalid — falling back to local mode.");
  if (typeof setAuthState === "function") {
    setAuthState(""); // clears localStorage and resets authToken in app.js
  } else {
    localStorage.removeItem("auth_token");
    localStorage.removeItem("auth_email");
  }
  practiceMode = "local";
  // Clear any stale backend question so local mode picks a fresh one
  practiceProgress.currentQuestion = null;
}

// --- Mode detection ---

// practiceMode is set once at init based on email + environment
let practiceMode = "local"; // default fallback

function detectPracticeMode() {
  const email = (typeof authEmail === "string") ? authEmail : "";
  if (typeof getPracticeMode === "function") {
    practiceMode = getPracticeMode(email);
  } else if (typeof apiFetch === "function" && typeof authToken === "string" && !!authToken) {
    practiceMode = "backend";
  }
  console.log("[practice] mode:", practiceMode);
}

// --- Questions bank (loaded from questions.json for supabase/local modes) ---

let questionsBank = null;      // array of question objects from questions.json
let questionsBankJson = null;  // JSON string for passing to Pyodide engine

async function loadQuestionsBank() {
  if (questionsBank) return questionsBank;
  try {
    const res = await fetch("questions.json");
    if (!res.ok) throw new Error("HTTP " + res.status);
    questionsBank = await res.json();
    questionsBank = questionsBank.filter((q) => !curatedExcludedIds.has(q.id));
    questionsBankJson = JSON.stringify(questionsBank);
    console.log(`[practice] loaded ${questionsBank.length} questions from questions.json`);
  } catch (e) {
    console.warn("[practice] failed to load questions.json, using fallback pool:", e.message);
    questionsBank = null;
    questionsBankJson = null;
  }
  return questionsBank;
}

// --- Practice engine (Pyodide) for supabase/local modes ---

let practiceEngineLoaded = false;

async function loadPracticeEngine(pyodide) {
  if (practiceEngineLoaded) return;
  try {
    const res = await fetch("practice_engine.py");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const source = await res.text();
    pyodide.runPython(source);
    practiceEngineLoaded = true;
    console.log("[practice] engine loaded in Pyodide");
  } catch (e) {
    console.error("[practice] failed to load practice_engine.py:", e);
  }
}

// --- Adaptive state management for supabase/local modes ---

let adaptiveStateJson = null; // JSON string of UserPracticeState

async function loadAdaptiveState() {
  const email = (typeof authEmail === "string" && authEmail.trim()) ? authEmail.trim() : "guest";

  if (practiceMode === "supabase") {
    const sbState = await loadPracticeStateFromSupabase(email);
    if (sbState) {
      adaptiveStateJson = JSON.stringify(sbState);
      return;
    }
  }

  // Try localStorage
  const localKey = `adaptive_state_${email}`;
  const saved = localStorage.getItem(localKey);
  if (saved) {
    adaptiveStateJson = saved;
    return;
  }

  // Init fresh state via engine
  const pyodide = await initPyodide();
  if (pyodide && practiceEngineLoaded) {
    const api = pyodide.globals.get("engine_api");
    adaptiveStateJson = api.init_state(email);
  } else {
    adaptiveStateJson = null;
  }
}

async function saveAdaptiveState() {
  if (!adaptiveStateJson) return;
  const email = (typeof authEmail === "string" && authEmail.trim()) ? authEmail.trim() : "guest";
  const localKey = `adaptive_state_${email}`;

  // Always save to localStorage as backup
  localStorage.setItem(localKey, adaptiveStateJson);

  // Also save to Supabase if in supabase mode
  if (practiceMode === "supabase") {
    const stateObj = JSON.parse(adaptiveStateJson);
    await savePracticeStateToSupabase(email, stateObj);
  }
}

function getTargetDifficultyFromAdaptiveState(subtopic) {
  if (!adaptiveStateJson || !subtopic) return null;
  try {
    const state = JSON.parse(adaptiveStateJson);
    const subState = state?.subtopic_states?.[subtopic];
    const value = subState?.target_difficulty;
    return Number.isFinite(value) ? value : null;
  } catch (_err) {
    return null;
  }
}

function getEwmaFromAdaptiveState(subtopic) {
  if (!adaptiveStateJson || !subtopic) return null;
  try {
    const state = JSON.parse(adaptiveStateJson);
    const subState = state?.subtopic_states?.[subtopic];
    const value = subState?.p;
    return Number.isFinite(value) ? value : null;
  } catch (_err) {
    return null;
  }
}

// --- Progress persistence (question count, etc.) ---

const getPracticeStorageKey = () => {
  const keyEmail = (typeof authEmail === "string" && authEmail.trim()) ? authEmail.trim() : "guest";
  return `practice_progress_${keyEmail}`;
};

const loadPracticeProgress = () => {
  const saved = localStorage.getItem(getPracticeStorageKey());
  if (!saved) return null;
  try {
    return JSON.parse(saved);
  } catch (e) {
    return null;
  }
};

const savePracticeProgress = (progress) => {
  localStorage.setItem(getPracticeStorageKey(), JSON.stringify(progress));
};

// --- PracticeAPI (three-way routing) ---

const PracticeAPI = {
  currentQuestion: practiceQuestionPool[0],

  async getNextQuestion() {
    if (practiceMode === "backend") {
      // Admin on localhost — use backend API
      const res = await apiFetch("/api/practice/next-question");
      if (res.status === 401) {
        handleExpiredToken();
        // fall through to local mode below
      } else if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Failed to load next question.");
      } else {
        const data = await res.json();
        this.currentQuestion = {
          ...data,
          target_difficulty: Number.isFinite(data.target_difficulty)
            ? data.target_difficulty
            : data.difficulty,
        };
        practiceProgress.currentQuestion = this.currentQuestion;
        practiceProgress.currentQuestionId = data.question_id;
        savePracticeProgress(practiceProgress);
        return this.currentQuestion;
      }
    }

    // supabase or local mode — use Pyodide engine
    const pyodide = await initPyodide();
    const bank = await loadQuestionsBank();

    if (pyodide && practiceEngineLoaded && bank && adaptiveStateJson) {
      const api = pyodide.globals.get("engine_api");
      const resultJson = api.next_question(adaptiveStateJson, questionsBankJson);
      const result = JSON.parse(resultJson);
      adaptiveStateJson = result.state;

      if (result.question) {
        const q = result.question;
        this.currentQuestion = {
          question_id: q.id,
          question_text: q.question_text,
          subtopic: q.subtopic,
          difficulty: q.difficulty_score,
          expected_output: q.expected_output,
          solution_code: q.answer_code,
          target_difficulty: getTargetDifficultyFromAdaptiveState(q.subtopic) ?? q.difficulty_score,
        };
      }
    } else {
      // Fallback to hardcoded pool
      const completed = new Set(practiceProgress.completedQuestionIds);
      let attempts = 0;
      let nextIndex = practiceQuestionIndex;
      do {
        nextIndex = (nextIndex + 1) % practiceQuestionPool.length;
        attempts++;
        if (attempts >= practiceQuestionPool.length) break;
      } while (completed.has(practiceQuestionPool[nextIndex].question_id));
      practiceQuestionIndex = nextIndex;
      this.currentQuestion = practiceQuestionPool[practiceQuestionIndex];
    }

    practiceProgress.currentQuestionId = this.currentQuestion.question_id;
    practiceProgress.currentQuestion = this.currentQuestion;
    savePracticeProgress(practiceProgress);
    return this.currentQuestion;
  },

  async submitAnswer(questionId, userCode) {
    if (practiceMode === "backend") {
      const res = await apiFetch("/api/practice/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question_id: questionId, user_code: userCode }),
      });
      if (res.status === 401) {
        handleExpiredToken();
        // fall through to local mode below
      } else if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Failed to submit answer.");
      } else {
        return await res.json();
      }
    }

    // supabase/local — run code with Pyodide and AI judge
    const pyodide = await initPyodide();
    let actualOutput = "";
    if (pyodide) {
      pyodide.runPython("import sys\nfrom io import StringIO\nsys.stdout = StringIO()\nsys.stderr = StringIO()\nimport numpy as np\nnp.random.seed(0)");
      try {
        pyodide.runPython(userCode);
        actualOutput = pyodide.runPython("sys.stdout.getvalue()").trim();
      } catch (e) {
        actualOutput = "[ERROR]";
      } finally {
        pyodide.runPython("sys.stdout = sys.__stdout__\nsys.stderr = sys.__stderr__");
      }
    }
    const expected = (this.currentQuestion.expected_output || "").trim();

    const solCode = this.currentQuestion.solution_code || "";
    const questionText = this.currentQuestion.question_text || "";
    let correct = false;
    try {
      const verdict = await fetchAIJudge(questionText, solCode, userCode, actualOutput, expected);
      correct = verdict === "1";
    } catch (err) {
      throw new Error("AI judge unavailable. Please sign in or use backend mode.");
    }

    // Record attempt in adaptive engine
    if (practiceEngineLoaded && adaptiveStateJson) {
      const api = pyodide.globals.get("engine_api");
      adaptiveStateJson = api.submit_answer(
        adaptiveStateJson,
        this.currentQuestion.question_id,
        this.currentQuestion.subtopic,
        this.currentQuestion.difficulty || 50,
        correct
      );
    }

    return { correct, actual_output: actualOutput, expected_output: expected };
  },

  async sendFeedback(questionId, feedback) {
    if (practiceMode === "backend") {
      const res = await apiFetch("/api/practice/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question_id: questionId, feedback }),
      });
      if (res.status === 401) {
        handleExpiredToken();
        return; // feedback is non-critical, just skip it
      } else if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Failed to send feedback.");
      }
      return await res.json();
    }

    // supabase/local — apply feedback in Pyodide engine
    const pyodide = await initPyodide();
    if (pyodide && practiceEngineLoaded && adaptiveStateJson) {
      const api = pyodide.globals.get("engine_api");
      adaptiveStateJson = api.send_feedback(adaptiveStateJson, feedback);
      await saveAdaptiveState();
    }
    return { success: true };
  },

  async overrideCorrect(questionId) {
    if (practiceMode === "backend") {
      const res = await apiFetch("/api/practice/override", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question_id: questionId, correct: true }),
      });
      if (res.status === 401) {
        handleExpiredToken();
        return;
      } else if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Failed to override attempt.");
      }
      return await res.json();
    }

    const pyodide = await initPyodide();
    if (pyodide && practiceEngineLoaded && adaptiveStateJson) {
      const api = pyodide.globals.get("engine_api");
      adaptiveStateJson = api.override_attempt(adaptiveStateJson, questionId, true);
      await saveAdaptiveState();
    }
    return;
  },
};

// --- DOM elements ---

const timedModeToggle = document.getElementById("timed-mode-toggle");
const timerDisplay = document.getElementById("timer-display");
const timerControls = document.getElementById("timer-controls");
const timerPlayBtn = document.getElementById("timer-play-btn");
const questionNumber = document.getElementById("question-number");
const questionText = document.getElementById("question-text");
const subtopicLabel = document.getElementById("subtopic-label");
const difficultyLabel = document.getElementById("difficulty-label");
const targetDifficultyTitle = document.getElementById("target-difficulty-title");
const targetDifficultyFill = document.getElementById("target-difficulty-fill");
const targetDifficultyDelta = document.getElementById("target-difficulty-delta");
const targetDifficultyMarkerOld = document.getElementById("target-difficulty-marker-old");
const targetDifficultyNumberOld = document.getElementById("target-difficulty-number-old");
const targetDifficultyMarkerNew = document.getElementById("target-difficulty-marker-new");
const targetDifficultyNumberNew = document.getElementById("target-difficulty-number-new");
const practiceSubmitArea = document.getElementById("practice-submit-area");
const practiceSubmitBtn = document.getElementById("practice-submit-btn");
const ewmaAccuracy = document.getElementById("ewma-accuracy");
const ewmaAccuracyLabel = document.getElementById("ewma-accuracy-label");
const ewmaAccuracyFill = document.getElementById("ewma-accuracy-fill");
const ewmaAccuracyValue = document.getElementById("ewma-accuracy-value");
const practiceFeedbackArea = document.getElementById("practice-feedback-area");
const resultBadge = document.getElementById("result-badge");
const overrideRow = document.getElementById("override-row");
const overrideCorrectBtn = document.getElementById("override-correct-btn");
const nextProblemBtn = document.getElementById("next-problem-btn");
const solutionCode = document.getElementById("solution-code");
const aiExplanationSection = document.getElementById("ai-explanation-section");
const aiExplanationText = document.getElementById("ai-explanation-text");
const codeEditor = document.getElementById("code-editor");
const runBtn = document.getElementById("run-btn");
const outputArea = document.getElementById("output-area");
const feedbackPrompt = document.getElementById("feedback-prompt");
const feedbackButtons = document.querySelectorAll(".feedback-btn");

// --- Timer ---

let timerInterval = null;
let timerSeconds = 10; // 10 seconds default

timedModeToggle.addEventListener("change", () => {
  timerControls.classList.toggle("hidden", !timedModeToggle.checked);
  if (!timedModeToggle.checked) {
    clearInterval(timerInterval);
    timerInterval = null;
    timerSeconds = 10;
    timerDisplay.textContent = "00:10";
    timerPlayBtn.textContent = "\u25B6";
  }
});

function updateTimerDisplay() {
  const m = Math.floor(timerSeconds / 60);
  const s = timerSeconds % 60;
  timerDisplay.textContent = String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
}

timerPlayBtn.addEventListener("click", () => {
  if (timerInterval) {
    // Pause
    clearInterval(timerInterval);
    timerInterval = null;
    timerPlayBtn.textContent = "\u25B6";
  } else {
    // Start / Resume
    timerPlayBtn.textContent = "\u23F8";
    timerInterval = setInterval(() => {
      timerSeconds--;
      updateTimerDisplay();
      if (timerSeconds <= 0) {
        clearInterval(timerInterval);
        timerInterval = null;
        timerPlayBtn.textContent = "\u25B6";
        timerSeconds = 10;
        // Auto-submit
        if (!practiceSubmitArea.classList.contains("hidden")) {
          practiceSubmitBtn.click();
        }
      }
    }, 1000);
  }
});

// --- Question rendering ---

const curatedExcludedIds = new Set([9, 20, 21, 33, 39, 44, 45, 57, 88, 161, 188, 203, 221, 222, 223, 226]);
const savedProgress = loadPracticeProgress();
const staleGaussianQuestion = (q) =>
  typeof q?.question_text === "string" &&
  q.question_text.startsWith("Generate a generic 2D Gaussian-like array");

if (
  (savedProgress?.currentQuestionId && curatedExcludedIds.has(savedProgress.currentQuestionId)) ||
  (savedProgress?.currentQuestion?.question_id && curatedExcludedIds.has(savedProgress.currentQuestion.question_id)) ||
  staleGaussianQuestion(savedProgress?.currentQuestion)
) {
  savedProgress.currentQuestionId = null;
  savedProgress.currentQuestion = null;
  savePracticeProgress(savedProgress);
}
const practiceProgress = {
  currentQuestion: savedProgress?.currentQuestion || null,
  currentQuestionId: savedProgress?.currentQuestionId || practiceQuestionPool[0].question_id,
  questionCount: Number.isFinite(savedProgress?.questionCount) ? savedProgress.questionCount : 1,
  completedQuestionIds: Array.isArray(savedProgress?.completedQuestionIds)
    ? savedProgress.completedQuestionIds
    : [],
  pendingFeedback: savedProgress?.pendingFeedback || null,
  currentTargetDifficulty: Number.isFinite(savedProgress?.currentTargetDifficulty)
    ? savedProgress.currentTargetDifficulty
    : null,
  lastResultCorrect: typeof savedProgress?.lastResultCorrect === "boolean"
    ? savedProgress.lastResultCorrect
    : null,
};

if (practiceProgress.currentQuestion) {
  PracticeAPI.currentQuestion = practiceProgress.currentQuestion;
} else {
  const savedIndex = practiceQuestionPool.findIndex(
    (q) => q.question_id === practiceProgress.currentQuestionId
  );
  practiceQuestionIndex = savedIndex >= 0 ? savedIndex : 0;
  PracticeAPI.currentQuestion = practiceQuestionPool[practiceQuestionIndex];
}

let practiceQuestionCount = practiceProgress.questionCount;

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
  subtopicLabel.textContent = q.subtopic;
  difficultyLabel.textContent = "Difficulty: " + q.difficulty + " / 100";
  setTargetDifficultyInitial(getTargetDifficultyForQuestion(q));
  solutionCode.textContent = q.solution_code;
  overrideRow.classList.add("hidden");

  // Reset to pre-submit state
  practiceSubmitArea.classList.remove("hidden");
  practiceFeedbackArea.classList.add("hidden");
  practiceFeedbackArea.classList.remove("checking");
  ewmaAccuracy.classList.add("hidden");
  ewmaAccuracyFill.style.width = "0%";
  showFeedbackButtons();

  // Reset AI explanation
  aiExplanationSection.classList.add("hidden");
  aiExplanationText.textContent = "";

  // Reset timer for next question if timed mode is on
  if (timedModeToggle.checked) {
    clearInterval(timerInterval);
    timerInterval = null;
    timerSeconds = 10;
    updateTimerDisplay();
    timerPlayBtn.textContent = "\u25B6";
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

function clampDifficulty(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

function formatDifficulty(value) {
  if (!Number.isFinite(value)) return "--";
  return value.toFixed(1);
}

function getTargetDifficultyForQuestion(q) {
  if (q && Number.isFinite(q.target_difficulty)) return q.target_difficulty;
  const fromState = getTargetDifficultyFromAdaptiveState(q?.subtopic);
  if (Number.isFinite(fromState)) return fromState;
  return Number.isFinite(q?.difficulty) ? q.difficulty : 0;
}

function setTargetDifficultyInitial(targetDifficulty) {
  const clamped = clampDifficulty(targetDifficulty);
  targetDifficultyTitle.textContent = `Old target difficulty = ${formatDifficulty(clamped)}`;
  targetDifficultyFill.style.width = `${clamped}%`;
  targetDifficultyDelta.classList.add("hidden");
  targetDifficultyDelta.style.width = "0%";
  targetDifficultyMarkerOld.style.left = `${clamped}%`;
  targetDifficultyNumberOld.textContent = formatDifficulty(clamped);
  targetDifficultyMarkerNew.classList.add("hidden");
}

function setTargetDifficultyFinal(oldTarget, newTarget) {
  const oldClamped = clampDifficulty(oldTarget);
  const newClamped = clampDifficulty(newTarget);
  const diff = Math.abs(newClamped - oldClamped);
  targetDifficultyTitle.textContent = `New target difficulty = ${formatDifficulty(newClamped)}`;
  targetDifficultyFill.style.width = `${newClamped}%`;
  targetDifficultyMarkerOld.style.left = `${oldClamped}%`;
  targetDifficultyNumberOld.textContent = formatDifficulty(oldClamped);
  targetDifficultyMarkerNew.classList.remove("hidden");
  targetDifficultyMarkerNew.style.left = `${newClamped}%`;
  targetDifficultyNumberNew.textContent = formatDifficulty(newClamped);

  if (diff < 0.01) {
    targetDifficultyDelta.classList.add("hidden");
    targetDifficultyDelta.style.width = "0%";
    return;
  }
  const left = Math.min(oldClamped, newClamped);
  targetDifficultyDelta.style.left = `${left}%`;
  targetDifficultyDelta.style.width = `${diff}%`;
  targetDifficultyDelta.classList.remove("hidden");
  targetDifficultyDelta.classList.toggle("up", newClamped > oldClamped);
  targetDifficultyDelta.classList.toggle("down", newClamped < oldClamped);
}

function animateTargetDifficulty(oldTarget, newTarget, onComplete) {
  const oldClamped = clampDifficulty(oldTarget);
  const newClamped = clampDifficulty(newTarget);
  const isUp = newClamped > oldClamped;
  const start = performance.now();
  const duration = 900;

  targetDifficultyMarkerNew.classList.remove("hidden");
  targetDifficultyMarkerNew.style.left = `${oldClamped}%`;
  targetDifficultyNumberNew.textContent = formatDifficulty(oldClamped);
  targetDifficultyTitle.textContent = `Old target difficulty = ${formatDifficulty(oldClamped)}`;
  targetDifficultyDelta.classList.toggle("up", isUp);
  targetDifficultyDelta.classList.toggle("down", !isUp && newClamped !== oldClamped);
  targetDifficultyDelta.classList.remove("hidden");

  const tick = (now) => {
    const progress = Math.min((now - start) / duration, 1);
    const value = oldClamped + (newClamped - oldClamped) * progress;
    targetDifficultyFill.style.width = `${value}%`;
    targetDifficultyMarkerNew.style.left = `${value}%`;
    targetDifficultyNumberNew.textContent = formatDifficulty(value);
    const left = Math.min(oldClamped, value);
    const width = Math.abs(value - oldClamped);
    targetDifficultyDelta.style.left = `${left}%`;
    targetDifficultyDelta.style.width = `${width}%`;

    if (progress < 1) {
      requestAnimationFrame(tick);
      return;
    }
    targetDifficultyTitle.textContent = `New target difficulty = ${formatDifficulty(newClamped)}`;
    targetDifficultyFill.style.width = `${newClamped}%`;
    targetDifficultyMarkerNew.style.left = `${newClamped}%`;
    targetDifficultyNumberNew.textContent = formatDifficulty(newClamped);
    if (Math.abs(newClamped - oldClamped) < 0.01) {
      targetDifficultyDelta.classList.add("hidden");
      targetDifficultyDelta.style.width = "0%";
    }
    if (typeof onComplete === "function") onComplete();
  };

  requestAnimationFrame(tick);
}

function showFeedbackButtons() {
  feedbackButtons.forEach((btn) => btn.classList.remove("hidden"));
  nextProblemBtn.classList.add("hidden");
}

function showNextProblemButton() {
  feedbackButtons.forEach((btn) => btn.classList.add("hidden"));
  nextProblemBtn.classList.remove("hidden");
}

function shortSubtopicName(subtopic) {
  if (!subtopic) return subtopic;
  const colon = subtopic.indexOf(": ");
  return colon >= 0 ? subtopic.slice(colon + 2) : subtopic;
}

function showEwmaAccuracy(p, subtopic) {
  if (!Number.isFinite(p)) return;
  const pct = Math.round(p * 1000) / 10; // one decimal place
  ewmaAccuracyLabel.textContent = shortSubtopicName(subtopic) + " accuracy";
  ewmaAccuracyValue.textContent = pct.toFixed(1) + "%";
  ewmaAccuracy.classList.remove("hidden");
  // Trigger CSS transition by setting width after a microtask
  requestAnimationFrame(() => {
    ewmaAccuracyFill.style.width = pct + "%";
  });
}

function applyPendingFeedbackState(pending) {
  practiceSubmitArea.classList.add("hidden");
  practiceFeedbackArea.classList.remove("hidden");
  applyResult(!!pending.correct);
  overrideRow.classList.add("hidden");
  showNextProblemButton();
  setTargetDifficultyFinal(pending.oldTarget, pending.newTarget);
  if (Number.isFinite(pending.pAfter)) {
    showEwmaAccuracy(pending.pAfter, pending.subtopic);
  }
}

// --- AI helpers ---

// Apply correct/incorrect result to the feedback area UI.
function applyResult(correct) {
  resultBadge.textContent = correct ? "Correct" : "Incorrect";
  resultBadge.className = "result-badge " + (correct ? "correct" : "incorrect");
  overrideRow.classList.toggle("hidden", correct);
  practiceFeedbackArea.classList.remove("checking");
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

// Fetch AI explanation and update the explanation element when done.
async function fetchAIExplanation(questionText, solCode, userCode, actualOutput, expectedOutput) {
  try {
    const res = await apiFetch("/api/practice/ai-explanation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question_text: questionText,
        solution_code: solCode,
        user_code: userCode,
        actual_output: actualOutput,
        expected_output: expectedOutput,
      }),
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      aiExplanationText.textContent = "Could not load explanation." + (detail ? "\n" + detail : "");
      return;
    }
    const data = await res.json();
    aiExplanationText.textContent = data.explanation || "No explanation available.";
  } catch (e) {
    aiExplanationText.textContent = "Could not load explanation.";
  }
}

// Fetch AI judge verdict ("1" = correct, "0" = incorrect).
async function fetchAIJudge(questionText, solCode, userCode, actualOutput, expectedOutput) {
  const payload = {
    question_text: questionText,
    solution_code: solCode,
    user_code: userCode,
    actual_output: actualOutput,
    expected_output: expectedOutput,
  };
  const res = typeof apiFetch === "function"
    ? await apiFetch("/api/practice/ai-judge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
    : await fetch("/api/practice/ai-judge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
  if (!res.ok) throw new Error("Judge request failed");
  const data = await res.json();
  return data.verdict; // "0" or "1"
}

// --- Submit ---

practiceSubmitBtn.addEventListener("click", async () => {
  const q = PracticeAPI.currentQuestion;
  const userCode = codeEditor.value;
  practiceSubmitBtn.disabled = true;
  let result;
  try {
    result = await PracticeAPI.submitAnswer(q.question_id, userCode);
  } catch (err) {
    outputArea.textContent = "Submit failed: " + err.message;
    practiceSubmitBtn.disabled = false;
    return;
  }

  const solCode = q.solution_code || result.solution_code || "";
  const actualOutput = result.actual_output || "";
  const expectedOutput = result.expected_output || q.expected_output || "";

  solutionCode.textContent = solCode;
  practiceSubmitArea.classList.add("hidden");
  practiceFeedbackArea.classList.remove("hidden");

  applyResult(result.correct);
  practiceProgress.lastResultCorrect = result.correct;
  practiceProgress.currentTargetDifficulty = getTargetDifficultyForQuestion(q);
  savePracticeProgress(practiceProgress);
  if (practiceMode === "backend" || practiceMode === "supabase") {
    aiExplanationSection.classList.remove("hidden");
    aiExplanationText.textContent = "Loading explanation...";
    fetchAIExplanation(q.question_text, solCode, userCode, actualOutput, expectedOutput);
  }
});

overrideCorrectBtn.addEventListener("click", async () => {
  const q = PracticeAPI.currentQuestion;
  await PracticeAPI.overrideCorrect(q.question_id);

  resultBadge.textContent = "Correct";
  resultBadge.className = "result-badge correct";
  feedbackPrompt.textContent = "Nailed it! How hard should we go next?";
  const labels = ["Inch it up", "Rev the engine", "Full throttle"];
  feedbackButtons.forEach((btn, i) => { btn.textContent = labels[i]; });
  overrideRow.classList.add("hidden");
  practiceProgress.lastResultCorrect = true;
  savePracticeProgress(practiceProgress);
});

// --- Feedback ---

feedbackButtons.forEach((btn) => {
  btn.addEventListener("click", async () => {
    const feedback = btn.dataset.feedback;
    const q = PracticeAPI.currentQuestion;
    const oldTarget = Number.isFinite(practiceProgress.currentTargetDifficulty)
      ? practiceProgress.currentTargetDifficulty
      : getTargetDifficultyForQuestion(q);
    const response = await PracticeAPI.sendFeedback(q.question_id, feedback);
    const backendTarget = Number.isFinite(response?.target_difficulty_after)
      ? response.target_difficulty_after
      : null;
    const newTarget = Number.isFinite(backendTarget)
      ? backendTarget
      : (getTargetDifficultyFromAdaptiveState(q.subtopic) ?? oldTarget);

    const pAfter = Number.isFinite(response?.p_after)
      ? response.p_after
      : getEwmaFromAdaptiveState(q.subtopic);

    if (!practiceProgress.completedQuestionIds.includes(q.question_id)) {
      practiceProgress.completedQuestionIds.push(q.question_id);
    }

    showNextProblemButton();
    animateTargetDifficulty(oldTarget, newTarget, () => {
      setTargetDifficultyFinal(oldTarget, newTarget);
    });
    if (Number.isFinite(pAfter)) {
      showEwmaAccuracy(pAfter, q.subtopic);
    }

    practiceProgress.pendingFeedback = {
      questionId: q.question_id,
      subtopic: q.subtopic,
      oldTarget,
      newTarget,
      correct: !!practiceProgress.lastResultCorrect,
      pAfter,
    };
    savePracticeProgress(practiceProgress);
  });
});

nextProblemBtn.addEventListener("click", async () => {
  practiceProgress.currentQuestion = null;
  practiceProgress.pendingFeedback = null;
  practiceProgress.currentTargetDifficulty = null;
  practiceProgress.lastResultCorrect = null;

  // Reset to pre-submit state (ready for next question)
  practiceSubmitArea.classList.remove("hidden");
  practiceFeedbackArea.classList.add("hidden");
  ewmaAccuracy.classList.add("hidden");
  ewmaAccuracyFill.style.width = "0%";
  showFeedbackButtons();

  // Reset code editor
  codeEditor.value = "import numpy as np\nnp.random.seed(0)\n\n# Write your solution here\n";
  outputArea.textContent = "";

  // Load next question
  const nextQ = await PracticeAPI.getNextQuestion();
  const nextCount = practiceQuestionCount + 1;
  practiceProgress.questionCount = nextCount;
  savePracticeProgress(practiceProgress);
  renderQuestion(nextQ, nextCount);
});

// --- Pyodide code runner ---

let pyodideInstance = null;
let pyodideLoading = false;

const normalizeOutput = (value) => (value || "").replace(/\r\n/g, "\n").trim();

async function initPyodide() {
  if (pyodideInstance) return pyodideInstance;
  if (pyodideLoading) {
    while (pyodideLoading) {
      await new Promise((r) => setTimeout(r, 100));
    }
    return pyodideInstance;
  }
  pyodideLoading = true;
  outputArea.textContent = "Loading Python...";
  try {
    pyodideInstance = await loadPyodide();
    await pyodideInstance.loadPackage("numpy");
    outputArea.textContent = "";
  } catch (e) {
    outputArea.textContent = "Failed to load Python: " + e.message;
  }
  pyodideLoading = false;
  return pyodideInstance;
}

runBtn.addEventListener("click", async () => {
  runBtn.disabled = true;
  runBtn.textContent = "Running...";
  outputArea.textContent = "";

  try {
    let actualOutput = "";
    let runFailed = false;

    let useLocalPyodide = practiceMode !== "backend";

    if (practiceMode === "backend") {
      try {
        const res = await apiFetch("/api/practice/run-code", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ code: codeEditor.value }),
        });
        if (res.status === 401) {
          handleExpiredToken();
          useLocalPyodide = true; // fall back to in-browser Pyodide
        } else if (!res.ok) {
          const detail = await res.text();
          outputArea.textContent = detail || "Failed to run code.";
          runFailed = true;
        } else {
          const data = await res.json();
          const stdout = normalizeOutput(data.stdout);
          const stderr = normalizeOutput(data.stderr);
          actualOutput = stdout;
          outputArea.textContent = stdout || stderr || "(No output)";
          if (stderr) {
            runFailed = true;
          }
        }
      } catch (_fetchErr) {
        // Backend unreachable — fall back to in-browser Pyodide
        useLocalPyodide = true;
      }
    }

    if (useLocalPyodide) {
      const pyodide = await initPyodide();
      if (!pyodide) {
        runBtn.disabled = false;
        runBtn.textContent = "Run";
        return;
      }

      // Redirect stdout to capture print output
      pyodide.runPython(`
import sys
from io import StringIO
sys.stdout = StringIO()
sys.stderr = StringIO()
import numpy as np
np.random.seed(0)
`);

      try {
        pyodide.runPython(codeEditor.value);
        const stdout = normalizeOutput(pyodide.runPython("sys.stdout.getvalue()"));
        const stderr = normalizeOutput(pyodide.runPython("sys.stderr.getvalue()"));
        actualOutput = stdout;
        let output = stdout || "";
        if (stderr) {
          output += (output ? "\n" : "") + stderr;
          runFailed = true;
        }
        outputArea.textContent = output || "(No output)";
      } catch (pyErr) {
        const stderr = normalizeOutput(pyodide.runPython("sys.stderr.getvalue()"));
        outputArea.textContent = stderr || pyErr.message;
        runFailed = true;
      } finally {
        // Reset stdout/stderr
        pyodide.runPython(`
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
`);
      }
    }

  } catch (e) {
    outputArea.textContent = "Error: " + e.message;
  }

  runBtn.disabled = false;
  runBtn.textContent = "Run";
});

// --- Initialize ---

const initPractice = async () => {
  detectPracticeMode();

  // For supabase/local modes, load engine + questions + state
  if (practiceMode !== "backend") {
    const pyodide = await initPyodide();
    if (pyodide) {
      await loadPracticeEngine(pyodide);
    }
    await loadQuestionsBank();
    await loadAdaptiveState();
  }

  if (practiceProgress.currentQuestion) {
    savePracticeProgress(practiceProgress);
    renderQuestion(PracticeAPI.currentQuestion, practiceQuestionCount);
    return;
  }
  const nextQ = await PracticeAPI.getNextQuestion();
  savePracticeProgress(practiceProgress);
  renderQuestion(nextQ, practiceQuestionCount);
};

initPractice();
