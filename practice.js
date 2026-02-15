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
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Failed to load next question.");
      }
      const data = await res.json();
      this.currentQuestion = data;
      practiceProgress.currentQuestion = data;
      practiceProgress.currentQuestionId = data.question_id;
      savePracticeProgress(practiceProgress);
      return data;
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
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Failed to submit answer.");
      }
      return await res.json();
    }

    // supabase/local — run code with Pyodide and compare output
    const pyodide = await initPyodide();
    let actualOutput = "";
    if (pyodide) {
      pyodide.runPython("import sys\nfrom io import StringIO\nsys.stdout = StringIO()\nsys.stderr = StringIO()");
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
    const correct = actualOutput === expected;

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
      if (!res.ok) {
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
const practiceSubmitArea = document.getElementById("practice-submit-area");
const practiceSubmitBtn = document.getElementById("practice-submit-btn");
const practiceFeedbackArea = document.getElementById("practice-feedback-area");
const resultBadge = document.getElementById("result-badge");
const solutionCode = document.getElementById("solution-code");
const codeEditor = document.getElementById("code-editor");
const runBtn = document.getElementById("run-btn");
const outputArea = document.getElementById("output-area");
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

const savedProgress = loadPracticeProgress();
const practiceProgress = {
  currentQuestion: savedProgress?.currentQuestion || null,
  currentQuestionId: savedProgress?.currentQuestionId || practiceQuestionPool[0].question_id,
  questionCount: Number.isFinite(savedProgress?.questionCount) ? savedProgress.questionCount : 1,
  completedQuestionIds: Array.isArray(savedProgress?.completedQuestionIds)
    ? savedProgress.completedQuestionIds
    : [],
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
  practiceQuestionCount = count;
  questionNumber.textContent = "Question " + practiceQuestionCount;
  questionText.textContent = q.question_text;
  subtopicLabel.textContent = q.subtopic;
  difficultyLabel.textContent = "Difficulty: " + q.difficulty + " / 100";
  solutionCode.textContent = q.solution_code;

  // Reset to pre-submit state
  practiceSubmitArea.classList.remove("hidden");
  practiceFeedbackArea.classList.add("hidden");

  // Reset timer for next question if timed mode is on
  if (timedModeToggle.checked) {
    clearInterval(timerInterval);
    timerInterval = null;
    timerSeconds = 10;
    updateTimerDisplay();
    timerPlayBtn.textContent = "\u25B6";
  }
}

// --- Submit ---

practiceSubmitBtn.addEventListener("click", async () => {
  const q = PracticeAPI.currentQuestion;
  const result = await PracticeAPI.submitAnswer(q.question_id, codeEditor.value);

  // Show feedback area
  resultBadge.textContent = result.correct ? "Correct" : "Incorrect";
  resultBadge.className = "result-badge " + (result.correct ? "correct" : "incorrect");
  practiceSubmitArea.classList.add("hidden");
  practiceFeedbackArea.classList.remove("hidden");
});

// --- Feedback ---

feedbackButtons.forEach((btn) => {
  btn.addEventListener("click", async () => {
    const feedback = btn.dataset.feedback;
    const q = PracticeAPI.currentQuestion;
    await PracticeAPI.sendFeedback(q.question_id, feedback);
    practiceProgress.currentQuestion = null;
    if (!practiceProgress.completedQuestionIds.includes(q.question_id)) {
      practiceProgress.completedQuestionIds.push(q.question_id);
    }

    // Reset to pre-submit state (ready for next question)
    practiceSubmitArea.classList.remove("hidden");
    practiceFeedbackArea.classList.add("hidden");

    // Reset code editor
    codeEditor.value = "import numpy as np\n\n# Write your solution here\n";
    outputArea.textContent = "";

    // Load next question
    const nextQ = await PracticeAPI.getNextQuestion();
    const nextCount = practiceQuestionCount + 1;
    practiceProgress.questionCount = nextCount;
    savePracticeProgress(practiceProgress);
    renderQuestion(nextQ, nextCount);
  });
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
    const expected = normalizeOutput(PracticeAPI.currentQuestion?.expected_output);
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
        if (!res.ok) {
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

    if (!runFailed && expected && normalizeOutput(actualOutput) === expected) {
      practiceSubmitBtn.click();
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
