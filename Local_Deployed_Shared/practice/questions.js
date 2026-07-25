/* ================================================================
   PRACTICE QUESTIONS — questions.json loader
   ================================================================ */

let questionsBank = null; // array of question objects from questions.json
let questionsBankJson = null; // JSON string for passing to Pyodide engine
let questionsSourceIndex = null; // Map from question id -> ARENA prereq source record
let questionsSourceJson = null; // JSON string for passing to Pyodide engine when needed
const PRACTICE_ARENA_CONTENT_BASE =
  "/content/ARENA_5.0-main/chapter0_fundamentals/exercises/part0_prereqs/";
const PRACTICE_PREREQ_NOTEBOOK_URL =
  "/arena-book/chapter0_fundamentals/exercises/part0_prereqs/0.0_Prerequisites_exercises.html";
const PRACTICE_PREREQ_NOTEBOOK_IPYNB_URL =
  `${PRACTICE_ARENA_CONTENT_BASE}0.0_Prerequisites_exercises.ipynb`;
const PRACTICE_PREREQ_NUMBERS_URL = `${PRACTICE_ARENA_CONTENT_BASE}numbers.npy`;
const PRACTICE_PREREQ_NOTEBOOK_IPYNB_FALLBACKS = [
  PRACTICE_PREREQ_NOTEBOOK_IPYNB_URL,
  "/content/ARENA_4.0-main/chapter0_fundamentals/exercises/part0_prereqs/0.0_Prerequisites_exercises.ipynb",
  "/content/ARENA_3.0-main/chapter0_fundamentals/exercises/part0_prereqs/0.0_Prerequisites_exercises.ipynb",
  "/arena-book/_sources/chapter0_fundamentals/exercises/part0_prereqs/0.0_Prerequisites_exercises.ipynb",
];

let practiceNotebookPromise = null;
let practiceNotebookData = null;

async function loadQuestionsBank() {
  if (questionsBank) return questionsBank;
  try {
    await loadQuestionsSourceIndex();
    const res = await fetch(`questions.json?v=20260428-rewrite`, { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    questionsBank = await res.json();
    questionsBank = questionsBank.filter((q) => !curatedExcludedIds.has(q.id));
    if (questionsSourceIndex) {
      questionsBank = questionsBank.map((q) => {
        const source = questionsSourceIndex.get(Number(q.id));
        if (!source) return q;
        return {
          ...q,
          arena_source_path: source?.source?.path || null,
          arena_source_cell_index: Number.isFinite(source?.source?.cell_index) ? source.source.cell_index : null,
          arena_source_type: source?.source?.type || null,
          arena_notebook_url: buildArenaNotebookUrl(source?.source?.path),
          arena_function_names: Array.isArray(source?.exercise?.function_names)
            ? source.exercise.function_names.filter(Boolean)
            : [],
          arena_curriculum: source?.curriculum || null,
        };
      });
    }
    questionsBankJson = JSON.stringify(questionsBank);
    console.log(`[practice] loaded ${questionsBank.length} questions from questions.json`);
  } catch (e) {
    console.warn("[practice] failed to load questions.json, using fallback pool:", e.message);
    questionsBank = null;
    questionsBankJson = null;
  }
  return questionsBank;
}

async function loadQuestionsSourceIndex() {
  if (questionsSourceIndex) return questionsSourceIndex;
  try {
    const res = await fetch(`questions_structured.json?v=20260513-links`, { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    const index = new Map();
    for (const entry of data) {
      index.set(Number(entry.id), entry);
    }
    questionsSourceIndex = index;
    questionsSourceJson = JSON.stringify(data);
    console.log(`[practice] loaded ${data.length} structured question records`);
  } catch (e) {
    console.warn("[practice] failed to load questions_structured.json:", e.message);
    questionsSourceIndex = null;
    questionsSourceJson = null;
  }
  return questionsSourceIndex;
}

function buildArenaNotebookUrl(sourcePath) {
  if (!sourcePath || typeof sourcePath !== "string") return null;
  const match = sourcePath.match(/(?:^|\/)ARENA_[^/]+-main\/(.+)\.ipynb$/);
  if (!match) return null;
  return `/arena-book/${match[1]}.html`;
}

function buildArenaNotebookTextFragmentUrl(notebookUrl, text) {
  if (!notebookUrl || !text) return notebookUrl;
  return `${notebookUrl}#:~:text=${encodeURIComponent(text)}`;
}

function getQuestionSourceRecord(questionId) {
  if (!questionsSourceIndex) return null;
  return questionsSourceIndex.get(Number(questionId)) || null;
}

function getQuestionNotebookMetadata(questionId) {
  const record = getQuestionSourceRecord(questionId);
  if (!record) return null;
  const notebookUrl = buildArenaNotebookUrl(record?.source?.path) || PRACTICE_PREREQ_NOTEBOOK_URL;
  const functionNames = Array.isArray(record?.exercise?.function_names)
    ? record.exercise.function_names.filter(Boolean)
    : record?.exercise?.function_name
      ? [record.exercise.function_name]
      : [];
  return {
    notebookUrl,
    functionNames,
    curriculum: record?.curriculum || null,
    cellIndex: Number.isFinite(record?.source?.cell_index) ? record.source.cell_index : null,
    sourceType: record?.source?.type || null,
    sourcePath: record?.source?.path || null,
  };
}

async function loadPracticeNotebookData() {
  if (practiceNotebookData) return practiceNotebookData;
  if (!practiceNotebookPromise) {
    practiceNotebookPromise = (async () => {
      let lastErr = null;
      for (const path of PRACTICE_PREREQ_NOTEBOOK_IPYNB_FALLBACKS) {
        try {
          const res = await fetch(path, { cache: "no-store" });
          if (!res.ok) {
            lastErr = new Error("HTTP " + res.status + " for " + path);
            continue;
          }
          return await res.json();
        } catch (err) {
          lastErr = err;
        }
      }
      throw lastErr || new Error("Failed to fetch prereq notebook");
    })().then((data) => {
      practiceNotebookData = data;
      return data;
    }).catch((err) => {
      practiceNotebookPromise = null;
      throw err;
    });
  }
  return practiceNotebookPromise;
}

function getNotebookCellSource(cell) {
  return Array.isArray(cell?.source) ? cell.source.join("") : "";
}

function getNotebookCellSourceLines(cell) {
  return getNotebookCellSource(cell).split(/\r?\n/);
}

function getNotebookCellText(cell) {
  return getNotebookCellSource(cell).trim();
}

function getNotebookCellIndex(cells, targetCell) {
  return Math.max(0, cells.indexOf(targetCell));
}

function extractNotebookHelperItems(question, notebook) {
  const cells = Array.isArray(notebook?.cells) ? notebook.cells : [];
  const items = [];

  const importCell = cells.find((cell) => {
    const source = getNotebookCellSource(cell);
    return cell?.cell_type === "code" && /import\s+einops\b/.test(source);
  });
  if (question?.primary_library === "einops" || question?.primary_library === "einops.einsum" || question?.topic === "Einops" || question?.topic === "Einsum") {
    if (importCell) {
      for (const line of getNotebookCellSourceLines(importCell)) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        if (!/^(import|from)\s+/.test(trimmed)) continue;
        if (/^import\s+numpy\b/.test(trimmed)) continue;
        items.push({
          label: trimmed,
          code: trimmed,
          note: "Notebook cell " + (getNotebookCellIndex(cells, importCell) + 1),
          context: "",
        });
      }
    }
  }

  if (questionNeedsArenaArray(question)) {
    const arrayCell = cells.find((cell) => {
      const source = getNotebookCellSource(cell);
      return cell?.cell_type === "code" && /numbers\.npy|delta_numbers\.npy/.test(source);
    });
    if (arrayCell) {
      const sourceLines = getNotebookCellSourceLines(arrayCell);
      const loadLine = sourceLines.find((line) => /numbers\.npy|delta_numbers\.npy/.test(line));
      if (loadLine) {
        const nextCell = cells[getNotebookCellIndex(cells, arrayCell) + 1];
        const context =
          nextCell?.cell_type === "markdown" ? getNotebookCellText(nextCell).slice(0, 500) : "";
        items.push({
          label: "numbers.npy",
          code: loadLine.trim(),
          note: "Notebook cell " + (getNotebookCellIndex(cells, arrayCell) + 1),
          context,
          kind: "arena-array",
          dataUrl: PRACTICE_PREREQ_NUMBERS_URL,
        });
      }
    }
  }

  return items;
}

async function getNotebookHelperItems(question) {
  try {
    const notebook = await loadPracticeNotebookData();
    return extractNotebookHelperItems(question, notebook);
  } catch (err) {
    console.warn("[practice] failed to load prereq notebook helpers:", err.message);
    return [];
  }
}

function getPracticeEligibleQuestions() {
  if (!Array.isArray(questionsBank)) return null;
  // Single-KC practice (concept-graph maximize) pins the queue to one subtopic.
  // Deliberately bypasses isSubtopicEnabled: the learner clicked this concept,
  // so a stats-page toggle shouldn't empty the queue underneath them.
  const focus = window.__kcFocusSubtopics;
  if (Array.isArray(focus) && focus.length) {
    const focused = questionsBank.filter((q) => focus.includes(q.subtopic));
    if (focused.length) return focused;
  }
  if (typeof isSubtopicEnabled !== "function") return questionsBank;
  return questionsBank.filter((q) => isSubtopicEnabled(q.subtopic, q.topic || ""));
}

function isPracticeQuestionAllowed(question) {
  if (!question) return false;
  const focus = window.__kcFocusSubtopics;
  if (Array.isArray(focus) && focus.length) return focus.includes(question.subtopic);
  if (typeof isSubtopicEnabled !== "function") return true;
  return isSubtopicEnabled(question.subtopic, question.topic || "");
}

// Bank record → the shape PracticeAPI.currentQuestion / renderQuestion expect.
// Single source of truth for that mapping: the adaptive queue (api.js) and the
// single-KC lesson ladder (lessons.js) must agree, or a faded item would grade
// against different fields than the same question served by the queue.
// `overrides` lets the faded tier swap in its blanked starter_code.
function buildPracticeQuestionFromBank(q, overrides = {}) {
  if (!q) return null;
  return {
    question_id: q.id,
    question_text: q.question_text,
    topic: q.topic || "",
    subtopic: q.subtopic,
    difficulty: q.difficulty_score,
    expected_output: q.expected_output,
    solution_code: q.answer_code,
    primary_library: q.primary_library || null,
    task_type: q.task_type || null,
    expected_artifact_type: q.expected_artifact_type || "stdout",
    supports_visual_output: !!q.supports_visual_output,
    function_name: q.function_name || null,
    starter_code: q.starter_code || null,
    test_cases: Array.isArray(q.test_cases) ? q.test_cases : [],
    submission_mode: q.submission_mode || "stdout",
    target_difficulty:
      (typeof getTargetDifficultyFromAdaptiveState === "function"
        ? getTargetDifficultyFromAdaptiveState(q.subtopic)
        : null) ?? q.difficulty_score,
    ...overrides,
  };
}

function getQuestionFromBank(questionId) {
  if (!Array.isArray(questionsBank)) return null;
  const numericId = Number(questionId);
  return questionsBank.find((q) => q.id === numericId) || null;
}

function hydrateSavedPracticeQuestionFromBank(savedQuestion) {
  if (!savedQuestion) return null;
  const bankQ = getQuestionFromBank(savedQuestion.question_id || savedQuestion.id);
  if (!bankQ) return savedQuestion;

  const artifactChanged =
    (savedQuestion.question_text || "") !== (bankQ.question_text || "") ||
    (savedQuestion.solution_code || "") !== (bankQ.answer_code || "") ||
    (savedQuestion.starter_code || "") !== (bankQ.starter_code || "") ||
    JSON.stringify(savedQuestion.test_cases || []) !== JSON.stringify(bankQ.test_cases || []) ||
    !!savedQuestion.supports_visual_output !== !!bankQ.supports_visual_output;

  return {
    ...savedQuestion,
    question_id: bankQ.id,
    question_text: bankQ.question_text,
    topic: bankQ.topic || "",
    subtopic: bankQ.subtopic,
    difficulty: bankQ.difficulty_score,
    expected_output: bankQ.expected_output,
    solution_code: bankQ.answer_code,
    primary_library: bankQ.primary_library || null,
    task_type: bankQ.task_type || null,
    expected_artifact_type: bankQ.expected_artifact_type || "stdout",
    supports_visual_output: !!bankQ.supports_visual_output,
    function_name: bankQ.function_name || null,
    starter_code: bankQ.starter_code || null,
    test_cases: Array.isArray(bankQ.test_cases) ? bankQ.test_cases : [],
    submission_mode: bankQ.submission_mode || "stdout",
    _artifactChanged: artifactChanged,
  };
}
