/* ================================================================
   PRACTICE QUESTIONS — questions.json loader
   ================================================================ */

let questionsBank = null; // array of question objects from questions.json
let questionsBankJson = null; // JSON string for passing to Pyodide engine
// Ids tagged to at least one lesson-graph KC (lessons/qmatrix_tags.json).
// null = q-matrix unavailable, see servableQuestions().
let kcTaggedIds = null;
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
    await loadKcTaggedIds();
    questionsBankJson = JSON.stringify(servableQuestions() || questionsBank);
    const parked = kcTaggedIds ? questionsBank.length - (servableQuestions() || []).length : 0;
    console.log(
      `[practice] loaded ${questionsBank.length} questions from questions.json` +
        (parked ? ` (${parked} parked: no lesson-graph KC)` : "")
    );
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

// Q391's old API payload disclosed the exact answer in its prompt/docstring.
// Read the corrected prompt from the shipped bank while the backend catches up.
function repairKnownQuestionContent(question) {
  if (Number(question?.question_id) !== 391) return question;
  const authored = getQuestionSourceRecord(391)?.exercise;
  if (authored) question.question_text = authored.question_text;
  else if (question.question_text) {
    question.question_text = question.question_text.replace(" via 'c h w -> c (h w)'", "");
  }
  if (question.starter_code) {
    question.starter_code = question.starter_code.replace(
      "Return shape (c, h*w) via 'c h w -> c (h w)'.",
      "Return one flattened pixel row per channel.",
    );
  }
  return question;
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

/* The lesson graph is being validated chapter by chapter. A question with no
   target KC has no validated structure to be scheduled against, so it is
   PARKED: still in the bank and still resolvable by id (saved/in-flight
   questions keep hydrating), but never selected. Mirrors the backend's
   lessons.kc_only_serving() — see This-Directory-Only/backend/app/lessons.py.

   Failure mode is deliberate: if qmatrix_tags.json cannot be fetched,
   kcTaggedIds stays null and NOTHING is filtered. An unfiltered offline queue
   is a smaller harm than an empty one, and the backend enforces the parking
   authoritatively for signed-in practice, which is the path that matters. */
async function loadKcTaggedIds() {
  if (kcTaggedIds) return kcTaggedIds;
  try {
    const res = await fetch("lessons/qmatrix_tags.json", { cache: "force-cache" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const qmatrix = await res.json();
    const ids = new Set();
    for (const [qid, tags] of Object.entries(qmatrix || {})) {
      if (tags?.target_kcs?.length) ids.add(Number(qid));
    }
    // An EMPTY set is a real answer ("nothing is tagged yet") and must park
    // everything. Only a fetch/parse failure leaves kcTaggedIds null, which is
    // what disables parking — see the note above. Conflating the two would
    // reopen the whole bank the moment the q-matrix was emptied.
    kcTaggedIds = ids;
  } catch (e) {
    kcTaggedIds = null;
    console.warn("[practice] q-matrix unavailable — KC parking disabled locally:", e.message);
  }
  return kcTaggedIds;
}

function servableQuestions() {
  if (!Array.isArray(questionsBank)) return null;
  if (!kcTaggedIds) return questionsBank;
  return questionsBank.filter((q) => kcTaggedIds.has(Number(q.id)));
}

function isKcServable(question) {
  if (!kcTaggedIds) return true;
  return kcTaggedIds.has(Number(question?.id));
}

function getPracticeEligibleQuestions() {
  if (!Array.isArray(questionsBank)) return null;
  const bank = servableQuestions() || questionsBank;
  // Single-KC practice (concept-graph maximize) pins the queue to one subtopic.
  // Deliberately bypasses isSubtopicEnabled: the learner clicked this concept,
  // so a saved per-subtopic toggle shouldn't empty the queue underneath them.
  const focus = window.__kcFocusSubtopics;
  if (Array.isArray(focus) && focus.length) {
    const focused = bank.filter((q) => focus.includes(q.subtopic));
    if (focused.length) return focused;
  }
  if (typeof isSubtopicEnabled !== "function") return bank;
  const enabled = bank.filter((q) => isSubtopicEnabled(q.subtopic, q.topic || ""));
  // The only UI that could DISABLE a subtopic was the Statistics tab's Advanced
  // table, removed 2026-07-31. Anyone still carrying disabled flags in
  // `delta_drills_weights` has no way to switch them back on, so a filter that
  // empties the bank would be a permanent dead end. Ignore the flags in that
  // case rather than serving nothing.
  if (!enabled.length && bank.length) return bank;
  return enabled;
}

function isPracticeQuestionAllowed(question) {
  if (!question) return false;
  if (!isKcServable(question)) return false;
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
    // Authored near-miss outputs, rendered under the prompt by
    // practice/question-examples.js. Empty for most bank questions.
    wrong_examples: Array.isArray(q.wrong_examples) ? q.wrong_examples : [],
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
    wrong_examples: Array.isArray(bankQ.wrong_examples) ? bankQ.wrong_examples : [],
    _artifactChanged: artifactChanged,
  };
}
