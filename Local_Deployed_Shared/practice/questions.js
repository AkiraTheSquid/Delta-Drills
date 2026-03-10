/* ================================================================
   PRACTICE QUESTIONS — questions.json loader
   ================================================================ */

let questionsBank = null; // array of question objects from questions.json
let questionsBankJson = null; // JSON string for passing to Pyodide engine

async function loadQuestionsBank() {
  if (questionsBank) return questionsBank;
  try {
    const res = await fetch(`questions.json?v=20260310i`, { cache: "no-store" });
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

function getPracticeEligibleQuestions() {
  if (!Array.isArray(questionsBank)) return null;
  if (typeof isSubtopicEnabled !== "function") return questionsBank;
  return questionsBank.filter((q) => isSubtopicEnabled(q.subtopic, q.topic || ""));
}

function isPracticeQuestionAllowed(question) {
  if (!question) return false;
  if (typeof isSubtopicEnabled !== "function") return true;
  return isSubtopicEnabled(question.subtopic, question.topic || "");
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
