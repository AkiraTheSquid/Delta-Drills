/* ================================================================
   PRACTICE QUESTIONS — questions.json loader
   ================================================================ */

let questionsBank = null; // array of question objects from questions.json
let questionsBankJson = null; // JSON string for passing to Pyodide engine

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
