/* ================================================================
   PRACTICE AI — explanation helper
   ================================================================

   `fetchAIJudge` lived here until 2026-07-31. It was the guest-mode fallback
   grader: with no test cases to compare against, an LLM decided whether the
   learner's code was right. Practice does not run or grade code any longer —
   the learner reports the result themselves — so there was nothing left to
   judge. Deleted rather than left dangling, because an unused grader is an
   invitation to route something back through it.
*/

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
