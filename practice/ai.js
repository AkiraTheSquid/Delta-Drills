/* ================================================================
   PRACTICE AI — judge + explanation helpers
   ================================================================ */

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
  const res =
    typeof apiFetch === "function"
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
