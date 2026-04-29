/* ================================================================
   PRACTICE API — backend / supabase / local routing
   ================================================================ */

function emitPracticeStateChanged() {
  window.dispatchEvent(new CustomEvent("delta:practice-state-changed"));
}

const PracticeAPI = {
  currentQuestion: practiceQuestionPool[0],

  outputsMatch(actualOutput, expectedOutput) {
    return (actualOutput || "").trim() === (expectedOutput || "").trim();
  },

  async recordLocalEval(questionId, correct) {
    if (practiceMode !== "backend") return;
    const res = await apiFetch("/api/practice/submit-local-eval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question_id: questionId, correct }),
    });
    if (res.status === 401) {
      handleExpiredToken();
      return;
    }
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(detail || "Failed to record local evaluation.");
    }
    await res.json();
  },

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
    const eligibleBank = getPracticeEligibleQuestions();

    if (pyodide && practiceEngineLoaded && bank && adaptiveStateJson) {
      if (!eligibleBank?.length) {
        throw new Error("No enabled practice sections. Re-enable at least one area in Statistics.");
      }
      const api = pyodide.globals.get("engine_api");
      const resultJson = api.next_question(adaptiveStateJson, JSON.stringify(eligibleBank));
      const result = JSON.parse(resultJson);
      adaptiveStateJson = result.state;
      await saveAdaptiveState();

      if (result.question) {
        const q = result.question;
        this.currentQuestion = {
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
    const requiresLocalPyodide = questionNeedsEinops(this.currentQuestion);

    if (practiceMode === "backend" && !requiresLocalPyodide) {
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

    // supabase/local or backend+einops fallback — run code with Pyodide and AI judge
    const pyodide = await initPyodide();
    let actualOutput = "";
    if (pyodide) {
      const preamble = await buildPyodidePreamble(this.currentQuestion);
      pyodide.runPython(preamble);
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
    const failed_tests = [];
    const solCode = this.currentQuestion.solution_code || "";
    const questionText = this.currentQuestion.question_text || "";
    let correct = false;

    if (
      this.currentQuestion.task_type === "stdout_prediction" &&
      expected &&
      !this.currentQuestion.supports_visual_output
    ) {
      correct = this.outputsMatch(actualOutput, expected);

      if (practiceEngineLoaded && adaptiveStateJson) {
        const api = pyodide.globals.get("engine_api");
        adaptiveStateJson = api.submit_answer(
          adaptiveStateJson,
          this.currentQuestion.question_id,
          this.currentQuestion.subtopic,
          this.currentQuestion.difficulty || 50,
          correct
        );
        await saveAdaptiveState();
      }

      if (practiceMode === "backend" && requiresLocalPyodide) {
        await this.recordLocalEval(questionId, correct);
      }

      return { correct, actual_output: actualOutput, expected_output: expected, failed_tests };
    }

    if (this.currentQuestion.submission_mode === "function" && this.currentQuestion.test_cases?.length) {
      const preamble = await buildPyodidePreamble(this.currentQuestion);
      const testsJsonLiteral = JSON.stringify(JSON.stringify(this.currentQuestion.test_cases));
      pyodide.runPython(preamble);
      try {
        pyodide.runPython(userCode);
        const resultJson = pyodide.runPython(`
import json
import numpy as np

def _delta_to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_delta_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_delta_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _delta_to_jsonable(v) for k, v in value.items()}
    return value

def _delta_equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_delta_equal(x, y) for x, y in zip(a, b))
    return bool(a == b)

_delta_results = []
for _delta_case in json.loads(${testsJsonLiteral}):
    try:
        if _delta_case.get("setup_code"):
            np.random.seed(0)
            exec(_delta_case["setup_code"], globals())
        _delta_actual = eval(_delta_case["call"], globals())
        _delta_expected_setup = _delta_case.get("expected_setup_code") or _delta_case.get("setup_code")
        if _delta_expected_setup:
            np.random.seed(0)
            exec(_delta_expected_setup, globals())
        _delta_expected = eval(_delta_case["expected_expr"], globals())
        _delta_results.append({
            "passed": bool(_delta_equal(_delta_actual, _delta_expected)),
            "actual": repr(_delta_to_jsonable(_delta_actual)),
            "expected": repr(_delta_to_jsonable(_delta_expected)),
            "error": "",
        })
    except Exception as _delta_exc:
        _delta_results.append({
            "passed": False,
            "actual": "",
            "expected": "",
            "error": f"{type(_delta_exc).__name__}: {_delta_exc}",
        })
json.dumps(_delta_results)
`);
        const parsed = JSON.parse(resultJson);
        failed_tests.push(...parsed.filter((test) => !test.passed));
        correct = failed_tests.length === 0;
        actualOutput = pyodide.runPython("sys.stdout.getvalue()").trim();
      } catch (e) {
        actualOutput = pyodide.runPython("sys.stderr.getvalue()").trim() || e.message;
        correct = false;
      } finally {
        pyodide.runPython("sys.stdout = sys.__stdout__\nsys.stderr = sys.__stderr__");
      }
      if (practiceMode === "backend" && requiresLocalPyodide) {
        await this.recordLocalEval(questionId, correct);
      }
      if (practiceEngineLoaded && adaptiveStateJson) {
        const api = pyodide.globals.get("engine_api");
        adaptiveStateJson = api.submit_answer(
          adaptiveStateJson,
          this.currentQuestion.question_id,
          this.currentQuestion.subtopic,
          this.currentQuestion.difficulty || 50,
          correct
        );
        await saveAdaptiveState();
      }
      return { correct, actual_output: actualOutput, expected_output: expected, failed_tests };
    }
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
      await saveAdaptiveState();
    }

    if (practiceMode === "backend" && requiresLocalPyodide) {
      await this.recordLocalEval(questionId, correct);
    }

    return { correct, actual_output: actualOutput, expected_output: expected, failed_tests };
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
      const response = await res.json();
      emitPracticeStateChanged();
      return response;
    }

    // supabase/local — apply feedback in Pyodide engine
    const pyodide = await initPyodide();
    if (pyodide && practiceEngineLoaded && adaptiveStateJson) {
      const api = pyodide.globals.get("engine_api");
      adaptiveStateJson = api.send_feedback(adaptiveStateJson, feedback);
      await saveAdaptiveState();
    }
    emitPracticeStateChanged();
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
      const response = await res.json();
      emitPracticeStateChanged();
      return response;
    }

    const pyodide = await initPyodide();
    if (pyodide && practiceEngineLoaded && adaptiveStateJson) {
      const api = pyodide.globals.get("engine_api");
      adaptiveStateJson = api.override_attempt(adaptiveStateJson, questionId, true);
      await saveAdaptiveState();
    }
    emitPracticeStateChanged();
    return;
  },
};
