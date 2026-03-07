/* ================================================================
   PRACTICE API — backend / supabase / local routing
   ================================================================ */

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
          topic: q.topic || "",
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
      pyodide.runPython(
        "import sys\nfrom io import StringIO\nsys.stdout = StringIO()\nsys.stderr = StringIO()\nimport numpy as np\nnp.random.seed(0)"
      );
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
