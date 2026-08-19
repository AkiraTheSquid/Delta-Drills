/* ================================================================
   PRACTICE TUTOR — post-answer chat with ChatGPT
   ================================================================

   A ChatGPT-shaped thread that opens under the AI Explanation once a
   question has been graded. Tutor turns sit on the LEFT, the learner's
   on the right.

   The thread is per-question and deliberately not persisted: a new
   question is a new conversation (ui.js calls `reset()` on render).
   The client is stateless — every turn POSTs the whole visible thread
   plus the problem context to /api/practice/ai-tutor, which owns the
   system prompt.
   ================================================================ */

const PracticeTutor = (() => {
  // Problem context for the current question, captured at grade time.
  let context = null;
  // Visible thread: [{ role: "user" | "assistant", content }]
  let thread = [];
  let pending = false;

  // ---- Markdown-ish rendering -------------------------------------
  // The tutor is told to answer in Markdown, but pulling in a parser for
  // three constructs is not worth the dependency. Everything is escaped
  // first, so only what we deliberately re-introduce can become markup.

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function renderInline(text) {
    return escapeHtml(text)
      .replace(/`([^`\n]+)`/g, (_, code) => `<code>${code}</code>`)
      .replace(/\*\*([^*\n]+)\*\*/g, (_, bold) => `<strong>${bold}</strong>`)
      .replace(/\n/g, "<br />");
  }

  // Split on fenced blocks so code survives verbatim and prose gets the
  // inline treatment.
  function renderMarkdown(text) {
    const parts = String(text == null ? "" : text).split(/```/);
    return parts
      .map((part, i) => {
        if (i % 2 === 0) return renderInline(part);
        // Odd chunks are fenced; drop the language tag on the first line.
        const body = part.replace(/^[a-zA-Z0-9_+-]*\n/, "");
        return `<pre class="tutor-code">${escapeHtml(body.replace(/\n$/, ""))}</pre>`;
      })
      .join("");
  }

  // ---- DOM --------------------------------------------------------

  function appendBubble(role, content, extraClass) {
    if (!tutorThread) return null;
    const row = document.createElement("div");
    row.className = `tutor-row tutor-row--${role}` + (extraClass ? ` ${extraClass}` : "");
    const bubble = document.createElement("div");
    bubble.className = `tutor-bubble tutor-bubble--${role}`;
    if (role === "assistant") {
      bubble.innerHTML = renderMarkdown(content);
    } else {
      bubble.textContent = content;
    }
    row.appendChild(bubble);
    tutorThread.appendChild(row);
    tutorThread.scrollTop = tutorThread.scrollHeight;
    return row;
  }

  function setPending(isPending) {
    pending = isPending;
    if (tutorSendBtn) tutorSendBtn.disabled = isPending;
    if (tutorInput) tutorInput.disabled = isPending;
  }

  function autosize() {
    if (!tutorInput) return;
    tutorInput.style.height = "auto";
    tutorInput.style.height = Math.min(tutorInput.scrollHeight, 160) + "px";
  }

  // ---- Lifecycle --------------------------------------------------

  // Called on every question render: empty thread, hidden panel.
  function reset() {
    context = null;
    thread = [];
    setPending(false);
    if (tutorThread) tutorThread.innerHTML = "";
    if (tutorInput) {
      tutorInput.value = "";
      tutorInput.style.height = "auto";
    }
    if (tutorSection) tutorSection.classList.add("hidden");
  }

  // Called once the attempt is graded — this is what makes the tutor
  // available. `ctx` carries everything the backend needs about the drill.
  function open(ctx) {
    context = ctx || {};
    thread = [];
    setPending(false);
    if (tutorThread) tutorThread.innerHTML = "";
    if (tutorSection) tutorSection.classList.remove("hidden");
    if (tutorEmpty) tutorEmpty.classList.remove("hidden");
  }

  // The AI Explanation lands after the panel opens; fold it into the
  // context so the tutor does not repeat it back.
  function setExplanation(text) {
    if (context) context.explanation = text || "";
  }

  async function send(text) {
    const message = String(text == null ? "" : text).trim();
    if (!message || pending || !context) return;

    if (tutorEmpty) tutorEmpty.classList.add("hidden");
    thread.push({ role: "user", content: message });
    appendBubble("user", message);

    const thinking = appendBubble("assistant", "Thinking…", "tutor-row--thinking");
    setPending(true);
    try {
      const res = await apiFetch("/api/practice/ai-tutor", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question_text: context.questionText || "",
          solution_code: context.solutionCode || "",
          user_code: context.userCode || "",
          actual_output: context.actualOutput || "",
          expected_output: context.expectedOutput || "",
          explanation: context.explanation || "",
          was_correct:
            typeof context.wasCorrect === "boolean" ? context.wasCorrect : null,
          messages: thread,
        }),
      });
      if (thinking) thinking.remove();
      if (!res.ok) {
        const detail = await res.text().catch(() => "");
        appendBubble(
          "assistant",
          "Could not reach the tutor." + (detail ? "\n\n" + detail : ""),
          "tutor-row--error"
        );
        // Not added to `thread` — a failed turn should not become context.
        return;
      }
      const data = await res.json();
      const reply = data.reply || "No reply available.";
      thread.push({ role: "assistant", content: reply });
      appendBubble("assistant", reply);
    } catch (e) {
      if (thinking) thinking.remove();
      appendBubble("assistant", "Could not reach the tutor.", "tutor-row--error");
    } finally {
      setPending(false);
      if (tutorInput) tutorInput.focus();
    }
  }

  function sendFromInput() {
    if (!tutorInput) return;
    const text = tutorInput.value;
    tutorInput.value = "";
    tutorInput.style.height = "auto";
    send(text);
  }

  // ---- Wiring -----------------------------------------------------

  function init() {
    if (tutorSendBtn) tutorSendBtn.addEventListener("click", sendFromInput);
    if (tutorInput) {
      tutorInput.addEventListener("input", autosize);
      tutorInput.addEventListener("keydown", (e) => {
        // Enter sends, Shift+Enter newlines — the ChatGPT convention.
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendFromInput();
        }
      });
    }
    if (tutorSuggestions) {
      tutorSuggestions.addEventListener("click", (e) => {
        const btn = e.target.closest(".tutor-suggestion");
        if (btn) send(btn.textContent.trim());
      });
    }
  }

  return { init, reset, open, setExplanation, send };
})();

window.PracticeTutor = PracticeTutor;

// dom.js has already run by the time this script is evaluated, so the refs
// above are live — wire the composer here rather than waiting on initPractice,
// which only runs when the learner opens the practice page.
PracticeTutor.init();
