/* Optional math companion. Reuses lesson rendering and code execution; MCQ
   feedback is local to this visit and never writes coding mastery/exposure. */
(function () {
  const topicId = new URLSearchParams(location.search).get("math");
  if (topicId === null) return;
  window.__lessonDemoOnly = true;

  const el = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const link = (text, href) => {
    const node = el("a", text);
    node.href = href;
    return node;
  };

  async function start() {
    const host = document.getElementById("question-text");
    if (!host) return;
    document.querySelectorAll(".tab").forEach(tab =>
      tab.classList.toggle("active", tab.dataset.tab === "practice"));
    document.querySelectorAll(".page").forEach(page =>
      page.classList.toggle("hidden", page.id !== "page-practice"));
    document.getElementById("page-practice")?.classList.remove("session-idle");
    document.body.classList.add("lesson-mode", "math-mode");
    host.textContent = "Loading ray-tracing math…";
    try {
      const response = await fetch("lessons/math/arena-0-1.json?v=1");
      if (!response.ok) throw new Error(`Content request failed (${response.status})`);
      const data = await response.json();
      const topic = data.topics.find(item => item.id === topicId);
      const header = () => {
        host.replaceChildren();
        const nav = el("nav", "", "math-nav");
        nav.setAttribute("aria-label", "Ray-tracing math topics");
        for (const item of data.topics) {
          const a = link(item.title, `?math=${encodeURIComponent(item.id)}`);
          if (item === topic) a.setAttribute("aria-current", "page");
          nav.append(a);
        }
        host.append(nav);
      };
      if (!topic) {
        header();
        host.append(el("h2", data.title), el("p", data.scope));
        if (topicId) host.append(el("p", "Topic not found. Choose a topic above."));
        return;
      }
      let segmentIndex = 0;
      const firstAttempts = new Map();
      const prose = (text) => {
        const node = el("div", "", "nbv-md nb-scope");
        node.innerHTML = window.LessonGate.renderMarkdown(text);
        return node;
      };
      const button = (text, action) => {
        const node = el("button", text, "primary");
        node.type = "button";
        node.onclick = action;
        return node;
      };
      const focusTitle = () => {
        const title = host.querySelector("h2");
        title.tabIndex = -1;
        title.focus({ preventScroll: true });
        host.scrollIntoView({ block: "start" });
      };
      const transfer = () => {
        const aside = el("div", "", "math-transfer");
        aside.append(el("p", `Transfer to ARENA 0.1: ${topic.exercises.join(", ")}.`));
        aside.append(link("Open the connected coding lesson", `?lesson=${encodeURIComponent(topic.kc)}`));
        aside.append(document.createTextNode(" · "), link("Open ARENA 0.1", "?arena=0-1"));
        host.append(aside);
      };
      function lesson() {
        const section = topic.sections[segmentIndex];
        header();
        host.append(el("h2", section.title, "lesson-kp-title"));
        if (topic.prereqs.length) {
          const prerequisites = el("p", "Builds on: ");
          topic.prereqs.forEach((id, i) => {
            if (i) prerequisites.append(document.createTextNode(" · "));
            prerequisites.append(link(data.topics.find(t => t.id === id).title, `?math=${id}`));
          });
          host.append(prerequisites);
        }
        const fence = code => "\n\n```python\n" + code + "\n```";
        host.append(prose(section.concept + fence(section.code)));
        host.append(el("h3", "Worked example"));
        host.append(prose(section.worked + fence(section.worked_code)));
        host.append(button("Continue to the questions →", () => practice(section.questions, 0)));
        transfer();
        window.LessonNotebook.mount(host, `math:${topic.id}:${section.id}`);
        focusTitle();
      }
      function practice(questions, index) {
        const question = questions[index];
        header();
        host.append(el("h2", `${topic.title} · Question ${index + 1}/${questions.length}`, "lesson-kp-title"));
        const form = el("form", "", "math-question");
        const fieldset = el("fieldset");
        fieldset.append(el("legend", question.prompt));
        question.choices.forEach((choice, i) => {
          const label = el("label");
          const radio = document.createElement("input");
          radio.type = "radio";
          radio.name = "answer";
          radio.value = String(i);
          label.append(radio, document.createTextNode(choice));
          fieldset.append(label);
        });
        const submit = el("button", "Check answer", "primary");
        submit.type = "submit";
        submit.disabled = true;
        form.onchange = () => { submit.disabled = false; };
        const feedback = el("div", "", "math-feedback");
        feedback.setAttribute("role", "status");
        form.append(fieldset, submit, feedback);
        let submitted = false;
        form.onsubmit = event => {
          event.preventDefault();
          if (submitted) return;
          const selected = form.querySelector('input[name="answer"]:checked');
          if (!selected) return;
          submitted = true;
          const choice = Number(selected.value);
          const correct = choice === question.answer;
          if (!firstAttempts.has(question.id)) firstAttempts.set(question.id, correct);
          fieldset.disabled = true;
          submit.disabled = true;
          feedback.append(el("p", `${correct ? "Correct." : "Not quite."} ${question.feedback[choice]}`));
          if (!correct) feedback.append(el("p", `Correct answer: ${question.choices[question.answer]}. ${question.feedback[question.answer]}`));
          feedback.append(button(index + 1 < questions.length ? "Next question →" : "Finish practice →", () => {
            if (index + 1 < questions.length) practice(questions, index + 1);
            else complete();
          }));
        };
        host.append(form, button("Revisit the lesson", lesson));
        focusTitle();
      }
      function complete() {
        const section = topic.sections[segmentIndex];
        header();
        host.append(el("h2", "Practice complete", "lesson-kp-title"));
        const correct = section.questions.filter(q => firstAttempts.get(q.id)).length;
        host.append(el("p", `${correct}/${section.questions.length} correct on first attempt. Practice feedback only; coding mastery is unchanged.`));
        const missed = section.questions.filter(q => firstAttempts.get(q.id) === false);
        if (missed.length) host.append(button("Retry missed questions", () => practice(missed, 0)));
        if (segmentIndex + 1 < topic.sections.length) {
          host.append(button("Next concept →", () => { segmentIndex += 1; lesson(); }));
        } else {
          const next = data.topics[data.topics.indexOf(topic) + 1];
          if (next) host.append(link(`Next topic: ${next.title} →`, `?math=${next.id}`));
        }
        transfer();
        focusTitle();
      }
      lesson();
    } catch (error) {
      host.replaceChildren(el("h2", "Math content could not load"), el("p", error.message));
      host.append(link("Retry", location.href));
    }
  }
  if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", () => setTimeout(start, 300), { once: true });
  } else setTimeout(start, 300);
})();
