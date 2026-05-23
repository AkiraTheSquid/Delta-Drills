(function initArenaStage1() {
  const problemList = document.getElementById("arena-problem-list");
  const detail = document.getElementById("arena-problem-detail");
  const chapterFilter = document.getElementById("arena-chapter-filter");
  const problemCount = document.getElementById("arena-problem-count");

  if (!problemList || !detail || !chapterFilter || !problemCount) return;

  const problems = Array.isArray(window.ARENA_STAGE1_PROBLEMS) ? window.ARENA_STAGE1_PROBLEMS.slice() : [];
  if (!problems.length) {
    detail.innerHTML = '<div class="arena-detail-empty">No ARENA stage-1 problems are registered yet.</div>';
    return;
  }

  const chapterOptions = Array.from(
    new Map(problems.map((problem) => [problem.chapterId, problem.chapterLabel])).entries()
  );

  chapterOptions.forEach(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    chapterFilter.appendChild(option);
  });

  let selectedProblemId = problems[0].id;

  const toPercent = (weight) => `${Math.round(weight * 100)}%`;

  const createChipMarkup = (items) =>
    items.map((item) => `<span class="arena-chip">${item}</span>`).join("");

  const createSkillMarkup = (skills) =>
    skills
      .map(
        (entry) => `
          <div class="arena-skill-card">
            <div class="arena-skill-name">${entry.skill}</div>
            <div class="arena-skill-weight">${toPercent(entry.weight)}</div>
            <div class="arena-skill-note">Weighted contribution for this problem</div>
          </div>
        `
      )
      .join("");

  const metadataCard = (label, value) => `
    <div class="arena-metadata-card">
      <div class="arena-metadata-label">${label}</div>
      <div class="arena-metadata-value">${value}</div>
    </div>
  `;

  const exerciseRegistry = (window.ARENA_EXERCISES_BY_NOTEBOOK && typeof window.ARENA_EXERCISES_BY_NOTEBOOK === "object")
    ? window.ARENA_EXERCISES_BY_NOTEBOOK
    : {};

  const bookHrefFor = (notebookPath) => {
    if (typeof notebookPath !== "string") return "";
    const remapped = notebookPath
      .replace(/^content\/ARENA_5\.0-main\//, "arena-book/")
      .replace(/\.ipynb$/, ".html");
    return encodeURI(remapped);
  };

  const exerciseScoreFor = (problem) => {
    const fallback = Number(problem?.readinessScore) || 0;
    if (typeof window.computeArenaReadiness !== "function") return fallback;
    const s = window.computeArenaReadiness(problem?.skillWeights, fallback);
    return Number.isFinite(s) ? s : fallback;
  };

  const readinessTone = (score) => {
    if (score >= 75) return "ready";
    if (score >= 60) return "borderline";
    if (score >= 45) return "partial";
    return "needs-prereq";
  };

  const renderDetail = (problem) => {
    if (!problem) {
      detail.innerHTML = '<div class="arena-detail-empty">No ARENA problem matches the current filter.</div>';
      return;
    }

    const notebookHref = encodeURI(problem.notebookPath);
    const lessonHref = encodeURI(problem.lessonPath);
    const backupNotebook = problem.backupNotebookPath ? encodeURI(problem.backupNotebookPath) : "";
    const bookBase = bookHrefFor(problem.notebookPath);
    const exercises = Array.isArray(exerciseRegistry[problem.notebookPath])
      ? exerciseRegistry[problem.notebookPath]
      : [];
    const sectionNumber = (problem.sectionLabel || "").trim().split(/\s+/)[0] || problem.id;
    const exerciseScore = exerciseScoreFor(problem);
    const exerciseTone = readinessTone(exerciseScore);

    detail.innerHTML = `
      <div class="arena-detail-header">
        <div class="arena-detail-title-block">
          <div class="arena-chapter-pill">${problem.chapterLabel}</div>
          <h2>${problem.title}</h2>
          <div class="arena-problem-id">${problem.id} · ${problem.sectionLabel}</div>
        </div>
        <div class="arena-readiness">
          <div class="arena-readiness-label">Readiness</div>
          <div class="arena-readiness-value">${problem.readinessScore}/100 · ${problem.readinessLabel}</div>
          <div class="arena-readiness-note">${problem.readinessNote}</div>
        </div>
      </div>

      <div class="arena-section">
        <div class="arena-section-title">Problem summary</div>
        <div class="arena-summary">${problem.summary}</div>
      </div>

      <div class="arena-section">
        <div class="arena-section-title">Prerequisite concepts</div>
        <div class="arena-chip-row">${createChipMarkup(problem.prerequisiteTags)}</div>
      </div>

      <div class="arena-section">
        <div class="arena-section-title">Weighted skill profile</div>
        <div class="arena-skill-grid">${createSkillMarkup(problem.skillWeights)}</div>
      </div>

      <div class="arena-section">
        <div class="arena-section-title">Stage-1 launch actions</div>
        <div class="arena-action-row">
          <a class="primary arena-secondary-link" href="${notebookHref}" target="_blank" rel="noreferrer">Open exercise notebook file</a>
          <a class="arena-secondary-link" href="${lessonHref}" target="_blank" rel="noreferrer">Open lesson markdown</a>
          ${
            backupNotebook
              ? `<a class="arena-secondary-link" href="${backupNotebook}" target="_blank" rel="noreferrer">Open 3.0 backup notebook</a>`
              : ""
          }
        </div>
      </div>

      <div class="arena-section">
        <div class="arena-section-title">Exercises (${exercises.length})</div>
        ${
          exercises.length
            ? `<ol class="arena-exercise-list">${exercises
                .map(
                  (ex, idx) => `<li>
                    <span class="arena-exercise-id">${sectionNumber}.${idx}</span>
                    <a class="arena-exercise-link" href="${bookBase}#${encodeURIComponent(ex.anchor)}" target="_blank" rel="noreferrer">${ex.title}</a>
                    <span class="arena-exercise-readiness arena-exercise-readiness--${exerciseTone}" title="Predicted readiness">${exerciseScore.toFixed(0)}%</span>
                  </li>`
                )
                .join("")}</ol>`
            : '<div class="arena-exercise-empty">No exercises detected in this notebook.</div>'
        }
      </div>

      <div class="arena-section">
        <div class="arena-section-title">Implementation metadata</div>
        <div class="arena-metadata-grid">
          ${metadataCard("Execution mode", problem.executionMode)}
          ${metadataCard("Launch path", problem.launchPath)}
          ${metadataCard("Lesson path", problem.lessonPath)}
          ${metadataCard("Notebook path", problem.notebookPath)}
        </div>
      </div>
    `;
  };

  const renderList = () => {
    const activeChapter = chapterFilter.value;
    const visibleProblems = problems.filter((problem) => activeChapter === "all" || problem.chapterId === activeChapter);

    if (!visibleProblems.some((problem) => problem.id === selectedProblemId)) {
      selectedProblemId = visibleProblems[0]?.id || "";
    }

    problemCount.textContent = `${visibleProblems.length} problem${visibleProblems.length === 1 ? "" : "s"} shown`;
    problemList.innerHTML = "";

    visibleProblems.forEach((problem) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `arena-problem-card${problem.id === selectedProblemId ? " active" : ""}`;
      button.innerHTML = `
        <div class="arena-problem-card-title">${problem.title}</div>
        <div class="arena-problem-card-meta">${problem.chapterLabel} · ${problem.sectionLabel}</div>
        <div class="arena-problem-card-summary">${problem.summary}</div>
      `;
      button.addEventListener("click", () => {
        selectedProblemId = problem.id;
        renderList();
        renderDetail(problem);
      });
      problemList.appendChild(button);
    });

    renderDetail(visibleProblems.find((problem) => problem.id === selectedProblemId));
  };

  chapterFilter.addEventListener("change", renderList);
  renderList();
})();
