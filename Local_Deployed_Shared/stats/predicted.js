/* ================================================================
   STATS PREDICTED — predicted course scores (ARENA-sourced)

   Mirrors the Areas table layout but populates from the ARENA
   problem registry (`window.ARENA_STAGE1_PROBLEMS`). Each chapter
   becomes a top-level row; each ARENA subsection becomes an
   expandable sub-row; each exercise stays visible as a leaf row.

   Helpers split into sibling files (loaded BEFORE this in index.html):
     - predicted-links.js: URL builders + openLinkCell
     - predicted-data.js: section sort/key/aggregation helpers
   ================================================================ */

const openPredictedChapters = new Set();
const openPredictedSections = new Set();
const openPredictedProblems = new Set();
const openPredictedPrereqs = new Set();

const PREDICTED_COLSPAN = 10;

// Heading text shown in Colab / Jupyter Book strips markdown backticks
// (e.g. `nn.Module` renders as nn.Module), so Ctrl+F for the raw title
// fails. Strip them from the clipboard payload so the search matches.
const copyKeyAttr = (text) => String(text).replace(/`/g, "").replace(/"/g, "&quot;");

// Wraps the canonical exercisesForProblem helper with a fallback to the
// hardcoded temp scaffold (predicted-prereqs-temp.js) for the one 0.0
// notebook whose exercise list is empty in the auto-extracted registry.
// Delete the fallback when the concept-graph backend ships.
const getExercisesWithTempFallback = (problem) => {
  const real = exercisesForProblem(problem);
  if (real.length) return real;
  if (window.ARENA_PREREQS_TEMP_ENABLED &&
      problem?.notebookPath === window.ARENA_PREREQS_TEMP_NOTEBOOK_PATH) {
    return Array.isArray(window.ARENA_PREREQS_TEMP_EXERCISES)
      ? window.ARENA_PREREQS_TEMP_EXERCISES
      : [];
  }
  return real;
};

const buildPredictedAreas = () => {
  const problems = Array.isArray(window.ARENA_STAGE1_PROBLEMS)
    ? window.ARENA_STAGE1_PROBLEMS
    : [];
  if (!problems.length) return [];

  // Group problems by chapter, then by subsection, then keep the
  // individual exercise rows as leaves.
  const byChapter = new Map();
  problems.forEach((p) => {
    const key = p.chapterId || p.chapterLabel || "unknown";
    if (!byChapter.has(key)) {
      byChapter.set(key, {
        id: key,
        label: p.chapterLabel || key,
        problems: [],
      });
    }
    byChapter.get(key).problems.push(p);
  });

  // Curriculum-order rows (no ranking — students go through ARENA
  // sequentially, so chapter 0 → 3 reflects the actual study order).
  return Array.from(byChapter.values()).map((ch) => {
    const orderedProblems = ch.problems.slice().sort((a, b) => compareSectionLabels(a.sectionLabel, b.sectionLabel));
    const scores = orderedProblems.map((p) => computeProblemScore(p));
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    const chapterNumber = (ch.id.match(/chapter(\d+)/) || [])[1] || "";
    const subsectionMap = new Map();
    orderedProblems.forEach((problem) => {
      const subsectionKey = subsectionKeyForProblem(problem);
      if (!subsectionMap.has(subsectionKey)) {
        subsectionMap.set(subsectionKey, {
          id: `${ch.id}::${subsectionKey}`,
          label: subsectionKey,
          problems: [],
        });
      }
      subsectionMap.get(subsectionKey).problems.push(problem);
    });

    const subsections = Array.from(subsectionMap.values()).map((sub) => {
      const subScores = sub.problems.map((p) => computeProblemScore(p));
      const subAvgScore = subScores.reduce((a, b) => a + b, 0) / subScores.length;
      return {
        ...sub,
        avgScore: subAvgScore,
        topSkill: aggregateTopSkill(sub.problems),
      };
    });

    return { ...ch, avgScore, chapterNumber, subsections };
  });
};

const isChapterExpanded = (body, chapterId) => {
  const btn = body.querySelector(`[data-pred-chapter-toggle="${chapterId}"]`);
  return btn?.dataset.open === "true";
};

const renderPredictedTable = () => {
  const body = document.getElementById("predicted-table-body");
  if (!body) return;

  // Snapshot expand state
  body.querySelectorAll("[data-pred-chapter-toggle]").forEach((btn) => {
    const id = btn.getAttribute("data-pred-chapter-toggle");
    if (btn.dataset.open === "true") openPredictedChapters.add(id);
    else openPredictedChapters.delete(id);
  });
  body.querySelectorAll("[data-pred-section-toggle]").forEach((btn) => {
    const id = btn.getAttribute("data-pred-section-toggle");
    if (btn.dataset.open === "true") openPredictedSections.add(id);
    else openPredictedSections.delete(id);
  });
  body.querySelectorAll("[data-pred-problem-toggle]").forEach((btn) => {
    const id = btn.getAttribute("data-pred-problem-toggle");
    if (btn.dataset.open === "true") openPredictedProblems.add(id);
    else openPredictedProblems.delete(id);
  });
  body.querySelectorAll("[data-pred-prereq-toggle]").forEach((btn) => {
    const id = btn.getAttribute("data-pred-prereq-toggle");
    if (btn.dataset.open === "true") openPredictedPrereqs.add(id);
    else openPredictedPrereqs.delete(id);
  });

  body.innerHTML = "";

  const chapters = buildPredictedAreas();
  if (!chapters.length) {
    body.innerHTML = `<tr><td colspan="${PREDICTED_COLSPAN}" style="text-align:center;padding:1.5rem;color:var(--color-muted)">No ARENA registry data available.</td></tr>`;
    return;
  }

  // TEMP: when the prereq scaffold is enabled, auto-expand only the 0.0
  // chapter/section/problem on first render so the user lands directly on
  // the prereq-tagged exercises. Other chapters stay collapsed as usual.
  if (window.ARENA_PREREQS_TEMP_ENABLED && !window._arenaPrereqsTempSeeded) {
    const targetChapter = window.ARENA_PREREQS_TEMP_RESTRICT_CHAPTER;
    const targetProblemId = window.ARENA_PREREQS_TEMP_RESTRICT_PROBLEM_ID;
    chapters.forEach((ch) => {
      if (ch.id !== targetChapter) return;
      openPredictedChapters.add(ch.id);
      ch.subsections.forEach((sub) => {
        const matches = sub.problems.some((p) => p.id === targetProblemId);
        if (!matches) return;
        openPredictedSections.add(sub.id);
        sub.problems.forEach((p) => {
          if (p.id === targetProblemId) openPredictedProblems.add(`${sub.id}::${p.id}`);
        });
      });
    });
    window._arenaPrereqsTempSeeded = true;
  }

  chapters.forEach((ch) => {
    const gap = Math.max(0, 100 - ch.avgScore);
    const areaRow = document.createElement("tr");
    areaRow.className = "stats-row stats-row-top";
    areaRow.innerHTML = `
      <td class="stats-col-toggle">
        <button class="stats-toggle" type="button" data-pred-chapter-toggle="${ch.id}">▸</button>
      </td>
      <td class="stats-col-check">
        <input type="checkbox" class="stats-check" checked />
      </td>
      <td>${ch.chapterNumber}</td>
      <td class="stats-col-area">${ch.label}</td>
      <td class="stats-col-weight">${ch.subsections.length} subsections</td>
      <td class="stats-col-score">
        <div class="stats-bar">
          <div class="stats-bar-track">
            <div class="stats-bar-fill" style="width: ${ch.avgScore}%"></div>
          </div>
          <span class="stats-bar-value">${ch.avgScore.toFixed(0)}%</span>
        </div>
      </td>
      <td class="stats-col-solved">${ch.problems.length}</td>
      <td class="stats-col-lr">—</td>
      <td class="stats-col-delta">
        <div class="stats-bar">
          <div class="stats-bar-track">
            <div class="stats-bar-fill stats-bar-fill-delta" style="width: ${gap}%"></div>
          </div>
          <span class="stats-bar-value">${gap.toFixed(0)}%</span>
        </div>
      </td>
      ${openLinkCell(bookHrefForNotebook(ch.problems[0]?.notebookPath), `Open ${ch.label} (first section)`)}
    `;
    body.appendChild(areaRow);

    ch.subsections.forEach((sub) => {
      const subGap = Math.max(0, 100 - sub.avgScore);
      const subRow = document.createElement("tr");
      subRow.className = "stats-row stats-subrow stats-predicted-section-row hidden";
      subRow.dataset.predSectionFor = ch.id;
      subRow.innerHTML = `
        <td class="stats-col-toggle">
          <button class="stats-toggle" type="button" data-pred-section-toggle="${sub.id}">▸</button>
        </td>
        <td class="stats-col-check">
          <input type="checkbox" class="stats-check" checked />
        </td>
        <td>${sub.label}</td>
        <td class="stats-col-area stats-subarea">${sub.label}</td>
        <td class="stats-col-weight">${sub.problems.length} exercises</td>
        <td class="stats-col-score">
          <div class="stats-bar">
            <div class="stats-bar-track">
              <div class="stats-bar-fill" style="width: ${sub.avgScore}%"></div>
            </div>
            <span class="stats-bar-value">${sub.avgScore.toFixed(0)}%</span>
          </div>
        </td>
        <td class="stats-col-solved">${sub.problems.length}</td>
        <td class="stats-col-lr">${sub.topSkill}</td>
        <td class="stats-col-delta">
          <div class="stats-bar">
            <div class="stats-bar-track">
              <div class="stats-bar-fill stats-bar-fill-delta" style="width: ${subGap}%"></div>
            </div>
            <span class="stats-bar-value">${subGap.toFixed(0)}%</span>
          </div>
        </td>
        ${openLinkCell(bookHrefForNotebook(sub.problems[0]?.notebookPath), `Open section ${sub.label}`)}
      `;
      body.appendChild(subRow);

      sub.problems.forEach((p) => {
        const score = computeProblemScore(p);
        const leafGap = Math.max(0, 100 - score);
        const exercises = getExercisesWithTempFallback(p);
        const sectionNumber = sectionLabelForProblem(p);
        const problemKey = `${sub.id}::${p.id}`;
        const leafRow = document.createElement("tr");
        leafRow.className = "stats-row stats-subrow stats-predicted-problem-row hidden";
        leafRow.dataset.predProblemFor = sub.id;
        leafRow.innerHTML = `
          <td class="stats-col-toggle">${
            exercises.length
              ? `<button class="stats-toggle" type="button" data-pred-problem-toggle="${problemKey}">▸</button>`
              : ""
          }</td>
          <td class="stats-col-check">
            <input type="checkbox" class="stats-check" checked />
          </td>
          <td>${sectionNumber}</td>
          <td class="stats-col-area stats-subarea">${p.title || p.sectionLabel || p.id}</td>
          <td class="stats-col-weight">${exercises.length ? `${exercises.length} exercises` : (p.readinessLabel || "—")}</td>
          <td class="stats-col-score">
            <div class="stats-bar">
              <div class="stats-bar-track">
                <div class="stats-bar-fill" style="width: ${score}%"></div>
              </div>
              <span class="stats-bar-value">${score.toFixed(0)}%</span>
            </div>
          </td>
          <td class="stats-col-solved">${exercises.length || 1}</td>
          <td class="stats-col-lr">${topSkillLabel(p)}</td>
          <td class="stats-col-delta">
            <div class="stats-bar">
              <div class="stats-bar-track">
                <div class="stats-bar-fill stats-bar-fill-delta" style="width: ${leafGap}%"></div>
              </div>
              <span class="stats-bar-value">${leafGap.toFixed(0)}%</span>
            </div>
          </td>
          ${openLinkCell(bookHrefForNotebook(p.notebookPath), `Open ${p.title || p.id}`)}
        `;
        body.appendChild(leafRow);

        exercises.forEach((ex, exIdx) => {
          const exId = `${sectionNumber}.${exIdx}`;
          const exGap = Math.max(0, 100 - score);
          const exKey = `${problemKey}::${exIdx}`;
          const prereqs = (window.ARENA_PREREQS_TEMP_ENABLED &&
                           window.ARENA_PREREQS_TEMP_BY_EXERCISE &&
                           window.ARENA_PREREQS_TEMP_BY_EXERCISE[ex.title]) || [];
          const exRow = document.createElement("tr");
          exRow.className = "stats-row stats-subrow stats-predicted-exercise-row hidden";
          exRow.dataset.predExerciseFor = problemKey;
          exRow.innerHTML = `
            <td class="stats-col-toggle">${
              prereqs.length
                ? `<button class="stats-toggle" type="button" data-pred-prereq-toggle="${exKey}" title="Show / hide prerequisite concepts">▸</button>`
                : ""
            }</td>
            <td class="stats-col-check">
              <input type="checkbox" class="stats-check" checked />
            </td>
            <td>${exId}</td>
            <td class="stats-col-area stats-subarea">${ex.title || ex.anchor || `Exercise ${exIdx}`}</td>
            <td class="stats-col-weight">${p.readinessLabel || "—"}</td>
            <td class="stats-col-score">
              <div class="stats-bar">
                <div class="stats-bar-track">
                  <div class="stats-bar-fill" style="width: ${score}%"></div>
                </div>
                <span class="stats-bar-value">${score.toFixed(0)}%</span>
              </div>
            </td>
            <td class="stats-col-solved">1</td>
            <td class="stats-col-lr">${topSkillLabel(p)}</td>
            <td class="stats-col-delta">
              <div class="stats-bar">
                <div class="stats-bar-track">
                  <div class="stats-bar-fill stats-bar-fill-delta" style="width: ${exGap}%"></div>
                </div>
                <span class="stats-bar-value">${exGap.toFixed(0)}%</span>
              </div>
            </td>
            <td class="stats-col-open">
              <a class="stats-open-link" href="${bookHrefForNotebook(p.notebookPath)}#${encodeURIComponent(ex.anchor)}" target="_blank" rel="noreferrer" title="Read in Jupyter Book — deep-links straight to the exercise">Read ↗</a>
              <a class="stats-open-link stats-open-link--colab" href="${colabUpstreamHref(p.notebookPath)}" target="_blank" rel="noreferrer" data-copy-key="${copyKeyAttr(ex.title)}" title="Open whole notebook in Colab + copy heading to clipboard. Then Ctrl+F, Ctrl+V, Enter.">Colab ↗</a>
              <a class="stats-open-link stats-open-link--vscode" href="${vsCodeHrefFor(p.notebookPath)}" data-copy-key="${copyKeyAttr(ex.title)}" title="Open whole notebook in local VS Code + copy heading to clipboard. Then Ctrl+F, Ctrl+V, Enter.">VS Code</a>
              <button type="button" class="stats-open-link stats-open-link--copy" data-copy-key="${copyKeyAttr(ex.title)}" title="Copy heading without opening anything">📋</button>
            </td>
          `;
          body.appendChild(exRow);

          if (prereqs.length) {
            const prereqRow = document.createElement("tr");
            prereqRow.className = "stats-row stats-subrow stats-predicted-prereq-row hidden";
            prereqRow.dataset.predPrereqFor = exKey;
            prereqRow.innerHTML = `
              <td colspan="${PREDICTED_COLSPAN}">
                <div class="stats-prereq-panel">
                  <div class="stats-prereq-title">Prerequisite concepts (Delta Drills) — meet these before tackling this exercise</div>
                  ${prereqs.map((pr) => {
                    const sc = (typeof window.getArenaPrereqSubtopicScore === "function")
                      ? window.getArenaPrereqSubtopicScore(pr.topic, pr.subtopic)
                      : null;
                    const scoreLabel = (sc == null) ? "no data" : `${Math.round(sc)}%`;
                    const fillWidth = Math.max(0, Math.min(100, sc || 0));
                    const met = sc != null && sc >= pr.minPct;
                    return `
                      <div class="stats-prereq-item ${met ? "stats-prereq-met" : "stats-prereq-unmet"}">
                        <span class="stats-prereq-label">${pr.topic} / ${pr.subtopic}</span>
                        <span class="stats-prereq-bar">
                          <span class="stats-prereq-bar-track">
                            <span class="stats-prereq-bar-fill" style="width:${fillWidth}%"></span>
                          </span>
                          <span class="stats-prereq-bar-value">${scoreLabel}</span>
                        </span>
                        <span class="stats-prereq-target">target ≥ ${pr.minPct}%</span>
                        <button type="button" class="stats-prereq-jump" data-prereq-jump-topic="${pr.topic}" data-prereq-jump-subtopic="${pr.subtopic}" title="Jump to the Practice tab to work on this concept">Practice ↗</button>
                      </div>
                    `;
                  }).join("")}
                </div>
              </td>
            `;
            body.appendChild(prereqRow);
          }
        });
      });
    });
  });

  // Auto-copy the exercise heading on every click of an element with
  // data-copy-key. For the dedicated 📋 button we also flash a checkmark;
  // for the launch links (Colab / VS Code) the visible feedback is the new
  // tab opening, so we just fire clipboard.writeText silently and let the
  // anchor navigate. By the time the user lands in Colab and presses
  // Ctrl+F + Ctrl+V, the heading is on their clipboard.
  body.querySelectorAll("[data-copy-key]").forEach((el) => {
    el.addEventListener("click", () => {
      const text = el.getAttribute("data-copy-key") || "";
      if (navigator.clipboard?.writeText) navigator.clipboard.writeText(text).catch(() => {});
      if (el.tagName === "BUTTON" && el.classList.contains("stats-open-link--copy")) {
        el.textContent = "✓";
        el.classList.add("copied");
        setTimeout(() => { el.textContent = "📋"; el.classList.remove("copied"); }, 1200);
      }
    });
  });

  // Expand/collapse wiring.
  const showExercisesForProblem = (problemKey) => {
    body.querySelectorAll(`[data-pred-exercise-for="${problemKey}"]`).forEach((row) => row.classList.remove("hidden"));
  };

  body.querySelectorAll("[data-pred-chapter-toggle]").forEach((btn) => {
    const chapterId = btn.getAttribute("data-pred-chapter-toggle");
    if (openPredictedChapters.has(chapterId)) {
      btn.dataset.open = "true";
      btn.textContent = "▾";
      body.querySelectorAll(`[data-pred-section-for="${chapterId}"]`).forEach((row) => row.classList.remove("hidden"));
      body.querySelectorAll(`[data-pred-problem-for^="${chapterId}::"]`).forEach((row) => {
        const sectionId = row.getAttribute("data-pred-problem-for");
        if (openPredictedSections.has(sectionId)) row.classList.remove("hidden");
      });
      openPredictedProblems.forEach((problemKey) => {
        if (!problemKey.startsWith(`${chapterId}::`)) return;
        const sectionId = problemKey.split("::").slice(0, 2).join("::");
        if (openPredictedSections.has(sectionId)) showExercisesForProblem(problemKey);
      });
    }
    btn.addEventListener("click", () => {
      const isOpen = btn.dataset.open === "true";
      btn.dataset.open = isOpen ? "false" : "true";
      btn.textContent = isOpen ? "▸" : "▾";
      if (isOpen) openPredictedChapters.delete(chapterId);
      else openPredictedChapters.add(chapterId);
      body.querySelectorAll(`[data-pred-section-for="${chapterId}"]`).forEach((row) => row.classList.toggle("hidden", isOpen));
      body.querySelectorAll(`[data-pred-problem-for^="${chapterId}::"]`).forEach((row) => {
        const sectionId = row.getAttribute("data-pred-problem-for");
        const sectionOpen = openPredictedSections.has(sectionId);
        row.classList.toggle("hidden", isOpen ? !sectionOpen : true);
      });
      body.querySelectorAll(`[data-pred-exercise-for^="${chapterId}::"]`).forEach((row) => {
        const problemKey = row.getAttribute("data-pred-exercise-for");
        const sectionId = problemKey.split("::").slice(0, 2).join("::");
        const visible = !isOpen
          ? false
          : openPredictedSections.has(sectionId) && openPredictedProblems.has(problemKey);
        row.classList.toggle("hidden", !visible);
      });
      body.querySelectorAll(`[data-pred-prereq-for^="${chapterId}::"]`).forEach((row) => {
        const exKey = row.getAttribute("data-pred-prereq-for");
        const problemKey = exKey.split("::").slice(0, 3).join("::");
        const sectionId = problemKey.split("::").slice(0, 2).join("::");
        const visible = !isOpen
          && openPredictedSections.has(sectionId)
          && openPredictedProblems.has(problemKey)
          && openPredictedPrereqs.has(exKey);
        row.classList.toggle("hidden", !visible);
      });
    });
  });

  body.querySelectorAll("[data-pred-section-toggle]").forEach((btn) => {
    const sectionId = btn.getAttribute("data-pred-section-toggle");
    const chapterId = sectionId.split("::")[0];
    if (openPredictedSections.has(sectionId)) {
      btn.dataset.open = "true";
      btn.textContent = "▾";
      if (openPredictedChapters.has(chapterId)) {
        body.querySelectorAll(`[data-pred-problem-for="${sectionId}"]`).forEach((row) => row.classList.remove("hidden"));
      }
    }
    btn.addEventListener("click", () => {
      const isOpen = btn.dataset.open === "true";
      btn.dataset.open = isOpen ? "false" : "true";
      btn.textContent = isOpen ? "▸" : "▾";
      if (isOpen) openPredictedSections.delete(sectionId);
      else openPredictedSections.add(sectionId);
      body.querySelectorAll(`[data-pred-problem-for="${sectionId}"]`).forEach((row) => {
        const chapterVisible = isChapterExpanded(body, chapterId);
        row.classList.toggle("hidden", isOpen || !chapterVisible);
      });
      body.querySelectorAll(`[data-pred-exercise-for^="${sectionId}::"]`).forEach((row) => {
        const problemKey = row.getAttribute("data-pred-exercise-for");
        const visible = !isOpen && openPredictedProblems.has(problemKey) && isChapterExpanded(body, chapterId);
        row.classList.toggle("hidden", !visible);
      });
      body.querySelectorAll(`[data-pred-prereq-for^="${sectionId}::"]`).forEach((row) => {
        const exKey = row.getAttribute("data-pred-prereq-for");
        const problemKey = exKey.split("::").slice(0, 3).join("::");
        const visible = !isOpen
          && openPredictedProblems.has(problemKey)
          && openPredictedPrereqs.has(exKey)
          && isChapterExpanded(body, chapterId);
        row.classList.toggle("hidden", !visible);
      });
    });
  });

  body.querySelectorAll("[data-pred-problem-toggle]").forEach((btn) => {
    const problemKey = btn.getAttribute("data-pred-problem-toggle");
    const sectionId = problemKey.split("::").slice(0, 2).join("::");
    const chapterId = sectionId.split("::")[0];
    if (openPredictedProblems.has(problemKey)) {
      btn.dataset.open = "true";
      btn.textContent = "▾";
      if (openPredictedChapters.has(chapterId) && openPredictedSections.has(sectionId)) {
        showExercisesForProblem(problemKey);
      }
    }
    btn.addEventListener("click", () => {
      const isOpen = btn.dataset.open === "true";
      btn.dataset.open = isOpen ? "false" : "true";
      btn.textContent = isOpen ? "▸" : "▾";
      if (isOpen) openPredictedProblems.delete(problemKey);
      else openPredictedProblems.add(problemKey);
      body.querySelectorAll(`[data-pred-exercise-for="${problemKey}"]`).forEach((row) => {
        row.classList.toggle("hidden", isOpen);
      });
      body.querySelectorAll(`[data-pred-prereq-for^="${problemKey}::"]`).forEach((row) => {
        const exKey = row.getAttribute("data-pred-prereq-for");
        row.classList.toggle("hidden", isOpen || !openPredictedPrereqs.has(exKey));
      });
    });
  });

  body.querySelectorAll("[data-pred-prereq-toggle]").forEach((btn) => {
    const exKey = btn.getAttribute("data-pred-prereq-toggle");
    if (openPredictedPrereqs.has(exKey)) {
      btn.dataset.open = "true";
      btn.textContent = "▾";
      const problemKey = exKey.split("::").slice(0, 3).join("::");
      const sectionId = problemKey.split("::").slice(0, 2).join("::");
      const chapterId = sectionId.split("::")[0];
      if (openPredictedChapters.has(chapterId) &&
          openPredictedSections.has(sectionId) &&
          openPredictedProblems.has(problemKey)) {
        body.querySelectorAll(`[data-pred-prereq-for="${exKey}"]`).forEach((row) => row.classList.remove("hidden"));
      }
    }
    btn.addEventListener("click", () => {
      const isOpen = btn.dataset.open === "true";
      btn.dataset.open = isOpen ? "false" : "true";
      btn.textContent = isOpen ? "▸" : "▾";
      if (isOpen) openPredictedPrereqs.delete(exKey);
      else openPredictedPrereqs.add(exKey);
      body.querySelectorAll(`[data-pred-prereq-for="${exKey}"]`).forEach((row) => {
        row.classList.toggle("hidden", isOpen);
      });
    });
  });

  // Practice ↗ buttons inside the prereq panel — click the Practice nav
  // tab so the user lands on the practice page. Topic-specific filtering
  // is intentionally out of scope for the temp scaffold.
  body.querySelectorAll("[data-prereq-jump-topic]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tabBtn = document.querySelector('[data-tab="practice"]');
      if (tabBtn) tabBtn.click();
    });
  });
};
