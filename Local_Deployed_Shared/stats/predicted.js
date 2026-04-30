/* ================================================================
   STATS PREDICTED — predicted course scores (ARENA-sourced)

   Mirrors the Areas table layout but populates from the ARENA
   problem registry (`window.ARENA_STAGE1_PROBLEMS`). Each chapter
   becomes a top-level row; each ARENA problem in that chapter
   becomes an expandable sub-row.
   ================================================================ */

const openPredictedAreas = new Set();

const PREDICTED_COLSPAN = 9;

const buildPredictedAreas = () => {
  const problems = Array.isArray(window.ARENA_STAGE1_PROBLEMS)
    ? window.ARENA_STAGE1_PROBLEMS
    : [];
  if (!problems.length) return [];

  // Group problems by chapter
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

  // Build area-shaped rows. Sort chapters by avg readiness ASC
  // (lowest readiness = highest study priority = rank 1).
  const chapters = Array.from(byChapter.values()).map((ch) => {
    const scores = ch.problems.map((p) => Number(p.readinessScore) || 0);
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    return { ...ch, avgScore };
  });
  chapters.sort((a, b) => a.avgScore - b.avgScore);
  chapters.forEach((ch, i) => (ch.rank = i + 1));
  return chapters;
};

const topSkillLabel = (problem) => {
  const skills = Array.isArray(problem.skillWeights) ? problem.skillWeights : [];
  if (!skills.length) return "—";
  const top = skills.reduce((a, b) => (a.weight >= b.weight ? a : b));
  return `${top.skill} (${Math.round(top.weight * 100)}%)`;
};

const renderPredictedTable = () => {
  const body = document.getElementById("predicted-table-body");
  if (!body) return;

  // Snapshot expand state
  body.querySelectorAll("[data-pred-toggle]").forEach((btn) => {
    if (btn.dataset.open === "true")
      openPredictedAreas.add(btn.getAttribute("data-pred-toggle"));
    else openPredictedAreas.delete(btn.getAttribute("data-pred-toggle"));
  });

  body.innerHTML = "";

  const chapters = buildPredictedAreas();
  if (!chapters.length) {
    body.innerHTML = `<tr><td colspan="${PREDICTED_COLSPAN}" style="text-align:center;padding:1.5rem;color:var(--color-muted)">No ARENA registry data available.</td></tr>`;
    return;
  }

  chapters.forEach((ch) => {
    const gap = Math.max(0, 100 - ch.avgScore);
    const areaRow = document.createElement("tr");
    areaRow.className = "stats-row stats-row-top";
    areaRow.innerHTML = `
      <td class="stats-col-toggle">
        <button class="stats-toggle" type="button" data-pred-toggle="${ch.id}">▸</button>
      </td>
      <td class="stats-col-check">
        <input type="checkbox" class="stats-check" checked />
      </td>
      <td>${ch.rank}</td>
      <td class="stats-col-area">${ch.label}</td>
      <td class="stats-col-weight">${ch.problems.length} prob.</td>
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
    `;
    body.appendChild(areaRow);

    ch.problems.forEach((p, index) => {
      const score = Number(p.readinessScore) || 0;
      const subGap = Math.max(0, 100 - score);
      const subRow = document.createElement("tr");
      subRow.className = "stats-row stats-subrow hidden";
      subRow.dataset.predSubareaFor = ch.id;
      const sectionTitle = `${p.sectionLabel || ""}${p.title ? " — " + p.title : ""}`.trim();
      subRow.innerHTML = `
        <td class="stats-col-toggle"></td>
        <td class="stats-col-check">
          <input type="checkbox" class="stats-check" checked />
        </td>
        <td>${ch.rank}.${index + 1}</td>
        <td class="stats-col-area stats-subarea">${sectionTitle}</td>
        <td class="stats-col-weight">${p.readinessLabel || "—"}</td>
        <td class="stats-col-score">
          <div class="stats-bar">
            <div class="stats-bar-track">
              <div class="stats-bar-fill" style="width: ${score}%"></div>
            </div>
            <span class="stats-bar-value">${score.toFixed(0)}%</span>
          </div>
        </td>
        <td class="stats-col-solved">${(p.skillWeights || []).length}</td>
        <td class="stats-col-lr">${topSkillLabel(p)}</td>
        <td class="stats-col-delta">
          <div class="stats-bar">
            <div class="stats-bar-track">
              <div class="stats-bar-fill stats-bar-fill-delta" style="width: ${subGap}%"></div>
            </div>
            <span class="stats-bar-value">${subGap.toFixed(0)}%</span>
          </div>
        </td>
      `;
      body.appendChild(subRow);
    });
  });

  // Expand/collapse wiring (mirrors Areas)
  body.querySelectorAll("[data-pred-toggle]").forEach((btn) => {
    const areaId = btn.getAttribute("data-pred-toggle");
    if (openPredictedAreas.has(areaId)) {
      btn.dataset.open = "true";
      btn.textContent = "▾";
      body
        .querySelectorAll(`[data-pred-subarea-for="${areaId}"]`)
        .forEach((row) => row.classList.remove("hidden"));
    }
    btn.addEventListener("click", () => {
      const isOpen = btn.dataset.open === "true";
      btn.dataset.open = isOpen ? "false" : "true";
      btn.textContent = isOpen ? "▸" : "▾";
      if (isOpen) openPredictedAreas.delete(areaId);
      else openPredictedAreas.add(areaId);
      body
        .querySelectorAll(`[data-pred-subarea-for="${areaId}"]`)
        .forEach((row) => row.classList.toggle("hidden", isOpen));
    });
  });
};
