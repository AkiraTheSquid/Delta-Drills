/* ================================================================
   STATS RENDER — tables
   ================================================================ */

const openAreas = new Set();
const openAreasAdv = new Set();

// Sets a number input's width to exactly fit its digit count
const fitInputWidth = (input) => {
  input.style.width = Math.max(1, String(input.value).length) + "ch";
};

const renderStatsTable = () => {
  if (!statsTableBody) return;

  // Snapshot which areas are currently open before destroying DOM
  statsTableBody.querySelectorAll("[data-area-toggle]").forEach((btn) => {
    if (btn.dataset.open === "true") openAreas.add(btn.getAttribute("data-area-toggle"));
    else openAreas.delete(btn.getAttribute("data-area-toggle"));
  });

  statsTableBody.innerHTML = "";

  if (!statsData.length) {
    statsTableBody.innerHTML =
      '<tr><td colspan="9" style="text-align:center;padding:1.5rem;color:var(--color-muted)">No data yet — start practising to see your stats.</td></tr>';
    return;
  }

  const maxDelta = statsData.reduce((maxArea, area) => {
    const subMax = area.subareas.reduce((max, s) => Math.max(max, s.delta), 0);
    return Math.max(maxArea, subMax);
  }, 0);

  statsData.forEach((area) => {
    const areaDelta = area.subareas.reduce((max, s) => Math.max(max, s.delta), 0);
    const areaDeltaWidth = maxDelta > 0 ? (areaDelta / maxDelta) * 100 : 0;

    const areaRow = document.createElement("tr");
    areaRow.className = "stats-row stats-row-top";
    areaRow.innerHTML = `
      <td class="stats-col-toggle">
        <button class="stats-toggle" type="button" data-area-toggle="${area.id}">▸</button>
      </td>
      <td class="stats-col-check">
        <input
          type="checkbox"
          class="stats-check"
          data-topic-check="${area.area}"
          ${area.enabled ? "checked" : ""}
        />
      </td>
      <td>${area.rank}</td>
      <td class="stats-col-area">${area.area}</td>
      <td class="stats-col-weight">
        <input type="number" class="weight-input"
               data-weight-topic="${area.area}"
               value="${area.displayPct}"
               min="0" max="999" step="1" />%
      </td>
      <td class="stats-col-score">
        <div class="stats-bar">
          <div class="stats-bar-track">
            <div class="stats-bar-fill" style="width: ${area.currentScore}%"></div>
          </div>
          <span class="stats-bar-value">${area.currentScore.toFixed(0)}/100</span>
        </div>
      </td>
      <td class="stats-col-solved">${area.solved}</td>
      <td class="stats-col-lr">${area.learningRate.toFixed(2)}</td>
      <td class="stats-col-delta">
        <div class="stats-bar">
          <div class="stats-bar-track">
            <div class="stats-bar-fill stats-bar-fill-delta" style="width: ${areaDeltaWidth}%"></div>
          </div>
          <span class="stats-bar-value">${areaDelta.toFixed(3)}</span>
        </div>
      </td>
    `;
    statsTableBody.appendChild(areaRow);

    area.subareas.forEach((sub, index) => {
      const subDeltaWidth = maxDelta > 0 ? (sub.delta / maxDelta) * 100 : 0;
      const subRow = document.createElement("tr");
      subRow.className = "stats-row stats-subrow hidden";
      subRow.dataset.subareaFor = area.id;
      subRow.innerHTML = `
        <td class="stats-col-toggle"></td>
        <td class="stats-col-check">
          <input
            type="checkbox"
            class="stats-check"
            data-subtopic-check="${sub.id}"
            data-parent-topic="${area.area}"
            ${sub.enabled ? "checked" : ""}
          />
        </td>
        <td>${area.rank}.${index + 1}</td>
        <td class="stats-col-area stats-subarea">${sub.label}</td>
        <td class="stats-col-weight">
          <input type="number" class="weight-input"
                 data-weight-subtopic="${sub.id}"
                 data-parent-topic="${area.area}"
                 value="${sub.displayPct}"
                 min="0" max="999" step="1" />% × ${area.displayPct}%
        </td>
        <td class="stats-col-score">
          <div class="stats-bar">
            <div class="stats-bar-track">
              <div class="stats-bar-fill" style="width: ${sub.currentScore}%"></div>
            </div>
            <span class="stats-bar-value">${sub.currentScore.toFixed(0)}/100</span>
          </div>
        </td>
        <td class="stats-col-solved">${sub.solved}</td>
        <td class="stats-col-lr">${sub.learningRate.toFixed(2)}</td>
        <td class="stats-col-delta">
          <div class="stats-bar">
            <div class="stats-bar-track">
              <div class="stats-bar-fill stats-bar-fill-delta" style="width: ${subDeltaWidth}%"></div>
            </div>
            <span class="stats-bar-value">${sub.delta.toFixed(3)}</span>
          </div>
        </td>
      `;
      statsTableBody.appendChild(subRow);
    });
  });

  // --- Toggle expand/collapse ---
  statsTableBody.querySelectorAll("[data-area-toggle]").forEach((btn) => {
    // Restore previously open areas
    const areaId = btn.getAttribute("data-area-toggle");
    if (openAreas.has(areaId)) {
      btn.dataset.open = "true";
      btn.textContent = "▾";
      statsTableBody.querySelectorAll(`[data-subarea-for="${areaId}"]`).forEach((row) => {
        row.classList.remove("hidden");
      });
    }

    btn.addEventListener("click", () => {
      const isOpen = btn.dataset.open === "true";
      btn.dataset.open = isOpen ? "false" : "true";
      btn.textContent = isOpen ? "▸" : "▾";
      if (isOpen) openAreas.delete(areaId);
      else openAreas.add(areaId);
      statsTableBody.querySelectorAll(`[data-subarea-for="${areaId}"]`).forEach((row) => {
        row.classList.toggle("hidden", isOpen);
      });
    });
  });

  // --- Editable weight inputs ---
  statsTableBody.querySelectorAll(".weight-input").forEach((input) => {
    // Size the input to exactly fit its current digits
    fitInputWidth(input);

    // Resize as the user types
    input.addEventListener("input", () => fitInputWidth(input));

    // Enter blurs without doing anything else (prevents toggle button activation)
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        input.blur();
      }
    });

    input.addEventListener("change", async () => {
      const val = Math.max(0, Number(input.value) || 0);
      input.value = val;
      fitInputWidth(input);

      let weights = loadWeights();
      const topicKey = input.dataset.weightTopic;
      const subtopicKey = input.dataset.weightSubtopic;

      if (topicKey) {
        weights.topics[topicKey] = val;
      } else if (subtopicKey) {
        weights.subtopics[subtopicKey] = val;
      }

      weights = await persistWeights(weights);

      // Rebuild from cache (no re-fetch) and re-render
      if (rawSubtopicsCache) {
        statsData = buildAreas(rawSubtopicsCache, weights);
        renderStatsTable();
        renderAdvancedTable();
        pushWeightsToBackend(statsData);
      }
      if (typeof syncAdaptiveWeightsToPracticePreferences === "function") {
        await syncAdaptiveWeightsToPracticePreferences();
      }
    });
  });

  statsTableBody.querySelectorAll("[data-topic-check]").forEach((input) => {
    input.addEventListener("change", async () => {
      const topicName = input.dataset.topicCheck;
      const checked = input.checked;
      let weights = setTopicEnabled(topicName, checked);

      if (rawSubtopicsCache) {
        rawSubtopicsCache
          .filter((item) => (item.topic || item.subtopic.split(":")[0].trim()) === topicName)
          .forEach((item) => {
            weights = setSubtopicEnabled(item.subtopic, checked, weights);
          });
      }

      weights = await persistWeights(weights);

      if (rawSubtopicsCache) {
        statsData = buildAreas(rawSubtopicsCache, weights);
        renderStatsTable();
        renderAdvancedTable();
        pushWeightsToBackend(statsData);
      }
      if (typeof syncAdaptiveWeightsToPracticePreferences === "function") {
        await syncAdaptiveWeightsToPracticePreferences();
      }
    });
  });

  statsTableBody.querySelectorAll("[data-subtopic-check]").forEach((input) => {
    input.addEventListener("change", async () => {
      const subtopicId = input.dataset.subtopicCheck;
      const topicName = input.dataset.parentTopic;
      let weights = setSubtopicEnabled(subtopicId, input.checked);

      if (input.checked && !isTopicEnabled(topicName, weights)) {
        weights = setTopicEnabled(topicName, true, weights);
      }

      weights = await persistWeights(weights);

      if (rawSubtopicsCache) {
        statsData = buildAreas(rawSubtopicsCache, weights);
        renderStatsTable();
        renderAdvancedTable();
        pushWeightsToBackend(statsData);
      }
      if (typeof syncAdaptiveWeightsToPracticePreferences === "function") {
        await syncAdaptiveWeightsToPracticePreferences();
      }
    });
  });
};

const renderAdvancedTable = () => {
  const body = document.getElementById("adv-table-body");
  if (!body) return;

  // Snapshot which areas are currently open before destroying DOM
  body.querySelectorAll("[data-adv-toggle]").forEach((btn) => {
    if (btn.dataset.open === "true") openAreasAdv.add(btn.getAttribute("data-adv-toggle"));
    else openAreasAdv.delete(btn.getAttribute("data-adv-toggle"));
  });

  body.innerHTML = "";

  if (!statsData.length) {
    body.innerHTML =
      '<tr><td colspan="11" style="text-align:center;padding:1.5rem;color:var(--color-muted)">No data yet — start practising to see your stats.</td></tr>';
    return;
  }

  const maxDelta = statsData.reduce((maxArea, area) => {
    const subMax = area.subareas.reduce((max, s) => Math.max(max, s.delta), 0);
    return Math.max(maxArea, subMax);
  }, 0);

  statsData.forEach((area) => {
    const areaDelta = area.subareas.reduce((max, s) => Math.max(max, s.delta), 0);
    const areaDeltaWidth = maxDelta > 0 ? (areaDelta / maxDelta) * 100 : 0;

    const areaRow = document.createElement("tr");
    areaRow.className = "stats-row stats-row-top";
    areaRow.innerHTML = `
      <td class="stats-col-toggle">
        <button class="stats-toggle" type="button" data-adv-toggle="${area.id}">▸</button>
      </td>
      <td class="stats-col-check">
        <input type="checkbox" class="stats-check" disabled ${area.enabled ? "checked" : ""} />
      </td>
      <td class="stats-col-rank">${area.rank}</td>
      <td class="stats-col-area">${area.area}</td>
      <td class="stats-col-score">
        <div class="stats-bar">
          <div class="stats-bar-track">
            <div class="stats-bar-fill" style="width: ${area.currentScore}%"></div>
          </div>
          <span class="stats-bar-value">${area.currentScore.toFixed(0)}/100</span>
        </div>
      </td>
      <td class="stats-col-solved">${area.solved}</td>
      <td class="stats-col-lr">${area.learningRate.toFixed(2)}</td>
      <td class="stats-col-delta">
        <div class="stats-bar">
          <div class="stats-bar-track">
            <div class="stats-bar-fill stats-bar-fill-delta" style="width: ${areaDeltaWidth}%"></div>
          </div>
          <span class="stats-bar-value">${areaDelta.toFixed(3)}</span>
        </div>
      </td>
      <td class="stats-col-p">${(area.p * 100).toFixed(1)}%</td>
      <td class="stats-col-target">${area.targetDifficulty.toFixed(1)}</td>
      <td class="stats-col-mult">${area.difficultyMultiplier.toFixed(2)}×</td>
    `;
    body.appendChild(areaRow);

    area.subareas.forEach((sub, index) => {
      const subDeltaWidth = maxDelta > 0 ? (sub.delta / maxDelta) * 100 : 0;
      const subRow = document.createElement("tr");
      subRow.className = "stats-row stats-subrow hidden";
      subRow.dataset.advSubareaFor = area.id;
      subRow.innerHTML = `
        <td class="stats-col-toggle"></td>
        <td class="stats-col-check">
          <input type="checkbox" class="stats-check" disabled ${sub.enabled ? "checked" : ""} />
        </td>
        <td class="stats-col-rank">${area.rank}.${index + 1}</td>
        <td class="stats-col-area stats-subarea">${sub.label}</td>
        <td class="stats-col-score">
          <div class="stats-bar">
            <div class="stats-bar-track">
              <div class="stats-bar-fill" style="width: ${sub.currentScore}%"></div>
            </div>
            <span class="stats-bar-value">${sub.currentScore.toFixed(0)}/100</span>
          </div>
        </td>
        <td class="stats-col-solved">${sub.solved}</td>
        <td class="stats-col-lr">${sub.learningRate.toFixed(2)}</td>
        <td class="stats-col-delta">
          <div class="stats-bar">
            <div class="stats-bar-track">
              <div class="stats-bar-fill stats-bar-fill-delta" style="width: ${subDeltaWidth}%"></div>
            </div>
            <span class="stats-bar-value">${sub.delta.toFixed(3)}</span>
          </div>
        </td>
        <td class="stats-col-p">${(sub.p * 100).toFixed(1)}%</td>
        <td class="stats-col-target">${sub.targetDifficulty.toFixed(1)}</td>
        <td class="stats-col-mult">${sub.difficultyMultiplier.toFixed(2)}×</td>
      `;
      body.appendChild(subRow);
    });
  });

  // Toggle expand/collapse
  body.querySelectorAll("[data-adv-toggle]").forEach((btn) => {
    const areaId = btn.getAttribute("data-adv-toggle");
    if (openAreasAdv.has(areaId)) {
      btn.dataset.open = "true";
      btn.textContent = "▾";
      body.querySelectorAll(`[data-adv-subarea-for="${areaId}"]`).forEach((row) => {
        row.classList.remove("hidden");
      });
    }
    btn.addEventListener("click", () => {
      const isOpen = btn.dataset.open === "true";
      btn.dataset.open = isOpen ? "false" : "true";
      btn.textContent = isOpen ? "▸" : "▾";
      if (isOpen) openAreasAdv.delete(areaId);
      else openAreasAdv.add(areaId);
      body.querySelectorAll(`[data-adv-subarea-for="${areaId}"]`).forEach((row) => {
        row.classList.toggle("hidden", isOpen);
      });
    });
  });
};
