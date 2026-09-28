/* Per-member activity and competency charts, in the checklist's column. */
window.DDGroupProgress = (() => {
  const selections = new Map();
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const svgEl = (tag, attrs = {}, text) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
  const shortDate = (key) => window.DDGroupsDay.dateOf(key).toLocaleDateString(undefined, { month: "short", day: "numeric" });

  function build({ member, isYou, view, progress, progressState }) {
    const chart = el("div", "dd-member-day dd-progress");
    const selector = view === "graph" ? el("div", "dd-graph-legend") : null;
    chart.appendChild(el("div", "dd-member-day-title", view === "graph" ? "Competency over time" : "Problems per day"));
    const data = progress?.entries?.[member.member_id];
    if (progressState !== "ready" || !data) {
      chart.appendChild(el("p", "dd-group-note", progressState === "failed" || progressState === "ready"
        ? "Progress could not be read. Retry above." : "Reading progress…"));
      return { chart, selector };
    }
    if (view === "activity") {
      chart.appendChild(el("p", "dd-progress-total", `${data.activity.total} answered this ${ { daily: "day", weekly: "week", monthly: "month" }[progress.horizon]}`));
      const scroller = el("div", "dd-chart-scroll");
      const bars = el("div", "activity-week-chart dd-activity-chart");
      const labels = data.activity.days.map((day) => progress.horizon === "weekly"
        ? window.DDGroupsDay.dateOf(day.date).toLocaleDateString(undefined, { weekday: "short" })
        : String(Number(day.date.slice(-2))));
      window.DDActivityBars.render(bars, data.activity, labels);
      scroller.appendChild(bars);
      chart.append(scroller, el("p", "dd-group-note", `${shortDate(progress.start)} – ${shortDate(progress.end)} · Days`));
      return { chart, selector };
    }

    const areas = progress.areas || [];
    const aggregate = { id: "aggregate", label: "All scores · aggregate", color: "#78d4bb" };
    const choices = [aggregate, ...areas, { id: "all", label: "All areas · compare", color: "var(--muted)" }];
    let selected = selections.get(member.member_id) || "aggregate";
    if (!choices.some((area) => area.id === selected)) selected = "aggregate";
    const body = el("div", "dd-graph-body");
    chart.appendChild(body);
    selector.setAttribute("aria-label", `Graph areas for ${member.display_name || "learner"}`);

    function paint() {
      body.replaceChildren();
      selector.replaceChildren(el("div", "placement-areas-head", "Display area"));
      const history = data.history || [];
      choices.forEach((area) => {
        const button = el("button", "dd-graph-area");
        button.type = "button";
        button.setAttribute("aria-pressed", String(selected === area.id));
        const dot = el("i", "dd-graph-swatch");
        dot.style.background = area.color;
        const latest = [...history].reverse().find((day) => day.scores?.[area.id])?.scores[area.id];
        button.append(dot, el("span", "dd-graph-area-name", area.label),
          el("span", "dd-graph-area-score", area.id === "all" ? "" : latest ? `${Math.round(latest.score)}` : "—"));
        button.addEventListener("click", () => {
          selected = area.id;
          selections.set(member.member_id, selected);
          paint();
        });
        selector.appendChild(button);
      });
      const chosen = choices.find((area) => area.id === selected);
      const shown = selected === "all" ? areas : [chosen];
      const latest = [...history].reverse().find((day) => day.scores?.[selected])?.scores[selected];
      body.appendChild(el("p", "dd-graph-summary", selected === "all"
        ? "Compare every area · target setting unavailable in this view"
        : `${chosen.label}${latest ? ` · ${Math.round(latest.score)}/100 · ${latest.coverage}% evidence coverage · ${latest.proxies}/${latest.concepts} concepts use proxies or priors` : " · No recorded measurements in this period"}`));

      const W = 720, H = 286, L = 48, R = 692, T = 30, B = 236;
      const n = history.length;
      const x = (i) => n < 2 ? (L + R) / 2 : L + i / (n - 1) * (R - L);
      const y = (score) => B - score / 100 * (B - T);
      const scroller = el("div", "dd-chart-scroll");
      const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "dd-competency-chart", role: "img",
        "aria-label": `${member.display_name || "Learner"}: ${chosen.label}. Time on x-axis; normalized mastery from 0 to 100 on y-axis.` });
      svg.appendChild(svgEl("title", {}, `${chosen.label} — ${progress.start} to ${progress.end}`));
      [0, 20, 40, 60, 80, 100].forEach((score) => {
        svg.appendChild(svgEl("line", { x1: L, x2: R, y1: y(score), y2: y(score), class: "dd-graph-grid" }));
        svg.appendChild(svgEl("text", { x: L - 10, y: y(score) + 4, "text-anchor": "end", class: "dd-graph-label" }, score));
      });
      svg.appendChild(svgEl("text", { x: L, y: 15, class: "dd-graph-label" }, "Mastery / 100"));
      const benchmark = progress.benchmark ?? 80;
      svg.appendChild(svgEl("line", { x1: L, x2: R, y1: y(benchmark), y2: y(benchmark), class: "dd-graph-benchmark" }));
      svg.appendChild(svgEl("text", { x: R, y: y(benchmark) - 7, "text-anchor": "end", class: "dd-graph-label" }, `${benchmark} · ARENA reference`));
      const tickEvery = Math.max(1, Math.ceil(n / 7));
      history.forEach((day, i) => {
        if (i % tickEvery !== 0 && i !== n - 1) return;
        svg.appendChild(svgEl("text", { x: x(i), y: B + 22, "text-anchor": "middle", class: "dd-graph-label" }, shortDate(day.date)));
      });
      svg.appendChild(svgEl("text", { x: (L + R) / 2, y: H - 6, "text-anchor": "middle", class: "dd-graph-label" }, "Time · days"));
      let hasPoints = false;
      shown.forEach((area) => {
        let path = "", connected = false;
        history.forEach((day, i) => {
          const reading = day.scores?.[area.id];
          if (!Number.isFinite(reading?.score)) { connected = false; return; }
          hasPoints = true;
          path += `${connected ? "L" : "M"}${x(i)},${y(reading.score)} `;
          connected = true;
          const point = svgEl("circle", { cx: x(i), cy: y(reading.score), r: n > 14 ? 2 : 3.5, fill: area.color });
          point.appendChild(svgEl("title", {}, `${day.date} · ${area.label}: ${reading.score}/100 · coverage ${reading.coverage}%`));
          svg.appendChild(point);
        });
        svg.appendChild(svgEl("path", { d: path, fill: "none", stroke: area.color, "stroke-width": 2.5,
          "stroke-linejoin": "round", "stroke-linecap": "round" }));
      });
      if (!hasPoints) svg.appendChild(svgEl("text", { x: (L + R) / 2, y: y(45), "text-anchor": "middle", class: "dd-graph-label" }, "No recorded measurements in this period"));

      // A crosshair also names its two coordinates ON the axes: the date
      // sits over the x-axis tick row, the level over the y-axis labels,
      // each on a filled tag so it reads as "this one" against the ticks.
      const axisTag = (label, cx, cy, anchor) => {
        const w = label.length * 7 + 12, h = 18;
        const left = anchor === "end" ? cx - w : cx - w / 2;
        const tag = svgEl("g", { class: "dd-graph-axis-tag" });
        tag.append(svgEl("rect", { x: left, y: cy - 13, width: w, height: h, rx: 4 }),
          svgEl("text", { x: left + w / 2, y: cy, "text-anchor": "middle" }, label));
        return tag;
      };
      const crosshair = (date, level, cls) => {
        const i = history.findIndex((day) => day.date === date);
        const g = svgEl("g", { class: cls, "pointer-events": "none" });
        if (i < 0) return g;
        g.append(svgEl("line", { x1: x(i), x2: x(i), y1: T, y2: B }),
          svgEl("line", { x1: L, x2: R, y1: y(level), y2: y(level) }),
          svgEl("circle", { cx: x(i), cy: y(level), r: 5 }),
          axisTag(shortDate(date), x(i), B + 22, "middle"),
          axisTag(String(level), L - 4, y(level) + 4, "end"));
        return g;
      };
      const target = selected === "all" ? null : data.targets?.[selected];
      if (target) {
        svg.appendChild(crosshair(target.date, target.level, "dd-graph-target"));
        body.appendChild(el("p", "dd-target-saved", `Target: ${target.level}/100 by ${shortDate(target.date)}${target.date < progress.start || target.date > progress.end ? " · outside this period" : ""}`));
      }
      scroller.appendChild(svg);
      body.appendChild(scroller);
      if (!isYou || selected === "all" || !n) return;

      body.appendChild(el("p", "dd-group-note", "Hover to preview a target; click to choose. Save below. Keyboard: arrows move the crosshair; Enter chooses."));
      const form = el("form", "dd-target-form");
      const dateLabel = el("label", "", "Target day");
      const dateInput = el("input", "dd-target-date");
      dateInput.type = "date";
      dateInput.required = true;
      dateInput.value = target?.date || progress.end;
      dateLabel.appendChild(dateInput);
      const levelLabel = el("label", "", "Mastery / 100");
      const levelInput = el("input", "dd-target-level");
      levelInput.type = "number";
      levelInput.min = "0";
      levelInput.max = "100";
      levelInput.step = "1";
      levelInput.required = true;
      levelInput.value = String(target?.level ?? benchmark);
      levelLabel.appendChild(levelInput);
      const save = el("button", "dd-target-save", "Save target");
      save.type = "submit";
      const status = el("p", "dd-target-status", "");
      status.setAttribute("role", "status");
      form.append(dateLabel, levelLabel, save);
      body.append(form, status);

      let preview = null, candidate = { date: dateInput.value, level: Number(levelInput.value) };
      let pending = false;
      const showPreview = () => {
        preview?.remove();
        preview = crosshair(candidate.date, candidate.level, "dd-graph-crosshair");
        svg.appendChild(preview);
        status.textContent = `Preview: ${candidate.level}/100 by ${shortDate(candidate.date)}`;
      };
      const pointFrom = (event) => {
        const matrix = svg.getScreenCTM();
        if (!matrix) return null;
        const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
        if (point.x < L || point.x > R || point.y < T || point.y > B) return null;
        return { date: history[clamp(Math.round((point.x - L) / (R - L) * (n - 1)), 0, n - 1)].date,
          level: clamp(Math.round((B - point.y) / (B - T) * 100), 0, 100) };
      };
      svg.addEventListener("pointermove", (event) => {
        if (pending) return;
        const point = pointFrom(event);
        if (point) { candidate = point; if (!status.classList.contains("is-error")) showPreview(); }
      });
      svg.addEventListener("pointerleave", () => { preview?.remove(); });
      const choose = () => {
        dateInput.value = candidate.date;
        levelInput.value = String(candidate.level);
        status.textContent = `Selected: ${candidate.level}/100 by ${shortDate(candidate.date)}. Save target to keep it.`;
      };
      svg.addEventListener("click", (event) => {
        if (pending) return;
        const point = pointFrom(event);
        if (point) { candidate = point; showPreview(); choose(); }
      });
      svg.setAttribute("tabindex", "0");
      svg.addEventListener("keydown", (event) => {
        if (pending || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Enter"].includes(event.key)) return;
        event.preventDefault();
        if (event.key === "Enter") { choose(); return; }
        let i = Math.max(0, history.findIndex((day) => day.date === candidate.date));
        if (event.key === "ArrowLeft") i -= 1;
        if (event.key === "ArrowRight") i += 1;
        candidate = { date: history[clamp(i, 0, n - 1)].date,
          level: clamp(candidate.level + (event.key === "ArrowUp" ? 1 : event.key === "ArrowDown" ? -1 : 0), 0, 100) };
        showPreview();
      });
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (pending || !form.reportValidity()) return;
        pending = true;
        save.disabled = true;
        status.classList.remove("is-error");
        const area = selected;
        const requested = { date: dateInput.value, level: Number(levelInput.value) };
        status.textContent = "Saving target…";
        const answer = await window.DDGroupStore.saveTarget(area, requested.date, requested.level);
        pending = false;
        save.disabled = false;
        if (answer.error) {
          status.textContent = `Not saved: ${answer.error}`;
          status.classList.add("is-error");
          return;
        }
        data.targets ||= {};
        data.targets[area] = answer.target;
        // A section change can happen during the write; repaint the current
        // selection, without moving the target to the newly selected area.
        if (chart.isConnected) paint();
      });
    }
    paint();
    return { chart, selector };
  }
  return { build, reset: () => selections.clear() };
})();
