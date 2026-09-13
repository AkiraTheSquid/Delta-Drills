/* Shared answered-problem bars for Learner Home and the group board. */
window.DDActivityBars = {
  render(el, payload, labels) {
    const days = payload.days || [];
    const max = Math.max(1, ...days.map((day) => Number(day.count) || 0));
    el.replaceChildren();
    days.forEach((day, i) => {
      const count = Number(day.count) || 0;
      const col = document.createElement("div");
      col.className = `activity-day${day.date === payload.today ? " is-today" : ""}`;
      col.title = `${day.date}: ${count} answered — ${Number(day.practice) || 0} practice, ${Number(day.placement) || 0} placement`;
      col.tabIndex = 0;
      col.setAttribute("aria-label", col.title);
      const label = document.createElement("span");
      label.className = "activity-day-count";
      label.textContent = count > 0 ? String(count) : "";
      const track = document.createElement("div");
      track.className = "activity-day-track";
      const bar = document.createElement("i");
      bar.className = "activity-day-bar";
      bar.style.height = count > 0 ? `${Math.max(6, count / max * 100)}%` : "0";
      track.appendChild(bar);
      const letter = document.createElement("span");
      letter.className = "activity-day-letter";
      letter.textContent = labels?.[i] || day.date;
      col.append(label, track, letter);
      el.appendChild(col);
    });
    el.setAttribute("aria-label", `Problems answered: ${days.map((d) => `${d.date}: ${d.count}`).join(", ")}. Total ${payload.total || 0}.`);
  },
};
