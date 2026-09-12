/* Account-scoped curriculum focus. No localStorage: shared browsers must not
   inherit another learner's target. The server filters the actual queue. */
(() => {
  const picker = document.getElementById("practice-target");
  const note = document.getElementById("practice-target-note");
  const diagnosticBtn = document.getElementById("ray-placement-btn");
  if (!picker || !note || !diagnosticBtn) return;
  let generation = 0;
  let current = null;
  const paint = (data) => {
    current = data;
    picker.value = data.target;
    const focused = data.target === "raytracing-0.1";
    const ids = new Set(data.kcs || []);
    const cy = window.deltaConceptGraphCy?.();
    cy?.nodes().forEach((n) => {
      if (focused && !ids.has(n.id())) n.style("display", "none");
      else n.removeStyle("display");
    });
    cy?.edges().forEach((e) => {
      if (focused && (!ids.has(e.source().id()) || !ids.has(e.target().id()))) e.style("display", "none");
      else e.removeStyle("display");
    });
    note.textContent = focused
      ? `0.1 + ${ids.size} concepts including prerequisites. ${data.placement_ready?.length || 0} ready from placement (includes inferred prerequisites). Applies to your next practice question.`
      : "Practice across the curriculum.";
  };
  const request = async (target) => {
    const res = await apiFetch("/api/practice/practice-target", target === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ target }),
    });
    if (!res.ok) throw new Error("Could not load practice focus. Check your connection and sign-in.");
    return res.json();
  };
  const refresh = async () => {
    const mine = ++generation;
    try {
      const data = await request();
      if (mine !== generation) return;
      paint(data);
      picker.disabled = false;
    } catch (err) {
      if (mine === generation) { picker.disabled = true; note.textContent = err.message; }
    }
  };
  picker.addEventListener("change", async () => {
    const mine = ++generation;
    picker.disabled = true;
    try {
      const data = await request(picker.value);
      if (mine !== generation) return;
      paint(data);
      await window.deltaRefreshKcLattice?.();
      window.dispatchEvent(new CustomEvent("delta:adaptive-state-changed"));
    } catch (err) {
      if (current) picker.value = current.target;
      note.textContent = err.message;
    } finally { picker.disabled = false; }
  });
  diagnosticBtn.addEventListener("click", () => {
    window.PlacementPlan?.setScope?.("raytracing-0.1");
    const go = typeof switchTab !== "undefined" ? switchTab : window.switchTab;
    go?.("placement");
    window.DiagnosticPage?.refresh?.();
  });
  window.addEventListener("delta:practice-mode-ready", refresh);
  window.addEventListener("delta-drills-diagnostic-status", refresh);
  window.addEventListener("delta:practice-target-graph-ready", () => { if (current) paint(current); });
  if (window.DDPracticeModeReady) refresh();
})();
