/* ARENA share of practice — account-scoped, like practice-target.js: the
   server owns the number (POST /api/practice/arena-share) and the picker
   reads it on every /next-question (backend app/arena_mix.py). No
   localStorage: a shared browser must not inherit another learner's mix. */
(() => {
  const slider = document.getElementById("account-arena-share");
  const out = document.getElementById("account-arena-share-out");
  const note = document.getElementById("account-arena-share-note");
  if (!slider || !out || !note) return;
  let generation = 0;
  let current = null;
  const pct = (x) => `${Math.round((x || 0) * 100)}%`;
  const paint = (data) => {
    current = data;
    slider.value = String(Math.round((data.share || 0) * 100));
    out.textContent = pct(data.share);
    const scope = data.scope === "raytracing-0.1" ? "section 0.1" : "the whole course";
    if (!data.share) {
      note.textContent = `Off — every drill comes from the graph's frontier (prerequisites first, mastery-gated). Set a share to mix in ${scope}'s own exercise concepts before the gate says you are ready.`;
      return;
    }
    const recent = data.recent_fraction === null || data.recent_fraction === undefined
      ? "no answers yet" : `${pct(data.recent_fraction)} of your last ${data.window} answers`;
    const turn = data.next_turn === "arena" ? "an ARENA exercise concept" : "the graph (prerequisites)";
    note.textContent = `${pct(data.share)} of your drills come from ${scope}'s ${data.arena_kcs} ARENA exercise concepts — the course's own problems and alternatives on the same concept, unlocked early — the rest drill the graph. Lately: ${recent}. Next drill: ${turn}.`;
  };
  const request = async (share) => {
    const res = await apiFetch("/api/practice/arena-share", share === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ share }),
    });
    if (!res.ok) throw new Error("Could not load the ARENA share. Check your connection and sign-in.");
    return res.json();
  };
  const refresh = async () => {
    const mine = ++generation;
    try {
      const data = await request();
      if (mine !== generation) return;
      paint(data);
      slider.disabled = false;
    } catch (err) {
      if (mine === generation) { slider.disabled = true; note.textContent = err.message; }
    }
  };
  slider.addEventListener("input", () => { out.textContent = `${slider.value}%`; });
  slider.addEventListener("change", async () => {
    const mine = ++generation;
    slider.disabled = true;
    try {
      const data = await request(Number(slider.value) / 100);
      if (mine !== generation) return;
      paint(data);
      await window.deltaRefreshKcLattice?.();
      window.dispatchEvent(new CustomEvent("delta:adaptive-state-changed"));
    } catch (err) {
      if (current) paint(current);
      note.textContent = err.message;
    } finally { slider.disabled = false; }
  });
  window.addEventListener("delta:practice-mode-ready", refresh);
  window.addEventListener("delta-drills-diagnostic-status", refresh);
  if (window.DDPracticeModeReady) refresh();
})();
