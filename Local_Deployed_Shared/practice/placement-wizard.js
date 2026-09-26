/* ================================================================
   PLACEMENT SET-UP — a short run of questions before the test starts.

   Seth, 2026-09-24: the placement card "looks ugly as hell" — a status dot,
   two rows of chips, a native <select>, three self-report pills and a row of
   plan buttons, all at once. It becomes one question at a time with Back /
   Next underneath:

     1. areas   "Which areas do you want to improve?" — a learner who already
                knows they do not want linear algebra (or NumPy, or Python)
                leaves it out and it is not tested. Or: the 25-minute Ray
                Tracing 0.1 rapid check instead of a focus.
     2. level   "How much of this have you done before?" — the self-reported
                prior. Only while one can still apply (`can_set_prior`).
     3. length  "How long do you want to spend?" — three lengths CUT FROM THE
                FOCUS. /diagnostic/plan?areas=… answers with a quick /
                standard / full budget for exactly that set of concepts
                (app/placement_scope.py), so a two-area placement is offered
                a shorter test than the whole curriculum.

   The steps, the self-report buttons (practice/adaptive.js binds them) and
   the start button (practice/advance-events.js binds it) are STATIC in
   index.html so their handlers survive every repaint; this file only shows
   one step, fills the area and length grids, and says what to send when the
   start button fires (`startPayload`, read by api.js diagnosticStart).

   It is driven by PlacementPlan.render(status) — diagnostic-page.js hands
   every status there — and never fetches status itself.
   ================================================================ */
const PlacementWizard = (() => {
  const byId = (id) => document.getElementById(id);
  const RAY = "raytracing-0.1";
  const STORE_AREAS = "delta_drills_placement_areas";
  const STORE_TIER = "delta_drills_placement_tier";
  const DEFAULT_TIER = "standard";

  /* What each registry topic is called on the card and the one line under
     it. The KEYS are the backend's (kc_registry.json `topic`); an area the
     backend adds later still renders, under its own key and no description. */
  const AREA_TEXT = {
    Python: ["Python", "Functions, loops, comprehensions — the Python ARENA assumes."],
    Numpy: ["NumPy", "Arrays, indexing, broadcasting and vectorisation."],
    PyTorch: ["PyTorch", "Tensor ops, rays, modules and ResNets — ARENA 0.0 to 0.2."],
    Einops: ["einops", "rearrange, reduce, repeat and einsum."],
    Mathematics: ["Math", "Linear algebra and ray geometry for ARENA 0.1."],
    LeetCode: ["LeetCode Patterns", "Two pointers, sliding window, trees, graphs, heaps, DP."],
    "Delta Drills": ["Delta Drills", "How this app works: mastery, spaced repetition, why drills."],
  };
  const areaName = (key) => (AREA_TEXT[key] || [key])[0];
  // Listed in curriculum order, not the registry's; unknown areas go last.
  const AREA_ORDER = Object.keys(AREA_TEXT);
  const byCurriculum = (a, b) =>
    (AREA_ORDER.indexOf(a.key) + 1 || 99) - (AREA_ORDER.indexOf(b.key) + 1 || 99);
  /* "Full calibration" only when it IS one: on a wide focus even six hours
     cannot settle every concept, and the name would promise what the
     coverage bar underneath contradicts. */
  const tierName = (opt) =>
    opt.key === "full"
      ? (Number(opt.coverage) >= 1 ? "Full calibration" : "Thorough")
      : ({ quick: "Quick check", standard: "Standard" }[opt.key] || opt.key);
  const STEP_LABEL = { areas: "Areas", level: "Experience", length: "Length" };

  let status = null;
  let stepIndex = 0;
  let retakeOpen = false;
  let scope = "all";
  let catalog = null;          // [{key, kcs}] from /diagnostic/plan
  let chosen = null;           // Set of area keys; null until the catalogue arrives
  let tier = null;
  const plans = new Map();     // areas key -> plan response (or a pending promise)
  let catalogGen = 0;          // bumped by coursesChanged; a plan asked before it is dropped

  const _api = () => (typeof PracticeAPI !== "undefined" ? PracticeAPI : window.PracticeAPI);
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const read = (key) => { try { return localStorage.getItem(key); } catch (_) { return null; } };
  const write = (key, value) => { try { localStorage.setItem(key, value); } catch (_) {} };

  const hm = (mins) => {
    const m = Math.max(0, Math.round(Number(mins) || 0));
    const h = Math.floor(m / 60);
    const r = m % 60;
    if (h > 0) return r > 0 ? `${h}h ${r}m` : `${h}h`;
    return `${r} min`;
  };
  const mmss = (secs) => {
    const n = Math.max(0, Math.round(Number(secs) || 0));
    return `${Math.floor(n / 60)}:${String(n % 60).padStart(2, "0")}`;
  };

  /* ---- state ----------------------------------------------------------- */

  const selectedTier = () => {
    if (tier == null) tier = read(STORE_TIER) || DEFAULT_TIER;
    return tier;
  };

  const steps = () =>
    status?.can_set_prior ? ["areas", "level", "length"] : ["areas", "length"];
  const currentStep = () => {
    const list = steps();
    stepIndex = Math.min(Math.max(0, stepIndex), list.length - 1);
    return list[stepIndex];
  };

  // Every area chosen is the whole curriculum: send nothing, so a learner
  // whose catalogue grows later is not pinned to yesterday's list.
  const areasParam = () => {
    if (!chosen || !catalog) return null;
    if (catalog.every((a) => chosen.has(a.key))) return null;
    return catalog.map((a) => a.key).filter((k) => chosen.has(k));
  };
  // Saved as the same answer: every area → null, so a catalogue that grows
  // later still reads as "everything" on the next visit.
  const saveAreas = () => write(STORE_AREAS, JSON.stringify(areasParam()));
  const areasKey = () => (areasParam() || ["*"]).join(",");

  const isOpen = (s = status) =>
    !!s && !s.active && !s.unavailable && (!s.completed_at || retakeOpen);

  /* ---- data ------------------------------------------------------------ */

  const _adoptCatalog = (plan) => {
    if (catalog || !Array.isArray(plan?.area_catalog) || !plan.area_catalog.length) return;
    catalog = plan.area_catalog.slice().sort(byCurriculum);
    let saved = null;
    try { saved = JSON.parse(read(STORE_AREAS) || "null"); } catch (_) {}
    const keys = catalog.map((a) => a.key);
    const kept = Array.isArray(saved) ? saved.filter((k) => keys.includes(k)) : [];
    chosen = new Set(kept.length ? kept : keys);
  };

  const planFor = (key) => {
    const hit = plans.get(key);
    return hit && !(hit instanceof Promise) ? hit : null;
  };

  /* A failed fetch is retried, but not from paint(): paint runs on every
     status and every click, and a backend that is down would be asked once
     per repaint. One retry is scheduled instead, after a pause. */
  const RETRY_MS = 5000;
  let retryTimer = null;
  const loadPlan = () => {
    const key = areasKey();
    if (plans.has(key) || retryTimer) return;
    const params = areasParam();
    const asked = catalogGen;
    const pending = (async () => {
      let plan = null;
      try { plan = await _api()?.diagnosticPlan?.(params); } catch (_) {}
      if (asked !== catalogGen) return; // the courses changed while it was out
      if (plan && !plan.unavailable && Array.isArray(plan.options)) {
        plans.set(key, plan);
        _adoptCatalog(plan);
      } else {
        plans.delete(key);
        retryTimer = setTimeout(() => { retryTimer = null; paint(); }, RETRY_MS);
      }
      paint();
    })();
    plans.set(key, pending);
  };

  /* ---- painting -------------------------------------------------------- */

  const paintSteps = () => {
    const host = byId("placement-steps");
    if (!host) return;
    host.textContent = "";
    steps().forEach((name, i) => {
      const li = el("li", "placement-steps-item");
      li.classList.toggle("is-current", i === stepIndex);
      li.classList.toggle("is-done", i < stepIndex);
      if (i === stepIndex) li.setAttribute("aria-current", "step");
      const btn = el("button", "placement-steps-btn");
      btn.type = "button";
      btn.disabled = i >= stepIndex;
      btn.appendChild(el("span", "placement-steps-num", i < stepIndex ? "✓" : String(i + 1)));
      btn.appendChild(el("span", "placement-steps-label", STEP_LABEL[name]));
      btn.addEventListener("click", () => go(i));
      li.appendChild(btn);
      host.appendChild(li);
    });
  };

  const paintAreas = () => {
    const grid = byId("placement-focus-areas");
    const count = byId("placement-focus-count");
    const all = byId("placement-focus-all");
    const alt = byId("placement-focus-alt");
    if (!grid) return;
    grid.textContent = "";
    const ray = scope === RAY;
    if (!catalog) {
      grid.appendChild(el("p", "placement-step-loading", "Loading areas…"));
    } else {
      catalog.forEach((area) => {
        const [name, blurb] = AREA_TEXT[area.key] || [area.key, ""];
        const on = !ray && chosen.has(area.key);
        const btn = el("button", "placement-focus-tile");
        btn.type = "button";
        btn.dataset.area = area.key;
        btn.setAttribute("aria-pressed", String(on));
        btn.classList.toggle("is-selected", on);
        btn.classList.toggle("is-muted", ray);
        btn.appendChild(el("span", "placement-focus-check"));
        const body = el("span", "placement-focus-body");
        body.appendChild(el("span", "placement-focus-name", name));
        if (blurb) body.appendChild(el("span", "placement-focus-blurb", blurb));
        body.appendChild(el("span", "placement-focus-meta", `${area.kcs} concept${area.kcs === 1 ? "" : "s"}`));
        btn.appendChild(body);
        btn.addEventListener("click", () => {
          if (scope === RAY) {
            scope = "all";
            chosen = new Set([area.key]);
          } else if (chosen.has(area.key)) {
            chosen.delete(area.key);
          } else {
            chosen.add(area.key);
          }
          saveAreas();
          paint();
        });
        grid.appendChild(btn);
      });
    }
    if (all) {
      const every = !!catalog && !ray && catalog.every((a) => chosen.has(a.key));
      all.textContent = every ? "Clear all" : "Select all";
      all.disabled = !catalog;
    }
    if (count) {
      if (ray) count.textContent = "Rapid Ray Tracing 0.1 check selected";
      else if (catalog) {
        const picked = catalog.filter((a) => chosen.has(a.key));
        const kcs = picked.reduce((n, a) => n + (Number(a.kcs) || 0), 0);
        count.textContent = picked.length
          ? `${picked.length} of ${catalog.length} areas · ${kcs} concepts`
          : "Pick at least one area";
        count.classList.toggle("is-warning", !picked.length);
      } else count.textContent = "";
    }
    if (alt) {
      alt.textContent = "";
    }
    // The rapid check is ARENA 0.1's; a learner not studying ARENA has no
    // PyTorch area and nothing for it to test.
    if (alt && catalog && catalog.some((a) => a.key === "PyTorch")) {
      const btn = el("button", "placement-focus-alt-btn");
      btn.type = "button";
      btn.setAttribute("aria-pressed", String(ray));
      btn.classList.toggle("is-selected", ray);
      btn.appendChild(el("span", "placement-focus-check"));
      const body = el("span", "placement-focus-body");
      body.appendChild(el("span", "placement-focus-name", "Only aiming at ARENA 0.1 (Ray Tracing)?"));
      body.appendChild(el("span", "placement-focus-blurb",
        "Take the rapid check instead: up to 8 problems, 25 minutes, then practice targets 0.1."));
      btn.appendChild(body);
      btn.addEventListener("click", () => {
        scope = ray ? "all" : RAY;
        paint();
      });
      alt.appendChild(btn);
    }
  };

  const paintLength = () => {
    const host = byId("placement-length-options");
    const note = byId("placement-length-note");
    const hint = byId("placement-length-hint");
    if (!host) return;
    host.textContent = "";
    if (scope === RAY) {
      if (hint) hint.textContent = "The rapid check has one length. It jumps ahead after passes and checks prerequisites after misses.";
      const tile = el("div", "placement-plan-option is-selected is-static");
      tile.appendChild(el("span", "placement-plan-tier", "Rapid check"));
      tile.appendChild(el("span", "placement-plan-hours", "25 min"));
      tile.appendChild(el("span", "placement-plan-est", "up to 8 unscaffolded problems"));
      host.appendChild(tile);
      if (note) note.textContent = "Starts with ray–segment intersections. Untested concepts stay uncertain.";
      return;
    }
    if (hint) hint.textContent = "Longer tests pin down more concepts. It stops early once every concept is settled, and never runs past the time you pick.";
    const plan = planFor(areasKey());
    if (!plan) {
      loadPlan();
      ["quick", "standard", "full"].forEach((k) => {
        const tile = el("div", "placement-plan-option is-loading");
        tile.appendChild(el("span", "placement-plan-tier", tierName({ key: k, coverage: 0 })));
        tile.appendChild(el("span", "placement-plan-hours", "…"));
        tile.appendChild(el("span", "placement-plan-est", "estimating"));
        host.appendChild(tile);
      });
      if (note) note.textContent = "";
      return;
    }
    const keys = plan.options.map((o) => o.key);
    if (!keys.includes(selectedTier())) {
      tier = keys.includes(DEFAULT_TIER) ? DEFAULT_TIER : keys[keys.length - 1];
    }
    plan.options.forEach((opt) => {
      const on = opt.key === tier;
      const btn = el("button", "placement-plan-option");
      btn.type = "button";
      btn.dataset.tier = opt.key;
      btn.setAttribute("aria-pressed", String(on));
      btn.classList.toggle("is-selected", on);
      btn.appendChild(el("span", "placement-plan-tier", tierName(opt)));
      btn.appendChild(el("span", "placement-plan-hours", hm(opt.minutes)));
      /* The cap is the big number; the estimate is only worth saying when
         the test is expected to finish well inside it. */
      const probes = `~${Number(opt.est_probes) || 0} problems`;
      const early = Number(opt.est_minutes) < 0.9 * Number(opt.minutes);
      btn.appendChild(el("span", "placement-plan-est", early ? `${probes} · likely done in ${hm(opt.est_minutes)}` : probes));
      /* How much of a settle-everything test fits. A bar, because three
         percentages side by side read as a table; the words underneath say
         the one thing the bar cannot — that the top tier finishes the job. */
      const cov = Math.max(0, Math.min(1, Number(opt.coverage) || 0));
      const meter = el("span", "placement-plan-meter");
      const fill = el("i");
      fill.style.width = `${Math.round(cov * 100)}%`;
      meter.appendChild(fill);
      btn.appendChild(meter);
      btn.appendChild(el("span", "placement-plan-cov",
        cov >= 1 ? "enough to settle every concept" : `≈${Math.round(cov * 100)}% of a full calibration`));
      btn.addEventListener("click", () => {
        tier = opt.key;
        write(STORE_TIER, tier);
        paint();
      });
      host.appendChild(btn);
    });
    if (note) {
      const lo = Number(plan.per_problem_min_secs);
      const hi = Number(plan.per_problem_max_secs);
      const clock = lo > 0 && hi > 0
        ? (lo === hi ? `${mmss(hi)} per problem` : `${mmss(lo)}–${mmss(hi)} per problem, set per concept`)
        : "";
      const n = Number(plan.assessed_kcs) || 0;
      note.textContent = [n ? `${n} concepts to place` : "", clock].filter(Boolean).join(" · ");
    }
  };

  const paintNav = () => {
    const back = byId("placement-back-btn");
    const fwd = byId("placement-forward-btn");
    const startWrap = byId("placement-wizard-start");
    const list = steps();
    const last = stepIndex === list.length - 1;
    const step = currentStep();
    if (back) {
      // On the first step "Back" only means something for a retake: it
      // closes the questions and returns to the results.
      const canCancel = stepIndex === 0 && retakeOpen && !!status?.completed_at;
      back.textContent = canCancel ? "Cancel" : "← Back";
      back.classList.toggle("is-invisible", stepIndex === 0 && !canCancel);
    }
    if (fwd) {
      fwd.classList.toggle("hidden", last);
      fwd.disabled = step === "areas" && scope !== RAY && (!chosen || chosen.size === 0);
    }
    if (startWrap) {
      // The start button's own label and hidden-state belong to
      // diagnostic-page.js::renderStartButton; this wrapper only says whether
      // this is the step it belongs on, and waits for the lengths to load so
      // a click cannot send a budget nobody saw.
      const ready = scope === RAY || !!planFor(areasKey());
      startWrap.classList.toggle("hidden", !last || !ready);
    }
  };

  let focusNext = false;
  const paint = () => {
    const open = isOpen();
    byId("placement-plan")?.classList.toggle("hidden", !open);
    byId("placement-card")?.classList.toggle("is-planning", open);
    const retake = byId("placement-retake-btn");
    retake?.classList.toggle("hidden", !(status && !status.active && status.completed_at && !open));
    if (!open) return;
    if (!catalog) loadPlan();
    const step = currentStep();
    for (const name of ["areas", "level", "length"]) {
      byId(`placement-step-${name}`)?.classList.toggle("hidden", name !== step);
    }
    paintSteps();
    if (step === "areas") paintAreas();
    if (step === "length") paintLength();
    paintNav();
    if (focusNext) {
      focusNext = false;
      byId(`placement-step-${step}`)?.querySelector(".placement-step-q")?.focus({ preventScroll: true });
    }
  };

  const go = (i) => {
    stepIndex = i;
    focusNext = true;
    paint();
  };

  byId("placement-back-btn")?.addEventListener("click", () => {
    if (stepIndex === 0) {
      retakeOpen = false;
      paint();
      return;
    }
    go(stepIndex - 1);
  });
  byId("placement-forward-btn")?.addEventListener("click", () => {
    if (!byId("placement-forward-btn").disabled) go(stepIndex + 1);
  });
  byId("placement-focus-all")?.addEventListener("click", () => {
    if (!catalog) return;
    const every = scope !== RAY && catalog.every((a) => chosen.has(a.key));
    scope = "all";
    chosen = new Set(every ? [] : catalog.map((a) => a.key));
    saveAreas();
    paint();
  });
  byId("placement-retake-btn")?.addEventListener("click", () => {
    retakeOpen = true;
    go(0);
  });

  const render = (s) => {
    const wasActive = !!status?.active;
    status = s || null;
    // A run that started (or a fresh one after it) begins the questions from
    // the top next time; a finished run keeps them closed until "Retake".
    if (status?.active || (wasActive && !status?.active)) {
      retakeOpen = false;
      stepIndex = 0;
    }
    paint();
  };

  /* What the start button sends (api.js diagnosticStart). `minutes` is the
     chosen tier's budget from the plan the learner is looking at; `areas`
     null means the whole curriculum. */
  const startPayload = () => {
    if (scope === RAY) return { scope: RAY, minutes: null, areas: null };
    const plan = planFor(areasKey());
    const opt = plan?.options?.find((o) => o.key === selectedTier()) || null;
    return { scope: "all", minutes: opt ? Number(opt.minutes) : null, areas: areasParam() };
  };

  /* The onboarding course question (practice/course-pick.js) changed which
     concepts exist for this learner: drop the area catalogue, the plans cut
     from it and the saved focus, so the next status repaints from the new
     one with every area picked. */
  const coursesChanged = () => {
    catalogGen += 1;
    catalog = null;
    chosen = null;
    plans.clear();
    stepIndex = 0;
    if (scope === RAY) scope = "all";
    write(STORE_AREAS, "null");
    if (status) paint();
  };

  return {
    render,
    startPayload,
    coursesChanged,
    areaName,
    selectedScope: () => scope,
    /* practice-target.js routes here to take the rapid check. On a finished
       placement the questions are closed behind "Retake", so open them. */
    setScope: (value) => {
      scope = value === RAY ? RAY : "all";
      if (status?.completed_at) retakeOpen = true;
      stepIndex = 0;
      paint();
    },
  };
})();
window.PlacementWizard = PlacementWizard;
