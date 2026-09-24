/* ================================================================
   PLACEMENT PLAN — how long the test may take, chosen by the learner.

   The placement is graph-wide now (app/diagnostic.py, 2026-09-07): one
   estimate per enabled concept, adaptive, and it ends on evidence OR on a
   clock the learner picked. Seth: "a multiple choice option for different
   amounts of time … a 1 hour placement test, and a 3 hour placement test",
   then "a 6 hour version that helps with adjusting the edges". Each problem
   gets 20:00 (placement-timer.js); the plan is the TOTAL, and it is a hard
   cap — the server will not serve past it.

   This module owns the RUNNING READOUT on the placement card
   (#placement-length, the progress count and the status line), time-based:
   "1h 12m left", not "of at most 14".

   The set-up before a run — areas, experience, length — moved to
   practice/placement-wizard.js on 2026-09-24 (Seth: sequential questions
   with Back / Next, and lengths cut from the areas picked). `render` hands
   every status to it; `selectedScope` / `setScope` forward to it so callers
   (api.js, practice-target.js) keep one name to ask.

   It never fetches status. api.js announces every status it receives as
   `delta-drills-diagnostic-status`, and diagnostic-page.js hands each render
   here too; both paths land in `render`.

   STATIC ANCHORS ONLY: #placement-plan must be in index.html. No anchor →
   no set-up and a failing watch (watch_placement.py), never a runtime one.
   ================================================================ */
const PlacementPlan = (() => {
  const byId = (id) => document.getElementById(id);

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const hm = (secs) => {
    const s = Math.max(0, Math.round(Number(secs) || 0));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    if (h > 0) return m > 0 ? `${h}h ${m}m` : `${h}h`;
    return m > 0 ? `${m}m` : s > 0 ? "<1m" : "0m";
  };

  /* The clock is PER CONCEPT (lessons/placement_time_caps.json, enforced by
     app/diagnostic.py): quick one-call drills get a few minutes, einops and
     broadcasting get ARENA's own ten. The status carries the range over the
     concepts THIS learner is assessed on (disabled ones excluded). With no
     status, or an old backend that sends no range, quote the flat ceiling. */
  // m:ss, never rounded to the minute — a 330 s cap is 5:30, not 6:00.
  const mmss = (secs) => {
    const n = Math.max(0, Math.round(Number(secs) || 0));
    return `${Math.floor(n / 60)}:${String(n % 60).padStart(2, "0")}`;
  };
  const perProblemText = (status) => {
    const plan = status?.plan;
    // Top level first: before a run exists `plan` is null and the picker is
    // exactly where the range matters. Inside `plan` for older statuses.
    const lo = Number(status?.per_problem_min_secs ?? plan?.per_problem_min_secs);
    const hi = Number(status?.per_problem_max_secs ?? plan?.per_problem_max_secs);
    if (Number.isFinite(lo) && Number.isFinite(hi) && lo > 0 && hi > 0) {
      return lo === hi ? `${mmss(hi)} per problem` : `${mmss(lo)}–${mmss(hi)} per problem, set per concept`;
    }
    const per = Number(plan?.per_problem_secs) || 1200;
    return `${mmss(per)} per problem`;
  };

  /* ---- the running readout ------------------------------------------ */

  const _paintLength = (status) => {
    const host = byId("placement-length");
    if (!host) return;
    host.textContent = "";
    if (!status) {
      // Signed out / unreachable: no learner, so no plan and no run to quote.
      host.appendChild(el("span", "placement-chip", "up to 20:00 per problem"));
      return;
    }
    const plan = status.plan;
    const chip = (text) => host.appendChild(el("span", "placement-chip", text));
    chip(perProblemText(status));
    if (!plan) {
      // Not started: the picker above says how long; these say the shape.
      chip("stops early when settled");
      return;
    }
    if (status.active) {
      chip(`${hm(plan.remaining_secs)} of ${hm(plan.budget_secs)} left`);
    } else if (status.completed_at) {
      chip(`took ${hm(plan.spent_secs)} of ${hm(plan.budget_secs)}`);
    } else {
      chip("stops early when settled");
    }
  };

  const _paintProgress = (status) => {
    const plan = status?.plan;
    if (!plan || !status.active) return;
    const budget = Math.max(1, Number(plan.budget_secs) || 3600);
    const spent = Math.min(budget, Math.max(0, Number(plan.spent_secs) || 0));
    const fill = byId("placement-progress-fill");
    if (fill) fill.style.width = `${(spent / budget) * 100}%`;
    const tick = byId("placement-progress-tick");
    tick?.classList.add("hidden");
    const count = byId("placement-progress-count");
    const done = Number(status.probes_done) || 0;
    if (count) count.textContent = `${done} answered · ${hm(plan.remaining_secs)} left`;
    const host = byId("placement-progress");
    host?.setAttribute("aria-valuenow", String(spent));
    host?.setAttribute("aria-valuemax", String(budget));
  };

  const statusLine = (status) => {
    const plan = status?.plan;
    if (!plan) return null;
    const done = Number(status.probes_done) || 0;
    const focus = Array.isArray(status.focus_areas) && status.focus_areas.length
      ? `${status.focus_areas.map((a) => window.PlacementWizard?.areaName?.(a) || a).join(", ")} · `
      : "";
    // The rapid check names itself first; a focus reads after the state.
    const prefix = status.scope === "raytracing-0.1" ? "Ray Tracing 0.1 · " : "";
    const what = status.scope === "raytracing-0.1" ? "" : focus;
    if (status.active) return `${prefix}In progress · ${what}${done} answered · ${hm(plan.remaining_secs)} left`;
    if (status.completed_at) return `${prefix}Complete · ${what}${done} problem${done === 1 ? "" : "s"} in ${hm(plan.spent_secs)}`;
    return "Not started";
  };

  const progressLabel = (status) => {
    const plan = status?.plan;
    if (!plan) return "";
    const done = Number(status.probes_done) || 0;
    return `Placement problem ${done + 1} · ${hm(plan.remaining_secs)} left`;
  };

  let lastStatus = null;
  const render = (status) => {
    lastStatus = status || null;
    window.PlacementWizard?.render(status);
    _paintLength(status);
    _paintProgress(status);
  };

  window.addEventListener("delta-drills-diagnostic-status", (e) => render(e.detail));

  return {
    render,
    selectedScope: () => window.PlacementWizard?.selectedScope?.() || "all",
    setScope: (value) => window.PlacementWizard?.setScope?.(value),
    statusLine,
    progressLabel: () => progressLabel(lastStatus),
  };
})();
window.PlacementPlan = PlacementPlan;
