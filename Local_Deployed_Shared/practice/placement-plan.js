/* ================================================================
   PLACEMENT PLAN — how long the test may take, chosen by the learner.

   The placement is graph-wide now (app/diagnostic.py, 2026-09-07): one
   estimate per enabled concept, adaptive, and it ends on evidence OR on a
   clock the learner picked. Seth: "a multiple choice option for different
   amounts of time … a 1 hour placement test, and a 3 hour placement test",
   then "a 6 hour version that helps with adjusting the edges". Each problem
   gets 20:00 (placement-timer.js); the plan is the TOTAL, and it is a hard
   cap — the server will not serve past it.

   This module owns three things and nothing else:
     * the picker (#placement-plan): one button per plan length, each with the
       server's point estimate of how long that plan will REALLY take and how
       many problems it will hold. 20:00 is the cap per problem, not the
       expectation, so "3 hours" is usually ~35 problems in ~2 hours;
     * `selectedHours()`, which api.js reads when the start button fires — the
       button in advance-events.js knows nothing about this module;
     * the running readout on the placement card (#placement-length, the
       progress count and the status line), time-based: "1h 12m left", not
       "of at most 14".

   It never fetches status. api.js announces every status it receives as
   `delta-drills-diagnostic-status`, and diagnostic-page.js hands each render
   here too; both paths land in `render`.

   STATIC ANCHORS ONLY: #placement-plan must be in index.html. No anchor →
   no picker and a failing watch (watch_placement.py), never a runtime one.
   ================================================================ */
const PlacementPlan = (() => {
  const byId = (id) => document.getElementById(id);

  /* The picker's lengths. Mirrors PLAN_HOURS in app/diagnostic.py — the
     server refuses any other value and falls back to its default, so a plan
     offered here that the server does not know would silently become a
     1-hour test. watch_placement.py parses both. */
  const PLAN_HOURS = [1, 3, 6];
  const DEFAULT_HOURS = 3;
  const STORE_KEY = "delta_drills_placement_plan_hours";

  let selected = null;
  let options = null;      // from /diagnostic/plan, once
  let loading = null;

  const _api = () => (typeof PracticeAPI !== "undefined" ? PracticeAPI : window.PracticeAPI);

  const _readSaved = () => {
    try {
      const h = Number(localStorage.getItem(STORE_KEY));
      return PLAN_HOURS.includes(h) ? h : null;
    } catch (_) {
      return null;
    }
  };
  const _save = (h) => {
    try { localStorage.setItem(STORE_KEY, String(h)); } catch (_) {}
  };

  const selectedHours = () => {
    if (selected == null) selected = _readSaved() ?? DEFAULT_HOURS;
    return selected;
  };

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
  const hours = (h) => (h === 1 ? "1 hour" : `${h} hours`);

  /* ---- the picker ---------------------------------------------------- */

  const _loadOptions = async () => {
    if (options) return options;
    if (loading) return loading;
    loading = (async () => {
      const plan = await _api()?.diagnosticPlan?.();
      if (plan && !plan.unavailable && Array.isArray(plan.options)) {
        options = plan.options;
        options.assessed_kcs = plan.assessed_kcs;
        options.arena_linked_kcs = plan.arena_linked_kcs;
      }
      loading = null;
      return options;
    })();
    return loading;
  };

  const _paintPicker = (status) => {
    const host = byId("placement-plan");
    if (!host) return;
    // The picker is for a test that has not started: mid-run the plan is
    // fixed, and after one the retake button reopens it.
    const show = !!status && !status.active && !status.unavailable;
    host.classList.toggle("hidden", !show);
    if (!show) return;
    host.textContent = "";
    const head = el("div", "placement-plan-head");
    head.appendChild(el("span", "placement-plan-title", "How long do you have?"));
    const n = Number(options?.assessed_kcs);
    if (Number.isFinite(n) && n > 0) {
      head.appendChild(el("span", "placement-chip", `${n} concepts to place`));
    }
    host.appendChild(head);

    const row = el("div", "placement-plan-options");
    row.setAttribute("role", "group");
    row.setAttribute("aria-label", "Placement test length");
    const current = selectedHours();
    PLAN_HOURS.forEach((h) => {
      const opt = (options || []).find((o) => Number(o.hours) === h) || null;
      const btn = el("button", "placement-plan-option");
      btn.type = "button";
      // Toggle buttons, not a radio group: role=radio promises arrow-key
      // navigation this module does not implement.
      btn.setAttribute("aria-pressed", String(h === current));
      btn.classList.toggle("is-selected", h === current);
      btn.dataset.hours = String(h);
      btn.appendChild(el("span", "placement-plan-hours", hours(h)));
      btn.appendChild(el("span", "placement-plan-cap", "hard cap"));
      /* The point estimate. `est_minutes` is what the difficulty heuristic
         expects the plan to actually take; `est_probes` how many problems fit
         in it. Both are estimates of the learner's time, so they are said as
         "about". Without the server's numbers the button still works — it
         just cannot say how long. */
      if (opt) {
        const est = Number(opt.est_minutes) * 60;
        btn.appendChild(el("span", "placement-plan-est", `about ${hm(est)} · ~${Number(opt.est_probes) || 0} problems`));
      } else {
        btn.appendChild(el("span", "placement-plan-est", "estimating…"));
      }
      btn.addEventListener("click", () => {
        selected = h;
        _save(h);
        _paintPicker(status);
      });
      row.appendChild(btn);
    });
    host.appendChild(row);
    host.appendChild(el("p", "placement-plan-note",
      "20:00 per problem. The test stops early once every concept is settled, and never runs past the cap."));
  };

  /* ---- the running readout ------------------------------------------ */

  const _paintLength = (status) => {
    const host = byId("placement-length");
    if (!host) return;
    host.textContent = "";
    if (!status) {
      // Signed out / unreachable: no learner, so no plan and no run to quote.
      host.appendChild(el("span", "placement-chip", "20:00 per problem"));
      return;
    }
    const plan = status.plan;
    const chip = (text) => host.appendChild(el("span", "placement-chip", text));
    const per = Number(plan?.per_problem_secs) || 1200;
    chip(`${Math.round(per / 60)}:00 per problem`);
    if (!plan) {
      // Not started: the picker above says how long; these say the shape.
      chip("stops early when settled");
      return;
    }
    if (status.active) {
      chip(`${hm(plan.remaining_secs)} of ${hm(plan.budget_secs)} left`);
    } else if (status.completed_at) {
      chip(`took ${hm(plan.spent_secs)} of ${hours(Number(plan.hours))}`);
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
    if (status.active) return `In progress · ${done} answered · ${hm(plan.remaining_secs)} left`;
    if (status.completed_at) return `Complete · ${done} problems in ${hm(plan.spent_secs)}`;
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
    _paintPicker(status);
    _paintLength(status);
    _paintProgress(status);
    // The estimates arrive once; the picker repaints itself when they do.
    if (status && !status.active && !options) {
      _loadOptions().then(() => _paintPicker(lastStatus));
    }
  };

  window.addEventListener("delta-drills-diagnostic-status", (e) => render(e.detail));

  return {
    render,
    selectedHours,
    statusLine,
    progressLabel: () => progressLabel(lastStatus),
    PLAN_HOURS: PLAN_HOURS.slice(),
  };
})();
window.PlacementPlan = PlacementPlan;
