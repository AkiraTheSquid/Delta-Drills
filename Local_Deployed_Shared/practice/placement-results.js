/* ================================================================
   PLACEMENT RESULTS — what the test actually measured.

   The results card used to say "Placement results and ARENA curriculum
   progress will appear here." It said that to someone who had ALREADY
   finished the test, which made the whole surface look broken — and it was
   a lie by omission, because /api/practice/diagnostic/status has returned
   the real numbers the entire time:

       areas: [{ topic, theta, sd, probes }]   theta on the 0-100 difficulty scale
       atoms_seeded: <int>                     concepts seeded at finish()
       probes_done / budget / min_probes
       completed_at, self_reported_level

   Nothing rendered them. This file does, and it is the ONLY writer of the
   card's body — diagnostic-page.js decides WHETHER the card shows, this
   decides what is in it.

   THE HONESTY RULE THIS FILE EXISTS TO KEEP
     A placement built from two probes knows almost nothing about seven of
     its nine areas, and the backend still returns a theta for every one of
     them (the untouched ones are the prior, propagated). Printing those as
     bare percentages would invent confidence the test never earned. So an
     area with `probes === 0` is labelled "not probed" and dimmed, every
     area carries its own ± band from `sd`, and the summary says how many
     of the areas were actually probed. A number the learner can't trust is
     worse than no number, because they will plan around it.
   ================================================================ */

const PlacementResults = (() => {
  const byId = (id) => document.getElementById(id);

  /* Mirrors of app/diagnostic.py. finish() seeds per-atom BKT mastery through
     exactly this map (`_mastery_from_theta`), so these four numbers are the
     difference between reporting the learner's placement and reporting a
     figure nothing else in the system agrees with. practice/watch.py parses
     BOTH files and fails if they ever drift apart. */
  const DIFF_FLOOR = 20.0;         // diagnostic.py _DIFF_FLOOR
  const DIFF_SPAN = 80.0;          // diagnostic.py _DIFF_SPAN
  const SEED_MASTERY_FLOOR = 0.02; // diagnostic.py SEED_MASTERY_FLOOR
  const SEED_MASTERY_CAP = 0.92;   // diagnostic.py SEED_MASTERY_CAP

  const readiness = (theta) => {
    const m = (Number(theta) - DIFF_FLOOR) / DIFF_SPAN;
    if (!Number.isFinite(m)) return SEED_MASTERY_FLOOR;
    return Math.max(SEED_MASTERY_FLOOR, Math.min(SEED_MASTERY_CAP, m));
  };
  const pct = (r) => Math.round(r * 100);

  /* `sd` is on the same 0-100 scale as theta, so it converts to readiness
     points by the same span. Capped at 50 because a band wider than that is
     "we do not know" and a ±78 reads as precision about the uncertainty. */
  const band = (sd) => {
    const b = Math.round((Number(sd) / DIFF_SPAN) * 100);
    return Number.isFinite(b) ? Math.max(1, Math.min(50, b)) : null;
  };

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const completedOn = (iso) => {
    if (!iso) return null;
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return null;
    return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
  };

  const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

  /* ---- the three blocks -------------------------------------------- */

  const renderMeta = (status, areas, probedCount) => {
    const host = byId("placement-results-meta");
    if (!host) return;
    const bits = [];
    const on = completedOn(status.completed_at);
    if (on) bits.push(`Completed ${on}`);
    bits.push(plural(Number(status.probes_done) || 0, "question answered", "questions answered"));
    if (areas.length) bits.push(`${probedCount} of ${areas.length} areas probed`);
    if (Number.isFinite(Number(status.atoms_seeded))) {
      bits.push(plural(Number(status.atoms_seeded), "concept seeded", "concepts seeded"));
    }
    host.textContent = bits.join(" · ");
  };

  const renderOverall = (status, areas, probedCount) => {
    const host = byId("placement-readiness");
    if (!host) return;
    host.textContent = "";
    if (!areas.length) return;

    const mean = areas.reduce((sum, a) => sum + readiness(a.theta), 0) / areas.length;
    const meanTheta = areas.reduce((sum, a) => sum + Number(a.theta), 0) / areas.length;

    const figure = el("div", "placement-overall");
    const dial = el("div", "placement-overall-figure");
    dial.appendChild(el("strong", null, `${pct(mean)}%`));
    dial.appendChild(el("span", null, "ready for ARENA"));
    figure.appendChild(dial);

    const say = el("div", "placement-overall-say");
    say.appendChild(
      el(
        "p",
        null,
        `Averaged across the ${areas.length} areas the test covers. Practice now starts you around ` +
          `difficulty ${Math.round(meanTheta)} on the 0-100 scale, and everything below that is unlocked.`,
      ),
    );
    /* The caveat is not boilerplate — it is the difference between a number
       the learner can act on and one they will over-trust. It states exactly
       which part of the figure is evidence and which part is still a guess. */
    const unprobed = areas.length - probedCount;
    if (unprobed > 0) {
      say.appendChild(
        el(
          "p",
          "placement-overall-caveat",
          `${probedCount === 0 ? "No area" : plural(probedCount, "area was", "areas were")} probed directly; ` +
            `the other ${unprobed} ${unprobed === 1 ? "is" : "are"} carried from your answers and your stated ` +
            `starting point. Those move fastest once you practise them.`,
        ),
      );
    }
    figure.appendChild(say);
    host.appendChild(figure);
  };

  const renderAreas = (areas) => {
    const host = byId("placement-areas");
    if (!host) return;
    host.textContent = "";
    if (!areas.length) return;

    /* Deliberately NOT "this is the order practice will work through".
       prioritization.py is weakest-first, but over SUBTOPICS, weighted, and
       scaled by how reachable the easiest available question is — so the real
       order will not match this list row for row. Promising that it would is
       a claim the app then breaks in front of the learner. */
    host.appendChild(
      el(
        "div",
        "placement-areas-head",
        "Weakest first. Practice picks weakest-first too, but subtopic by subtopic inside these areas.",
      ),
    );

    // Weakest first: the card answers "what do I still need", so the thing
    // the learner needs most has to be the first row, not an alphabetical
    // accident. Ties break toward the area with more evidence behind it.
    const rows = areas
      .slice()
      .sort((a, b) => readiness(a.theta) - readiness(b.theta) || (b.probes || 0) - (a.probes || 0));

    rows.forEach((area) => {
      const probes = Number(area.probes) || 0;
      const r = readiness(area.theta);
      const row = el("div", `placement-area${probes ? "" : " placement-area--unprobed"}`);
      row.appendChild(el("span", "placement-area-name", String(area.topic || "Other")));

      const bar = el("span", "placement-area-bar");
      const fill = el("i");
      fill.style.width = `${pct(r)}%`;
      bar.appendChild(fill);
      row.appendChild(bar);

      row.appendChild(el("span", "placement-area-pct", `${pct(r)}%`));
      const b = band(area.sd);
      row.appendChild(el("span", "placement-area-conf", b === null ? "" : `±${b}`));
      row.appendChild(
        el("span", "placement-area-probes", probes ? plural(probes, "probe", "probes") : "not probed"),
      );
      host.appendChild(row);
    });
  };

  /* ---- entry point -------------------------------------------------- */

  const render = (status) => {
    const areas = Array.isArray(status?.areas) ? status.areas : [];
    const probedCount = areas.filter((a) => (Number(a.probes) || 0) > 0).length;
    renderMeta(status || {}, areas, probedCount);
    renderOverall(status || {}, areas, probedCount);
    renderAreas(areas);

    const empty = byId("placement-results-empty");
    // A finished placement with no areas at all means the bank changed under
    // the account. Say so rather than render three empty containers.
    empty?.classList.toggle("hidden", areas.length > 0);
  };

  return { render, readiness, band };
})();
window.PlacementResults = PlacementResults;
