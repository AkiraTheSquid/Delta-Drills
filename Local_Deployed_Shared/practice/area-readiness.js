/* ================================================================
   AREA-READINESS.JS — the one theta→readiness map for an area row.

   What is left of practice/placement-results.js after the placement test
   was retired (Seth, 2026-09-26). The group board (groups/groups_lane.js)
   still draws each member's areas from app/diagnostic.py
   `display_area_estimates` — the frozen readout of a stored placement, or
   the prior for a learner who never took one — in the
   `{topic, theta, sd, probes}` shape, and scores them here. Nothing else
   may grow a second copy of this map (groups/watch.py).

   `theta` is 100 × P(known); readiness is that P clipped to [0.02, 0.92],
   the range a placement could ever seed (the retired
   `diagnostic._mastery_from_p`): a placement could unlock, never certify.
   `sd` is on the same 0-100 scale, capped at 50 because a band wider than
   that is "we do not know" and a ±78 reads as precision about the
   uncertainty. An area with `probes === 0` is a prior wearing a percentage;
   the row that draws it says "not probed" and dims it.
   ================================================================ */
const AreaReadiness = (() => {
  const FLOOR = 0.02;
  const CAP = 0.92;

  const readiness = (theta) => {
    const m = Number(theta) / 100;
    if (!Number.isFinite(m)) return FLOOR;
    return Math.max(FLOOR, Math.min(CAP, m));
  };

  const band = (sd) => {
    const b = Math.round(Number(sd));
    return Number.isFinite(b) ? Math.max(1, Math.min(50, b)) : null;
  };

  return { readiness, band };
})();

window.AreaReadiness = AreaReadiness;
