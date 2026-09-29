/* ================================================================
   PRACTICE ENTRY — start after modules load
   ================================================================ */

if (typeof initPractice === "function") {
  // Kept so a flow that must not race the boot fetch can wait for it — the
  // ?lesson=<kc> drill-only start (practice/lessons.js). Boot handles its own
  // errors; this promise only says "boot is over".
  window.__practiceBoot = initPractice().catch(() => {});
}
