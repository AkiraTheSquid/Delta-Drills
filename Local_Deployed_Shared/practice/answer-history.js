/* ================================================================
   ANSWER HISTORY — how long THIS learner actually takes per problem

   Seth, 2026-09-09, on the drill slider: "it should increment based upon the
   amount of time averaged per problem that you do based upon your history of
   solving problems such that it knows about how long it takes you per problem
   … it should automatically display above it the expected amount of time to
   the left and the maximum amount of time to the right."

   The maximum was always computable — it is the answer cap plus the review cap
   times the count, arithmetic on numbers the learner chose. The EXPECTED time
   was not: nothing recorded how long a practice question actually took. The
   placement clock reports `elapsed_secs` on its probes (practice/api.js) and
   the ARENA unlock stopwatch times an exercise, but an ordinary drill answered
   on the Practice tab left no trace of its duration anywhere.

   This file is that trace, and it is deliberately the smallest one that
   answers the question:

     • ONE number per answered problem — the seconds between the question
       rendering and the submit that ended its answer phase.
     • LOCAL. It never leaves the browser. It is used to describe a block the
       learner is about to start, not to grade, score or place them, and
       sending it would make it a measurement of the person rather than a
       description of the block.
     • Per account, under the same storage key everything else on this tab is
       namespaced by, so switching test users does not blend two people's
       pace.

   🔴 SUBMITS ONLY. `abandon()` exists because a skip, a forced advance, a
   pause and an end-of-session are all ways for the answer phase to stop
   without the learner having answered anything, and folding those in would
   quietly describe the block with the time it takes to press Skip. The
   estimate is about problems they SOLVED, or at least tried to.
   ================================================================ */

const AnswerHistory = (() => {
  "use strict";

  /* Enough samples to be a description of a person rather than of one
     unusually good or bad morning. Below it the caller is told there is no
     estimate and falls back to the answer cap, which is honest: the app does
     not know yet. */
  const MIN_SAMPLES = 3;
  const KEEP = 60;
  /* A sample outside this band is not a pace, it is a story: a tab left open
     over lunch on one side, a mis-click through a question on the other.
     Clamped rather than dropped so a slow-but-real problem still counts. */
  const MIN_SECS = 5;
  const MAX_SECS = 45 * 60;

  let startedAt = null;
  let elapsed = 0;
  let owner = null;

  const _key = () => {
    try {
      return `${getPracticeStorageKey()}_answer_secs`;
    } catch (_) {
      return "practice_progress_guest_answer_secs";
    }
  };

  const _read = () => {
    try {
      const raw = JSON.parse(localStorage.getItem(_key()) || "[]");
      return Array.isArray(raw) ? raw.filter((n) => Number.isFinite(n) && n > 0) : [];
    } catch (_) {
      return [];
    }
  };

  const _write = (list) => {
    try {
      localStorage.setItem(_key(), JSON.stringify(list.slice(-KEEP)));
    } catch (_) { /* private mode — the estimate degrades, nothing breaks */ }
  };

  /** The answer phase for a question started. Idempotent: re-arming the clock
      after a failed submit (timer.js::resumeAnswerPhase) must not restart the
      measurement of a question that has already been half-answered. */
  const begin = () => {
    if (owner !== _key()) { elapsed = 0; startedAt = null; }
    owner = _key();
    if (startedAt === null) startedAt = Date.now();
  };

  const pause = () => {
    if (startedAt !== null) elapsed += Math.max(0, Date.now() - startedAt);
    startedAt = null;
  };

  /** The learner submitted. This is the one path that records. */
  const settle = () => {
    pause();
    const secs = Math.round(elapsed / 1000);
    elapsed = 0;
    if (owner !== _key()) return;
    if (!Number.isFinite(secs) || secs < MIN_SECS) return;
    _write(_read().concat(Math.min(MAX_SECS, secs)));
  };

  /** The answer phase ended without an answer. Drop the measurement. */
  const abandon = () => {
    startedAt = null;
    elapsed = 0;
    owner = null;
  };

  /** Samples recorded so far — the caller's own "do I trust this" check. */
  const samples = () => _read().length;

  /* MEDIAN, not mean. One question spent reading the docs in another tab is
     worth ten minutes and would drag a mean estimate past anything the learner
     recognises as their pace; the median simply ignores it. */
  const secondsPerProblem = () => {
    const list = _read();
    if (list.length < MIN_SAMPLES) return null;
    const sorted = list.slice().sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    return sorted.length % 2
      ? sorted[mid]
      : Math.round((sorted[mid - 1] + sorted[mid]) / 2);
  };

  /** Wipe — offered so a learner whose pace changed (or whose account was
      shared) is not described by someone else's history forever. */
  const reset = () => {
    abandon();
    _write([]);
  };

  return { begin, pause, settle, abandon, samples, secondsPerProblem, reset, MIN_SAMPLES };
})();

window.AnswerHistory = AnswerHistory;
