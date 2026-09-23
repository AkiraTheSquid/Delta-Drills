/* ================================================================
   PRACTICE UNTIL READY — the client half of the exercise dialog's route

   Seth, 2026-09-23: "starting with the concept, and practicing that, and then
   if you are not ready for it, it essentially routes backward to try to probe
   what the prerequisite weakness is ... and for whichever weakness you have it
   essentially does that for the more greedy portion." It REPLACES the
   dialog's question count: the block ends when the model says the learner is
   ready for the exercise, not after N questions.

   Every decision is the backend's (`POST /api/practice/ready-route`,
   backend/app/ready_route.py): one posterior over the exercise's concept and
   everything under it, the same model the math explorer runs. This file only
   remembers which questions this session has already asked (so none is asked
   twice), asks for the next one, and says in a few words what just happened.

   🔴 ORDER ONLY. Grading and every recorded answer go through the ordinary
   submit path; the route reads them back from there.
   ================================================================ */

const ReadyRoute = (() => {
  class Route {
    /**
     * @param {object} cfg
     *   kc          – the exercise's concept
     *   title       – its display name
     *   exerciseIds – the exercise's own bank ids (original + variants)
     */
    constructor(cfg) {
      this.kc = String(cfg.kc);
      this.title = cfg.title || this.kc;
      this.exerciseIds = (cfg.exerciseIds || []).filter(Number.isFinite);
      this.served = Array.isArray(cfg.served) ? cfg.served.filter(Number.isFinite) : [];
      this.skip = Array.isArray(cfg.skip) ? cfg.skip.filter(Number.isFinite) : [];
      this.done = !!cfg.done;
      this.reason = cfg.reason || null;
      this.pTarget = Number.isFinite(cfg.pTarget) ? cfg.pTarget : null;
      this.attempts = Number(cfg.attempts) || 0;
      this.pending = null; // the step fetched ahead, not yet served
      this._inflight = null;
    }

    async _fetch() {
      if (typeof apiFetch !== "function") throw new Error("no backend");
      const res = await apiFetch("/api/practice/ready-route", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kc: this.kc, exercise_ids: this.exerciseIds, served: this.served, skip: this.skip,
        }),
      });
      if (!res.ok) throw new Error(`ready-route ${res.status}`);
      const step = await res.json();
      if (Number.isFinite(step.p_target)) this.pTarget = step.p_target;
      if (step.done) {
        this.done = true;
        this.reason = step.reason || "ready";
        return null;
      }
      return step;
    }

    /** The next step, fetched once. Concurrent callers share one request. */
    peek() {
      if (this.done) return Promise.resolve(null);
      if (this.pending) return Promise.resolve(this.pending);
      if (!this._inflight) {
        this._inflight = this._fetch()
          .then((step) => { this.pending = step; return step; })
          .finally(() => { this._inflight = null; });
      }
      return this._inflight;
    }

    /** Resolves once any look-ahead in flight has landed (errors swallowed). */
    settle() {
      return this._inflight ? this._inflight.catch(() => null) : Promise.resolve(null);
    }

    commit(step) {
      this.served.push(step.question_id);
      if (step.attempt) this.attempts += 1;
      this.pending = null;
    }

    drop(step) {
      this.skip.push(step.question_id);
      this.pending = null;
    }

    /** A graded result. Looks ahead at once — the answer is already on the
        server — so the session knows it is over before it asks to advance. */
    async observe(step, correct) {
      if (!step) return "";
      await this.peek().catch(() => null);
      const pct = Number.isFinite(this.pTarget) ? ` (${Math.round(this.pTarget * 100)}% ready)` : "";
      if (this.done && this.reason === "ready") return `Ready for ${this.title}.`;
      const title = step.kc_title || step.kc;
      if (step.mode === "attempt") {
        return correct ? `Solved${pct} — one more to confirm.` : `Miss — checking what it rests on${pct}.`;
      }
      if (step.mode === "probe") {
        return correct ? `${title} holds.` : `Weak spot: ${title} — drilling it.`;
      }
      return correct ? `${title}: getting there${pct}.` : `${title}: another one.`;
    }

    /** Over, for any reason — ready, out of drills, or the fuse. */
    over() { return this.done; }

    /** Over BECAUSE the model says ready. */
    solved() { return this.done && this.reason === "ready"; }

    outcome() {
      const n = this.served.length;
      const qs = `${n} question${n === 1 ? "" : "s"}`;
      const text = this.reason === "ready"
        ? `Ready for ${this.title} after ${qs}. Recorded answers are kept.`
        : this.reason === "exhausted"
          ? `Out of drills on ${this.title} and what it rests on after ${qs} — not ready yet. Recorded answers are kept.`
          : this.reason === "fuse"
            ? `Stopped after ${qs} without reaching ready. Recorded answers are kept.`
            : `${qs} on ${this.title}. Recorded answers are kept.`;
      return {
        mode: "ready", ready: this.reason === "ready", solved: this.reason === "ready",
        reason: this.reason, attempts: this.attempts, used: n, pTarget: this.pTarget, text,
      };
    }

    serialize() {
      return {
        kc: this.kc, title: this.title, exerciseIds: this.exerciseIds.slice(),
        served: this.served.slice(), skip: this.skip.slice(),
        done: this.done, reason: this.reason, pTarget: this.pTarget, attempts: this.attempts,
      };
    }

    static restore(saved) {
      if (!saved || !saved.kc) return null;
      return new Route(saved);
    }
  }

  return { Route };
})();

window.ReadyRoute = ReadyRoute;
