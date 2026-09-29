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

   A CHOSEN CONCEPT (Seth, 2026-09-28; practice/concept-choice.js) runs on
   this same Route with `kind: "choice"` and `POST /api/practice/concept-route`
   (backend/app/concept_choice.py): the concept alone, until the XP model
   says it is learned ("ready") or the learner is not ready for it
   ("not_ready"). Only the words differ.
   ================================================================ */

const ReadyRoute = (() => {
  class Route {
    /**
     * @param {object} cfg
     *   kc          – the exercise's concept
     *   title       – its display name
     *   exerciseIds – the exercise's own bank ids (original + variants)
     *   kind        – "ready" (the exercise dialog) or "choice" (a concept
     *                 picked on the Learner Home), which also picks the endpoint
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
      this.kind = cfg.kind === "choice" ? "choice" : "ready";
      this.readyAt = Number.isFinite(cfg.readyAt) ? cfg.readyAt : null;
      this.pending = null; // the step fetched ahead, not yet served
      this._inflight = null;
    }

    async _fetch() {
      if (typeof apiFetch !== "function") throw new Error("no backend");
      const path = this.kind === "choice" ? "/api/practice/concept-route" : "/api/practice/ready-route";
      const res = await apiFetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kc: this.kc, exercise_ids: this.exerciseIds, served: this.served, skip: this.skip,
        }),
      });
      if (!res.ok) throw new Error(`ready-route ${res.status}`);
      const step = await res.json();
      if (Number.isFinite(step.p_target)) this.pTarget = step.p_target;
      if (Number.isFinite(step.ready_at)) this.readyAt = step.ready_at;
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
      if (this.kind === "choice") return this._choiceNote(correct);
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

    /** A chosen concept's note: its knowledge as XP toward the concept's 80. */
    _choiceNote(correct) {
      if (this.done && this.reason === "ready") return `Learned ${this.title}.`;
      if (this.done && this.reason === "not_ready") return `Not ready for ${this.title} yet.`;
      const xp = this._xp();
      const at = xp === null ? "" : ` (${xp} / ${this._xpFull()} XP)`;
      return correct ? `Getting there${at}.` : `Another one${at}.`;
    }

    _xpFull() { return Math.round((this.readyAt || 0.8) * 100); }

    _xp() {
      return Number.isFinite(this.pTarget)
        ? Math.round(Math.min(this.pTarget, this.readyAt || 0.8) * 100) : null;
    }

    /** Over, for any reason — ready, out of drills, or the fuse. */
    over() { return this.done; }

    /** Over BECAUSE the model says ready. */
    solved() { return this.done && this.reason === "ready"; }

    outcome() {
      const n = this.served.length;
      const qs = `${n} question${n === 1 ? "" : "s"}`;
      if (this.kind === "choice") return this._choiceOutcome(qs, n);
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

    /* The Learner Home prints this after the concept's name
       (exercise-session.js::onEnd), so it does not repeat the name. */
    _choiceOutcome(qs, n) {
      const text = this.reason === "ready"
        ? `learned after ${qs}.`
        : this.reason === "not_ready"
          ? "not ready yet — practice other concepts first. Recorded answers are kept."
          : this.reason === "exhausted"
            ? `out of drills for now after ${qs}. Recorded answers are kept.`
            : `stopped after ${qs}. Recorded answers are kept.`;
      return {
        mode: "choice", ready: this.reason === "ready", solved: this.reason === "ready",
        notReady: this.reason === "not_ready", reason: this.reason, attempts: this.attempts,
        used: n, pTarget: this.pTarget, xp: this._xp(), text,
      };
    }

    serialize() {
      return {
        kind: this.kind, readyAt: this.readyAt,
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
