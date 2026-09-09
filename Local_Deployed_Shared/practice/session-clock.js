/* ================================================================
   THE PER-QUESTION ALLOWANCE — the PROBLEM's own, read off the question

   🔴 THIS REVERSES 2026-08-28 (which reversed 2026-08-23). Seth,
   2026-09-09: "for practice make it such that for the problems, they are
   keyed to how much time you need for each of them rather than what you
   select at the beginning. it should be like the diagnostic for the timing
   of the questions ... using the same time." So the picker on the idle
   screen is gone, and the number is the one the placement test already
   uses: the concept's cap in lessons/placement_time_caps.json (5:00 for a
   one-call Python drill, ARENA's 10:00 for einops, 20:00 by default), sent
   by the server on every practice question as `secs_allowed`
   (backend question_pick.secs_allowed_for → NextQuestionResponse). This
   file only READS it. Nothing here is chosen, stored or remembered per
   learner — there is no store, no presets and no `set`.

   What the 08-28 picker was for — a first-encounter lesson is read on the
   answer clock — is carried by the table now: a concept's minutes are what
   its drills need, and the default is 20:00, not the 02:00 the picker
   defaulted to.

   🔴 `null` STILL MEANS NO LIMIT downstream (timer.js runs no interval on
   it), so this file never answers null: a question the server did not
   stamp — an older backend, a resume rebuilt from the static bank with no
   snapshot copy — gets the CEILING, not "no limit". "No limit" survives
   only where a block brings its own numbers: an ARENA exercise session's
   setup (practice/exercise-session.js), which timer.js reads before it
   asks here.

   Both steps get the same number — answering and reviewing alike, per
   question — exactly as the picker's one number did.

   🔴 THE PLACEMENT IS NOT ON THIS PATH. A probe carries
   `diagnostic_secs_allowed` and practice/placement-timer.js displays it;
   timer.js's `_answerSecsFor` asks PlacementTimer first. Same table, two
   readers, because the placement also CHARGES the time server-side.
   ================================================================ */

(function initSessionClock() {
  /* The server's own ceiling (diagnostic.PER_PROBLEM_SECS; the placement's
     PLACEMENT_ANSWER_SECS is the same number): what a question gets when the
     field is missing. Missing means "no opinion", never "no limit". */
  const CEILING_SECS = 1200;

  /* `PracticeAPI` is a script-global const, not `window.PracticeAPI`
     (practice/api.js) — same read timer.js uses. */
  const _question = () => {
    const api = typeof PracticeAPI !== "undefined" ? PracticeAPI : window.PracticeAPI;
    return api?.currentQuestion || null;
  };

  /* 🔴 THE TABLE ITSELF, FOR A QUESTION THE SERVER NEVER SAW. Only the
     backend queue stamps `secs_allowed`; a question built out of the static
     bank carries none, and there are real paths that do exactly that —
     `KcPractice` (the Knowledge Graph's Practice ⤢ ladder and the `?lesson=`
     loop) hydrates every rung through `buildPracticeQuestionFromBank`, the
     Pyodide engine builds its own, and a guest has no backend at all. Left at
     the ceiling those are a FLAT 20:00, which is the same defect as the flat
     02:00 this change removed, one number further out.

     So the concept's cap is read from the same file the server reads, and the
     question's `ladder_kc` is the key — `KcPractice._hydrate` stamps that
     field for its own reasons, so the concept is on the record even when the
     clock is not.

     🔴 `d.kcs` IS THE PARSE CHECK, not `res.ok`. A JSON file dropped from a
     deploy comes back through the SPA rewrite as 200 text/html, so `ok` is
     true and `.json()` throws into the catch — see the `.vercelignore` trap in
     the deploy notes. Until it lands (or if it never does) `_capFor` answers
     null and the ceiling stands; the fetch is started at parse time and a
     question is always at least one round trip away. */
  let _caps = null;
  let _qmatrix = null;

  const _load = (path, cache, keep) =>
    Promise.resolve()
      .then(() => fetch(path, { cache }))
      .then((res) => (res.ok ? res.json() : null))
      .then((d) => keep(d))
      .catch(() => {});

  /* 🔴 ONE PROMISE, AWAITED BEFORE THE FIRST QUESTION IS SERVED. Without it
     the two tables land whenever they land, and `secsFor` answers null in the
     meantime: `ANSWER_SECS()` at onQuestionRendered could hand back the 20:00
     ceiling while `REVIEW_SECS()`, a minute later, handed back the concept's
     real 6:00 — the SAME question on two different allowances, and a pause
     snapshot stamped with whichever one won. timer.js's `start()`/`resume()`
     wait on this, so by the time anything is on screen the answer is settled
     (or has permanently failed, in which case the ceiling stands everywhere).
     Codex, 2026-09-09. */
  const ready = Promise.all([
    // The cap table is small and is the number on screen — always fresh.
    _load("lessons/placement_time_caps.json", "no-cache", (d) => {
      if (d && d.kcs && typeof d.kcs === "object") _caps = d;
    }),
    // The q-matrix is 160 KB and questions.js already fetches it `force-cache`
    // for its own parking filter — same request, so this costs nothing.
    _load("lessons/qmatrix_tags.json", "force-cache", (d) => {
      if (d && typeof d === "object") _qmatrix = d;
    }),
  ]).then(() => undefined);

  /* Whole positive seconds only, and validated AFTER the rounding: a raw 0.4
     is > 0 and would have survived the check, then rounded to 0 — a question
     that expires the instant it is rendered. Clamped to the ceiling the way
     the server clamps the table. */
  const _clean = (raw) => {
    if (typeof raw !== "number" || !Number.isFinite(raw) || raw <= 0) return null;
    const secs = Math.min(CEILING_SECS, Math.round(raw));
    return secs >= 1 ? secs : null;
  };

  const _capFor = (kc) => {
    if (!_caps || typeof kc !== "string" || !kc) return null;
    const raw = Object.prototype.hasOwnProperty.call(_caps.kcs, kc)
      ? _caps.kcs[kc]
      : _caps.default_secs;
    return _clean(raw);
  };

  /* 🔴 THE SAME RULE THE SERVER RUNS, on the same two files.
     `question_pick.secs_allowed_for` is:

         kcs  = [ladder_kc] if ladder_kc else question_kcs(question_id)
         caps = [kc_cap_secs(kc) for kc in kcs if kc]
         return max(caps) if caps else kc_cap_secs(None)

     — `question_kcs` being the q-matrix row's `target_kcs` (kc_graph.py). A
     record with no `ladder_kc` is not rare: the Pyodide engine builds its own,
     a guest has no backend, and a plain bank record carries only its id. Those
     were all landing on the flat ceiling, which is the 02:00 defect this
     change removed, one number further out. The q-matrix is what the id is
     worth, so read it — questions.js already fetches this exact file
     `force-cache`, so this is a cache hit, and it discards the mapping rather
     than exporting it. */
  const _qmatrixCap = (questionId) => {
    if (!_qmatrix || !Number.isFinite(questionId)) return null;
    const row = _qmatrix[String(questionId)];
    const kcs = Array.isArray(row?.target_kcs) ? row.target_kcs : [];
    const caps = kcs.map(_capFor).filter((n) => typeof n === "number");
    return caps.length ? Math.max(...caps) : null;
  };

  /* The clock a question record carries, or null when nothing can name one.
     A string, 0, a negative or a boolean is a record nobody stamped, and the
     table (then the ceiling) is the honest fallback. */
  const secsFor = (question) => {
    const own = _clean(question?.secs_allowed);
    if (own !== null) return own;
    const ladder = _capFor(question?.ladder_kc);
    if (ladder !== null) return ladder;
    return _qmatrixCap(Number(question?.question_id));
  };

  /* Whether the question on screen carries its own number. The idle notch
     asks this: between blocks nothing is on screen, so there is no "next
     question's allowance" to show — it is set by the concept the queue
     picks, and that is not known until it is served. */
  const hasOwn = () => secsFor(_question()) !== null;
  const answerSecs = () => secsFor(_question()) ?? CEILING_SECS;
  const reviewSecs = () => answerSecs();
  const isUnlimited = () => false;

  window.SessionClock = { CEILING_SECS, ready, secsFor, hasOwn, answerSecs, reviewSecs, isUnlimited };
})();
