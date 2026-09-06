/* ================================================================
   SINGLE-KC PRACTICE LADDER — the ?lesson=<kc> flow

   Reached from the Knowledge Graph's "Practice ⤢" overlay. Replaces the
   old preview dead-end ("Demo lesson complete") with the real expertise-
   reversal ladder that lessons_structured.json has always carried but
   nothing ever served:

     lesson pages  →  faded_items  →  guided_items  →  independent_items
                                                    →  focused queue

   Scaffolding is chosen by the learner's BKT posterior for this KC's
   subtopic, using the same ERE bands as arena-unlock.js (< 0.75 ⇒ faded).
   Faded items are REAL bank questions re-served with a blanked starter, so
   grading, BKT updates and FIRe credit all run through the normal submit
   path — nothing here is a parallel scoring system.

   Once every ladder item is used, isActive() goes false and the normal
   adaptive queue takes over, still pinned to this subtopic by
   window.__kcFocusSubtopic (see questions.js).
   ================================================================ */

const KcPractice = (() => {
  const FADED_CEIL = 0.75; // ERE band shared with arena-unlock.js

  let active = false;
  let kcId = null;
  let kcTitle = null;      // the KP's own title — see _stamp
  let subtopicKeys = []; // see _resolveSubtopicKeys
  let queue = [];
  let served = 0;

  /* EXERCISE-SCOPED PRACTICE (2026-09-06). An ARENA exercise on the notebook
     page ("Practice make_rays_1d") starts this ladder on ONE KC and, on a
     miss, pulls that KC's PREREQUISITES in front of the remaining queue —
     Seth: "if you get it wrong, then it utilizes the existing rung structure
     but for the prerequisite problems". `scoped` is the record of that:
     which KC was asked for and which prerequisites have already been pulled
     in, so a second miss on the same concept does not queue the same drills
     twice. Every item carries its OWN kc/title (`_stamp` reads them), because
     a prerequisite drill served inside a make_rays_1d session is still a
     drill on `torch.out-argument` and the ladder card must say so.

     🔴 STILL NOT A SECOND SCORING SYSTEM. Grading, BKT, the rung estimate on
     the submit response — all unchanged. This file only decides ORDER. */
  let scoped = null; // { target, inserted: [kc, …] }

  /* This ladder's rung names, in the vocabulary the REST of the app already
     speaks. `item.kind` is this file's word for the authored bucket a drill
     came out of; `ladder_stage` is the backend's word for the rung it was
     served at, and every reader of a served question keys off that one —
     `practice/ladder.js` (_stageOf), `practice/stage-ladder.js`
     (STAGE_ALIASES), `practice/timer.js` (the pause snapshot). Mapped here
     rather than renaming `kind`, because the two are not the same thing: a
     `guided` item is an authored bucket, and the rung it serves is Faded.

     The pairs are the backend's own, from `kc_graph._STAGE_TO_RANKS` and the
     table above it:
       faded / guided  → `faded`    the blanked starter, or hints
       independent     → `partial`  write it unaided (displayed "Solo")
       integrated      → `solo`     the whole-KP problem (displayed "Integrated")
     Anything unrecognised maps to nothing and the readout hides, which is the
     behaviour this whole flow had before. */
  const STAGE_FOR_KIND = {
    faded: "faded",
    guided: "faded",
    independent: "partial",
    integrated: "solo",
  };

  /* The rungs that put support on the page — the same claim
     `kc_graph.stage_requires_support` makes server-side, and the reason it is
     only these two: a faded item hands over a blanked solution and a guided one
     hands over hints. An independent or integrated item hands over nothing. */
  const SUPPORTED_KINDS = new Set(["faded", "guided"]);

  /* 🔴 THE CONCEPT FIELDS EVERY READOUT KEYS OFF, PUT BACK ON A BANK QUESTION.

     `_hydrate` builds through `buildPracticeQuestionFromBank`, and the bank has
     no opinion about `ladder_kc` / `ladder_stage` / `ladder_kc_title` — those
     exist only on a question the BACKEND QUEUE served. `practice/timer.js` says
     so at length where the same gap broke a paused resume. Here it broke the
     Knowledge Graph's Practice ⤢ flow outright: `LadderUI.decorate` calls
     `_syncTopbar`, which reads exactly those two fields, finds neither, and
     calls `StageLadder.hide()` — so the ladder card went AND the topbar's
     concept pill went with it, on every question this ladder served. The one
     surface that is about a single concept was the one with no reading for it.
     Seth, 2026-09-01: "it gets rid of the competency bar at the top".

     Nothing here is a second scoring system, and it must not become one: the
     stage is the rung of the item being SERVED, which this file chose, and no
     number is invented. The fill within the rung still comes from the server's
     `ladder_estimate` on the submit response (`practice/events.js`
     setProgress), exactly as it does on the adaptive queue. */
  const _stamp = (q, item) => {
    const stage = STAGE_FOR_KIND[item.kind];
    const kc = item.kc || kcId;
    if (!q || !kc || !stage) return q;
    q.ladder_kc = kc;
    q.ladder_stage = stage;
    q.ladder_kc_title = item.kcTitle || (item.kc ? item.kc : kcTitle || kcId);
    q.ladder_support = SUPPORTED_KINDS.has(item.kind);
    q.ladder_integrated = item.kind === "integrated";
    return q;
  };

  // The two modes disagree on what a "subtopic" string is, and BKT is keyed by
  // whatever `question.subtopic` says:
  //   backend  → composite, "Numpy: Core array literacy" (questions.py prefixes
  //              the topic so Numpy/Einops subtopics stay distinct)
  //   local    → bare, "Core array literacy" (questions.json keeps the
  //              composite separately as subtopic_key)
  // Carrying BOTH and matching on either keeps the bar and the focus filter
  // correct in each mode without branching on practiceMode everywhere.
  const _resolveSubtopicKeys = (lesson, sampleBankQ) => {
    const keys = [];
    const push = (v) => { if (v && !keys.includes(v)) keys.push(v); };
    push(sampleBankQ && sampleBankQ.subtopic);
    push(sampleBankQ && sampleBankQ.subtopic_key);
    push(lesson && lesson.subtopic_key);
    return keys;
  };

  // Composite form — what the backend's focus_subtopic expects.
  const _compositeKey = () => subtopicKeys.find((k) => k.includes(": ")) || subtopicKeys[0] || null;

  const _mastery = () => {
    for (const key of subtopicKeys) {
      const live = window.__subtopicMastery && window.__subtopicMastery[key];
      if (Number.isFinite(live)) return live;
    }
    for (const key of subtopicKeys) {
      try {
        const p = getEwmaFromAdaptiveState(key);
        if (Number.isFinite(p)) return p;
      } catch (_err) { /* try the next key */ }
    }
    return null;
  };

  // The KP's numbered hints are one markdown list; the practice card shows a
  // single hint string. Keep the numbering — the hints are written to escalate
  // (conceptual nudge → names the function → near-solution), so a learner who
  // reads only the first line has still been helped the least.
  const _hintText = (markdown) =>
    String(markdown || "")
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .join("\n");

  // Three rungs, decreasing support: faded hands over a blanked solution,
  // guided hands over a hint, independent hands over nothing. Below the
  // independent band a learner climbs up; above it they start at the top and
  // the scaffolded items stay in reserve (still useful, just not the entry
  // point). Guided sits in the middle either way.
  const _buildQueue = (kp) => {
    const faded = (kp.faded_items || [])
      .filter((it) => Number.isFinite(it?.question_id))
      .map((it) => ({ kind: "faded", questionId: it.question_id, starter: it.starter_code || null }));
    const guided = (kp.guided_items || [])
      .filter((it) => Number.isFinite(it?.question_id))
      .map((it) => ({ kind: "guided", questionId: it.question_id, hint: _hintText(it.hints_markdown) }));
    const independent = (kp.independent_items || [])
      .filter((id) => Number.isFinite(id))
      .map((id) => ({ kind: "independent", questionId: id, starter: null }));
    /* The fourth rung. `integrated_items` is written by compile_lessons.py from
       a KP's `## Integrated practice`; KPs that have not been rewritten have
       none and this is empty, which is the same ladder as before. It always
       goes LAST, whichever way round the rest is ordered — a whole-KP problem
       is not an entry point for anybody, however strong the posterior. */
    const integrated = (kp.integrated_items || [])
      .filter((it) => Number.isFinite(it?.question_id))
      .map((it) => ({ kind: "integrated", questionId: it.question_id, starter: null }));
    const p = _mastery();
    const scaffoldFirst = !Number.isFinite(p) || p < FADED_CEIL;
    return scaffoldFirst
      ? [...faded, ...guided, ...independent, ...integrated]
      : [...independent, ...guided, ...faded, ...integrated];
  };

  // Any bank record for this KP — used only to read its subtopic naming.
  const _sampleBankQuestion = (kp) => {
    if (typeof getQuestionFromBank !== "function") return null;
    const ids = [
      ...(kp.faded_items || []).map((it) => it && it.question_id),
      ...(kp.guided_items || []).map((it) => it && it.question_id),
      ...(kp.independent_items || []),
      ...(kp.integrated_items || []).map((it) => it && it.question_id),
    ].filter((id) => Number.isFinite(id));
    for (const id of ids) {
      const q = getQuestionFromBank(id);
      if (q) return q;
    }
    return null;
  };

  /* The prerequisite ladder pulled in on a miss: one rung each of scaffolded
     → unaided, never the whole-KP integrated problem — the learner just missed
     the concept ABOVE this one; what they need is the supporting move, drilled
     small. ONE drill per rung: the first miss on make_rays_1d has five
     prerequisites, and their full ladders are 44 drills — more than any block
     the learner just sized. Three per concept keeps the pull proportionate,
     and a miss on one of THOSE pulls its own prerequisites the same way.
     Tagged with the prerequisite's own kc/title so the served question reads
     as that concept's rung. */
  const PREREQ_PER_RUNG = 1;
  const _prereqItems = (kp, kc) => {
    const tag = (it) => ({ ...it, kc, kcTitle: (kp && kp.title) || kc });
    const faded = (kp.faded_items || [])
      .filter((it) => Number.isFinite(it?.question_id))
      .slice(0, PREREQ_PER_RUNG)
      .map((it) => tag({ kind: "faded", questionId: it.question_id, starter: it.starter_code || null }));
    const guided = (kp.guided_items || [])
      .filter((it) => Number.isFinite(it?.question_id))
      .slice(0, PREREQ_PER_RUNG)
      .map((it) => tag({ kind: "guided", questionId: it.question_id, hint: _hintText(it.hints_markdown) }));
    const independent = (kp.independent_items || [])
      .filter((id) => Number.isFinite(id))
      .slice(0, PREREQ_PER_RUNG)
      .map((id) => tag({ kind: "independent", questionId: id, starter: null }));
    return [...faded, ...guided, ...independent];
  };

  const _hydrate = (item) => {
    const bankQ = typeof getQuestionFromBank === "function" ? getQuestionFromBank(item.questionId) : null;
    if (!bankQ) return null;
    const overrides = {};
    // Only override the starter when the faded item actually supplies one —
    // an empty string would wipe the question's own scaffold.
    if (item.kind === "faded" && item.starter) overrides.starter_code = item.starter;
    // Guided items carry the KP's authored hints. The bank question may have
    // its own hint; the lesson's is the specific one, so it wins.
    if (item.kind === "guided" && item.hint) overrides.hint = item.hint;
    return buildPracticeQuestionFromBank(bankQ, overrides);
  };

  /** Start the ladder for a KC. Returns false when the KC has no lesson data. */
  const start = async (kc) => {
    if (!window.LessonGate || typeof window.LessonGate.getKpEntry !== "function") return false;
    const entry = await window.LessonGate.getKpEntry(kc);
    if (!entry) return false;
    await loadQuestionsBank();

    kcId = kc;
    // The name the concept pill and the ladder's readout say out loud. The KP's
    // own title first — it is what the graph's node and its lesson pane both
    // show — with the KC id as the last resort so the readout is never blank.
    kcTitle = (entry.kp && entry.kp.title) || (entry.lesson && entry.lesson.title) || kc;
    // Keys must be resolved BEFORE _buildQueue — the faded/independent
    // ordering reads mastery, which is looked up by those keys.
    subtopicKeys = _resolveSubtopicKeys(entry.lesson, _sampleBankQuestion(entry.kp));
    queue = _buildQueue(entry.kp).map((it) => ({ ...it, kc, kcTitle }));
    served = 0;
    scoped = null;
    active = queue.length > 0;

    window.__kcFocusId = kc;
    window.__kcFocusSubtopics = subtopicKeys;
    window.__kcFocusSubtopic = _compositeKey();
    // The preview flag suppressed renderQuestion so a late resume couldn't
    // clobber the lesson. Graded practice starts now, so it has to come off.
    window.__lessonDemoOnly = false;

    if (window.CompetencyBar) {
      window.CompetencyBar.init(subtopicKeys);
      window.CompetencyBar.beginPractice();
    }
    return true;
  };

  /** Next ladder item, or null once the ladder is spent (queue takes over). */
  const nextQuestion = async () => {
    while (served < queue.length) {
      const item = queue[served++];
      const q = _hydrate(item);
      if (q) {
        q.ladder_kind = item.kind;
        _stamp(q, item);
        if (window.CompetencyBar) window.CompetencyBar.setPhaseKind(item.kind);
        return q;
      }
      // Item points at a question the bank no longer has (curated exclusion or
      // a regen drop) — skip it rather than dead-ending the ladder.
      console.warn("[kc-practice] ladder item missing from bank:", item.questionId);
    }
    active = false;
    return null;
  };

  const isActive = () => active && !!queue.length;

  /* ── exercise-scoped entry points (practice/exercise-session.js) ──── */

  const startScoped = async (kc) => {
    const ok = await start(kc);
    if (!ok) return false;
    scoped = { target: kc, inserted: [] };
    return true;
  };

  /* A miss on `kc` (the ladder_kc of the question just graded wrong) queues
     that concept's direct prerequisites in front of whatever is left. Returns
     how many drills were pulled in — 0 when nothing is scoped, the KC has no
     prerequisites with lessons, or they were already pulled in. Async because
     the KP entries come from lessons_structured.json through LessonGate. */
  const onMiss = async (kc) => {
    if (!scoped || !kc) return 0;
    const gate = window.LessonGate;
    if (!gate || typeof gate.getKpEntry !== "function") return 0;
    const entry = await gate.getKpEntry(kc);
    const prereqs = (entry && entry.kp && entry.kp.supporting_kcs) || [];
    const pulled = [];
    for (const pre of prereqs) {
      if (scoped.inserted.includes(pre) || pre === scoped.target) continue;
      /* Reserved BEFORE the await: two misses resolving together would both
         pass the check above and queue the same drills twice. Released again
         when the concept turns out to have nothing to pull. Codex, 2026-09-06. */
      scoped.inserted.push(pre);
      const preEntry = await gate.getKpEntry(pre);
      const items = preEntry && preEntry.kp ? _prereqItems(preEntry.kp, pre) : [];
      if (!items.length) {
        scoped.inserted = scoped.inserted.filter((k) => k !== pre);
        continue;
      }
      pulled.push(...items);
    }
    if (!pulled.length) return 0;
    queue.splice(served, 0, ...pulled);
    active = true;
    return pulled.length;
  };

  /* What the pause snapshot carries (practice/timer.js): the UNSERVED items
     and the scope record. `served` restarts at 0 over the remaining list —
     the served count the learner sees is timer.js's, not this one. */
  const serialize = () => {
    if (!kcId) return null;
    return {
      kc: kcId,
      title: kcTitle,
      subtopicKeys: subtopicKeys.slice(),
      remaining: queue.slice(served),
      scoped: scoped ? { target: scoped.target, inserted: scoped.inserted.slice() } : null,
    };
  };

  const restore = (saved) => {
    if (!saved || typeof saved !== "object" || !saved.kc) return false;
    kcId = String(saved.kc);
    kcTitle = saved.title ? String(saved.title) : kcId;
    subtopicKeys = Array.isArray(saved.subtopicKeys) ? saved.subtopicKeys.filter(Boolean) : [];
    queue = Array.isArray(saved.remaining)
      ? saved.remaining.filter((it) => it && Number.isFinite(it.questionId) && STAGE_FOR_KIND[it.kind])
      : [];
    served = 0;
    scoped = saved.scoped && saved.scoped.target
      ? { target: String(saved.scoped.target), inserted: Array.isArray(saved.scoped.inserted) ? saved.scoped.inserted.slice() : [] }
      : null;
    active = queue.length > 0;
    window.__kcFocusId = kcId;
    window.__kcFocusSubtopics = subtopicKeys;
    window.__kcFocusSubtopic = _compositeKey();
    window.__lessonDemoOnly = false;
    if (window.CompetencyBar) {
      window.CompetencyBar.init(subtopicKeys);
      window.CompetencyBar.beginPractice();
    }
    return true;
  };

  /* The scoped session ended (quota reached, discarded, failed to load): let go
     of the focus so the next plain block is the adaptive queue again. Only for
     a scoped run — the ?lesson=<kc> flow keeps its pin. */
  const stop = () => {
    if (!scoped) return;
    active = false;
    queue = [];
    served = 0;
    scoped = null;
    kcId = null;
    kcTitle = null;
    subtopicKeys = [];
    window.__kcFocusId = null;
    window.__kcFocusSubtopics = [];
    window.__kcFocusSubtopic = null;
  };

  const remaining = () => (active ? queue.length - served : 0);

  return {
    start,
    startScoped,
    onMiss,
    serialize,
    restore,
    stop,
    remaining,
    nextQuestion,
    isActive,
    get scoped() { return scoped ? { ...scoped, inserted: scoped.inserted.slice() } : null; },
    get kc() { return kcId; },
    get subtopicKeys() { return subtopicKeys.slice(); },
  };
})();

window.KcPractice = KcPractice;
