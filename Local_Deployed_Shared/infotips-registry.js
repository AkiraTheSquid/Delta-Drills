/* ================================================================
   INFOTIPS REGISTRY — what every ⓘ in the app says.

   Content only. The dot, the panel and the placement live in
   infotips.js; this file is the copy, kept separate so an explanation
   can be reworded without touching behaviour.

   A key here is matched by `data-dd-info="<key>"` in the markup. A key
   with no anchor is dead weight and a `data-dd-info` with no key here
   renders nothing, so the two lists have to stay in step —
   Local_Deployed_Shared/watch.py asserts that they do.

   HOUSE STYLE
     title  a handful of words, sentence case, the thing's own name
     body   one or two short sentences. Say what it does and what the
            learner gets from it. `body` is authored HTML (a <strong>
            or a <code>, nothing more) and is injected as innerHTML —
            never put anything user-supplied in here.
   ================================================================ */

window.DD_INFOTIPS = {
  /* ---- App chrome -------------------------------------------------- */
  /* 🔴 NO `tab.*` KEYS. Every tab used to carry one, and the ten dots they
     rendered were the loudest half of "there's information icons literally
     everywhere" (Seth, 2026-08-23). A tab's name is already the shortest
     description of that tab, and the two tabs that needed more than a name —
     Why This App Exists, How to use it — are entire pages of exactly that.
     Deleting the dots also straightened the active underline, which had been
     spanning the label while the eye read label+dot as the tab.
     `watch.py::check_infotips` asserts no `.tab-info` comes back. */
  "tp-banner": {
    title: "Targeted Practice mode",
    body: "Shown whenever a targeted session is running: questions come from the exercises you chose, not the adaptive queue. End it here to go back to normal Practice.",
  },

  /* ---- Practice: setting up ---------------------------------------- */
  "self-report": {
    title: "Starting point",
    body: "A prior, not a placement: it only sets where the queue <em>begins</em>. Your answers move it either way within a few questions.",
  },
  placement: {
    title: "Placement test",
    body: "A test that jumps between different topics in order to determine what concepts you need to learn for PyTorch.<br><br>Choosing your level influences what difficulty you start at for the test, so it's okay if you're not sure where your level is at.",
  },
  /* "placement-timer" was DELETED on 2026-08-23 with its anchor's ⓘ. The
     countdown moved out of .question-number-row and onto the notch tab
     (index.html), where a sibling dot would sit between the clock and the
     three dots in a 15px row. The copy — every probe gets the same 2:00, you
     do not set it, running out submits what you have — is what the tab's own
     `title` already says. Keys and anchors are asserted to match both ways
     (watch.py `check_infotips`), so this had to go when the anchor did. */
  /* ---- The four rungs, named inside the bar ------------------------ */
  "ladder.lesson": {
    title: "Lesson",
    body: "The teaching page for this concept. You read the explanation and run the examples — there is no drill on this rung, and leaving it is what earns you one.",
  },
  "ladder.faded": {
    title: "Faded",
    body: "Most of the solution is already written and the part that <strong>uses this concept</strong> is blanked out. You supply that part. Get a run of them right and the scaffold comes off.",
  },
  "ladder.example": {
    title: "Solo",
    body: "You write the whole function, unaided, on one idea at a time. Some of these open with a short solved example — the ones introducing a move you have not used yet — and they run out as you work through the rung, which is the point.",
  },
  "ladder.solo": {
    title: "Integrated",
    body: "The top rung: problems that need <strong>every</strong> idea in this concept at once, with nothing to read first. The bar stops here — the app decides the concept is <strong>learned</strong> from your record, not from arriving.",
  },
  "stage-ladder-kc": {
    title: "Current concept",
    body: "Skill this problem targets. Click to open it on knowledge graph.",
  },
  "concept-understanding": {
    title: "Concept understanding",
    body: "BKT estimate for this concept. Tier says whether mapping is measured or borrowed from its topic; coverage says how much rests on atoms you have attempted.",
  },

  /* ---- Practice: the problem --------------------------------------- */
  /* No `concept-topbar`, `stage-dots`, `difficulty` or `competency-bar` keys.
     All four annotated widgets the 2026-08-19 one-ladder change deleted: the
     concept strip and its four rung dots, the 96px difficulty copy inside it,
     and the single-KC mastery track. A key with no anchor and an anchor with no
     key both fail silently at runtime, which is why `watch.py::check_infotips`
     asserts both directions — it is what caught the leftovers each time. */
  // "cold-start" removed 2026-08-23 with #cold-start-badge. (Its copy was also
  // stale: it described the fixed-ramp calibration that was replaced by
  // adaptive-from-question-1 selection.)
  "torch-colab": {
    title: "PyTorch drills run in Colab",
    body: "The in-browser sandbox can't <code>import torch</code> inside the time limit, so this one opens as a notebook. Work it there, then tell us how it went.",
  },

  /* ---- Practice: after you answer ---------------------------------- */
  "feedback-rating": {
    title: "How far off was the difficulty?",
    body: "Difficulty is aimed by your mastery; this says how well that aim landed and steers the next problem. <strong>About right</strong> is the safe default.",
  },
  /* Legacy `ewma-accuracy` class now hosts scoped KC understanding. Difficulty
     stays text under ladder; it is not another progress track. */

  /* ---- Knowledge graph --------------------------------------------- */
  "kg-map": {
    title: "The map",
    body: "Every skill in the curriculum, with an edge from each prerequisite to what it unlocks. Click a node to read what it teaches and where you stand. Drag to pan, scroll to zoom, <strong>Fit</strong> to get back.",
  },
  "kg-legend": {
    title: "Legend",
    body: "What the node colours mean — mastered, unlocked and ready, or still locked behind a prerequisite.",
  },
  "kg-info": {
    title: "Concept detail",
    body: "What the selected concept covers, what it needs first, and your current mastery of it.",
  },
  "kg-practice-max": {
    title: "Practice ⤢",
    body: "Opens a full practice screen for this one skill, instead of letting the adaptive queue choose the topic.",
  },
  "kg-colab-link": {
    title: "Open in Colab",
    body: "Colab edition only: opens the lesson notebook for this concept.",
  },

  /* ---- Courses ------------------------------------------------------ */
  "github-username": {
    title: "GitHub username",
    body: "Fork <code>ARENA_3.0</code> and paste your handle here, and Colab links open <strong>your</strong> fork — so notebooks you save come back next time. Leave blank for the read-only upstream copy.",
  },

  /* ---- Account ------------------------------------------------------ */
  "account-mode": {
    title: "Advanced mode",
    body: "Off, the app is just the drills: <strong>Practice</strong> and the <strong>Diagnostic</strong>. On, it also shows the machinery behind them — the knowledge graph, the ARENA course content, the notebooks and targeted practice. Nothing is deleted either way; the toggle only changes which tabs are in the nav.",
  },
  "account-advanced": {
    title: "Advanced settings",
    body: "Developer knobs: which backend to talk to, third-party API keys, and your API token. Nothing here is needed to practise.",
  },
  "dd-token": {
    title: "DD_TOKEN",
    body: "Paste this into a drill notebook when it asks, and work done there counts against your account. It's your session token — keep it private.",
  },
  "api-base": {
    title: "API base URL",
    body: "Which backend this browser talks to. Leave blank unless you're pointing the app at your own server.",
  },

  /* ---- Targeted practice -------------------------------------------- */
  "tp-search": {
    title: "Search exercises",
    body: "Find ARENA exercises by their heading — type a couple of characters of a topic, a function, or a section name.",
  },
  "tp-selected": {
    title: "Your queue",
    body: "The exercises you've chosen. Submit to start a session drawing only from these; the adaptive queue is paused until you end it.",
  },

  /* ---- ARENA unlock -------------------------------------------------- */
  "arena-unlock": {
    title: "Exercise unlocked",
    body: "Your scores now clear every prerequisite for a real ARENA exercise. Go do it in Colab before the next drill — that's the point of the drills.",
  },
  "arena-unlock-timer": {
    title: "Exercise timer",
    body: "Start it when you open the notebook. Elapsed against target time becomes the rating that moves your prerequisite scores.",
  },
  "arena-unlock-rating": {
    title: "How did it go?",
    body: "Your honest self-rating is the only evidence the app gets from a Colab exercise, and it's what updates the scores shown next.",
  },
};
