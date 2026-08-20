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
  /* ---- Tabs -------------------------------------------------------- */
  "tab.why-this-app": {
    title: "Why this app exists",
    body: "The reasoning behind Delta Drills — the gap it was built to close, and why it works the way it does.",
  },
  "tab.how-it-works": {
    title: "How It Works",
    body: "A walkthrough of the machinery: how the app picks your next problem, how mastery is estimated, and what the scaffolding does.",
  },
  "tab.knowledge-graph": {
    title: "Knowledge Graph",
    body: "A map of every skill and what depends on what. Use it to see why something is locked and what unlocking it opens up. <strong>Practice</strong> picks from this map for you.",
  },
  "tab.courses": {
    title: "Course content",
    body: "The ARENA curriculum this app drills you for — every chapter and section, each one linking out to the original notebook.",
  },
  "tab.practice": {
    title: "Practice",
    body: "Your adaptive queue. The app chooses each next problem from what you're ready for, so you just answer.",
  },
  "tab.notebooks": {
    title: "Notebooks",
    body: "A whole lesson as one runnable notebook — the same cells the Colab edition publishes, on one live Python session on the server.",
  },
  "tab.targeted-practice": {
    title: "Targeted Practice",
    body: "You pick the skill instead. Search ARENA exercises, choose the ones you want, and drill only those.",
  },
  "tab.account": {
    title: "Account",
    body: "Who you're signed in as, your GitHub fork for Colab persistence, and the developer settings behind <strong>Advanced</strong>.",
  },
  "tab.split-tool": {
    title: "Split Tool",
    body: "An internal utility that cuts a PDF into chapters. Not part of the learning surface.",
  },

  /* ---- App chrome -------------------------------------------------- */
  "guest-banner": {
    title: "Guest mode",
    body: "Everything works signed out, but progress lives in <strong>this browser only</strong>. Continue with Google to keep it across devices.",
  },
  "topbar-auth": {
    title: "Signed in",
    body: "The account your progress is being saved to. Mastery follows the account, not the browser.",
  },
  "tp-banner": {
    title: "Targeted Practice mode",
    body: "Shown whenever a targeted session is running: questions come from the exercises you chose, not the adaptive queue. End it here to go back to normal Practice.",
  },

  /* ---- Practice: setting up ---------------------------------------- */
  "session-setup": {
    title: "Session setup",
    body: "Decide the block before you start — how many questions, and how long you get to answer and to review each one. Once it's running the timers are strict.",
  },
  "session-resume": {
    title: "Resume a session",
    body: "<strong>Pause &amp; exit</strong> keeps the exact question you were on. This brings it back rather than starting over.",
  },
  "self-report": {
    title: "Starting point",
    body: "A prior, not a placement: it only sets where the queue <em>begins</em>. Your answers move it either way within a few questions.",
  },
  placement: {
    title: "Placement diagnostic",
    body: "A short adaptive run that locates your level across every topic at once. It's shorter where you've already practised, and it can never lower mastery you've earned.",
  },
  "session-status": {
    title: "Session status",
    body: "Where you are in the block, which phase you're in, and the countdown. Answer time auto-submits; review time loads the next problem.",
  },

  /* ---- Practice: the problem --------------------------------------- */
  "stage-ladder-kc": {
    title: "Current concept",
    body: "The skill this problem is tagged to. Click it to open that concept on the knowledge graph and check the choice for yourself.",
  },
  "stage-ladder": {
    title: "Where you are",
    body: "Five rungs from reading an explanation to solving several concepts at once unaided. The section you're on is filled as far as your record has taken you toward the next one; the rungs after it are what's still ahead.",
  },
  "stage-ladder-difficulty": {
    title: "Difficulty",
    body: "Two different numbers, on purpose. The <em>aim</em> is where the queue is pointing for you right now. The <em>rating</em> is this problem's own — the queue picks by concept and rung first, then takes the closest difficulty it can reach, so the two rarely match exactly.",
  },
  /* No `concept-topbar`, `stage-dots`, `difficulty` or `competency-bar` keys.
     All four annotated widgets the 2026-08-19 one-ladder change deleted: the
     concept strip and its four rung dots, the 96px difficulty copy inside it,
     and the single-KC mastery track. A key with no anchor and an anchor with no
     key both fail silently at runtime, which is why `watch.py::check_infotips`
     asserts both directions — it is what caught the leftovers each time. */
  "cold-start": {
    title: "Calibrating",
    body: "The first three questions use fixed difficulties to find your level. Nothing you answer here moves the difficulty the queue aims at — it is locating you first.",
  },
  "question-id": {
    title: "Problem ID",
    body: "The stable identifier for this problem. Click to copy it — worth quoting when reporting something broken.",
  },
  "graph-jump": {
    title: "See in knowledge graph",
    body: "Opens the concept this problem is tagged to, so you can check the tutor's choice against the map it claims to be following.",
  },
  "question-imports": {
    title: "Imported helpers",
    body: "Names already defined for you in the sandbox. Use them directly — you don't need to write or import them.",
  },
  "question-visual": {
    title: "Target image",
    body: "What your code is supposed to produce. Run yours and compare against it.",
  },
  "torch-colab": {
    title: "PyTorch drills run in Colab",
    body: "The in-browser sandbox can't <code>import torch</code> inside the time limit, so this one opens as a notebook. Work it there, then tell us how it went.",
  },
  hints: {
    title: "Hints",
    body: "A nudge toward the idea, not the answer. Taking one is recorded and counts as partial help when your mastery is updated.",
  },
  "submit-area": {
    title: "Submit, skip, or say so",
    body: "<strong>Submit</strong> grades your code. <strong>Skip</strong> records nothing at all. During placement, <strong>I don't know yet</strong> is faster and more honest than guessing.",
  },
  editor: {
    title: "Code editor",
    body: "Write your solution here. It runs in a Python sandbox in your browser — nothing is sent anywhere to be executed.",
  },
  runner: {
    title: "Run",
    body: "Executes your code and prints the result below, as many times as you like. Running is not submitting and is never graded.",
  },

  /* ---- Practice: after you answer ---------------------------------- */
  "feedback-rating": {
    title: "How far off was the difficulty?",
    body: "Difficulty is aimed by your mastery; this says how well that aim landed and steers the next problem. <strong>About right</strong> is the safe default.",
  },
  "missed-fact": {
    title: "Missed one specific thing",
    body: "Separates “I forgot a single detail” from “this was too hard”, so one missed fact doesn't drag the whole difficulty estimate down.",
  },
  "problem-flags": {
    title: "Report a problem",
    body: "Flags the <em>content</em>, not your answer — broken, unclear, wrong image. It doesn't affect your score.",
  },
  /* No `ewma-accuracy` or `target-difficulty` keys either. The accuracy bar is
     deleted and difficulty is a caption on the ladder — see
     `stage-ladder-difficulty` above. */
  solution: {
    title: "Solution",
    body: "The worked answer, revealed after you submit. Read it even when you were right — the shorter route is often the point.",
  },
  "ai-explanation": {
    title: "AI explanation",
    body: "An on-demand walkthrough of this specific problem and your attempt at it.",
  },
  tutor: {
    title: "Tutor",
    body: "Follow-up chat about the problem you just finished. It already has the question, your code, the canonical solution, and the explanation above — so ask the next question, not the whole story again. The thread resets with each new problem.",
  },

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
  "courses-about": {
    title: "What this app is for",
    body: "Delta Drills exists to get you through <strong>one</strong> curriculum — ARENA. Below is that curriculum in order: open a chapter for its sections, and each section links out to Callum McDougall's original Colab notebook.",
  },
  "github-username": {
    title: "GitHub username",
    body: "Fork <code>ARENA_3.0</code> and paste your handle here, and Colab links open <strong>your</strong> fork — so notebooks you save come back next time. Leave blank for the read-only upstream copy.",
  },

  /* ---- Account ------------------------------------------------------ */
  "account-identity": {
    title: "Identity",
    body: "The email this progress is saved under. <strong>Guest</strong> means it's in this browser only.",
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
