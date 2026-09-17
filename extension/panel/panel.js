/* ================================================================
   PANEL.JS — the side panel's state machine.

   Flow, once connected:

     next-question ──▶ [lesson gate?] ──▶ problem + timer
                            │                   │
                       worked-seen          mark right/wrong
                            │                   │  (or timer expiry → wrong)
                            └───────────────────▶ submit-local-eval
                                                  │
                                            settle + undo window
                                                  │
                                                  ▶ next-question

   Nothing here decides what comes next. `nextQuestion()` is the scheduler.
   ================================================================ */

/* Everything is renamed on the way in, and that is load-bearing.
   These are classic <script> tags, not modules, so api.js, navigate.js and this
   file all share ONE global lexical scope. Destructuring `api` or `slugKc` under
   its own name redeclares the `const` the earlier file already made, and the
   whole of panel.js dies with a SyntaxError before a single handler is wired.
   The symptom is an empty panel with a ⚙ that does nothing — it reads like a CSS
   or a manifest problem and it is neither.
   Verify after touching any of these files:
       cat panel/notebook-index.js panel/notebook-index-questions.js \
           panel/notebook-index-concepts.js panel/api.js panel/navigate.js \
           panel/panel.js | node --check /dev/stdin
   `node --check` on the files one at a time cannot see this. */
const { api: ddApi, tab: ddTab, notebooks: ddNotebooks } = window.DD;
const {
  slugKc: navSlugKc,
  whoIsOpen: navWhoIsOpen,
  jumpTo: navJumpTo,
  paintNotebook: navPaintNotebook,
  renderNotebookList: navRenderNotebookList,
} = window.DDNav;

const DEFAULT_SECONDS = 180;
const SETTLE_SECONDS = 4;

const $ = (id) => document.getElementById(id);

const state = {
  q: null, // the current NextQuestionResponse
  deadline: 0, // epoch ms the timer expires
  tick: null, // countdown interval handle
  settle: null, // settle-countdown interval handle
  graded: false, // guards double-submit from click + expiry racing
  nav: 0, // render sequence, so a slow identify can't paint over a newer view
};

/* ── view switching ──────────────────────────────────────────── */

const VIEWS = ["connect", "loading", "gate", "problem", "settled", "error"];

function show(name, loadingText) {
  VIEWS.forEach((v) => $(`view-${v}`).classList.toggle("active", v === name));
  if (name === "loading" && loadingText) $("loading-text").textContent = loadingText;
}

function setConn(ok) {
  const el = $("conn");
  el.className = `conn ${ok ? "on" : "off"}`;
  el.textContent = ok ? "●" : "○";
}

function fail(err) {
  clearTimers();
  setConn(false);
  $("err-text").textContent = err.message || String(err);
  $("err-text").className = "status err";
  show("error");
}

/* ── timers ──────────────────────────────────────────────────── */

function clearTimers() {
  if (state.tick) clearInterval(state.tick);
  if (state.settle) clearInterval(state.settle);
  state.tick = null;
  state.settle = null;
}

function startTimer(seconds) {
  clearTimers();
  const total = seconds * 1000;
  state.deadline = Date.now() + total;

  const paint = () => {
    const left = Math.max(0, state.deadline - Date.now());
    const frac = left / total;
    const fill = $("t-fill");
    fill.style.width = `${frac * 100}%`;
    fill.className = frac <= 0 ? "out" : frac < 0.25 ? "low" : "";

    const s = Math.ceil(left / 1000);
    $("t-label").textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(
      s % 60,
    ).padStart(2, "0")}`;

    if (left <= 0) {
      clearInterval(state.tick);
      state.tick = null;
      // Timer expiry is a wrong answer. This is the design: an unbounded
      // problem produces no evidence, and "I ran out of time" is the same
      // signal to the ladder as "I got it wrong".
      grade(false, "Time's up.");
    }
  };

  paint();
  state.tick = setInterval(paint, 1000);
}

/* ── rendering ───────────────────────────────────────────────── */

const RUNG_LABEL = {
  worked: "Worked example",
  faded: "Faded",
  partial: "Partly faded",
  solo: "Independent",
  independent: "Independent",
};

function renderScaffold(q) {
  const host = $("p-scaffold");
  host.textContent = "";
  const stage = q.ladder_stage || "";

  // The starter the server cut for this rung. On the faded/partial rungs this
  // is the canonical solution with its TAIL removed (backward fading), not a
  // blank page — so showing it IS the scaffold.
  if (q.starter_code && stage !== "solo" && stage !== "independent") {
    const pre = document.createElement("pre");
    pre.textContent = q.starter_code;
    host.appendChild(pre);
  }

  if (q.hint) {
    const d = document.createElement("details");
    d.className = "hint";
    const s = document.createElement("summary");
    s.textContent = "Show a hint";
    const p = document.createElement("p");
    p.textContent = q.hint;
    d.append(s, p);
    host.appendChild(d);
  }
}

function renderMastery(q) {
  const est = q.ladder_estimate || {};
  const p = typeof est.p === "number" ? est.p : null;
  $("p-mastery-fill").style.width = `${(p ?? 0) * 100}%`;
  $("p-mastery-label").textContent =
    p === null
      ? "no record yet"
      : `${Math.round(p * 100)}% over ${est.n || 0} attempt${est.n === 1 ? "" : "s"}`;
}

function renderProblem(q) {
  state.q = q;
  state.graded = false;

  $("p-kc").textContent = q.ladder_kc_title || q.subtopic || q.topic || "Practice";
  $("p-rung").textContent = RUNG_LABEL[q.ladder_stage] || "Practice";
  $("p-text").textContent = q.question_text || "";
  $("goto-status").textContent = "";
  $("goto-status").className = "status small";

  renderMastery(q);
  renderScaffold(q);

  // Paint the notebook the problem lives in immediately from the index, then
  // fill in whether it is the open one once the tab answers.
  const target = ddNotebooks.forQuestion(q);
  navPaintNotebook("p", target, null);
  refreshOpenNotebook("p", target);

  $("btn-right").disabled = false;
  $("btn-wrong").disabled = false;

  show("problem");
  startTimer(DEFAULT_SECONDS);
}

function renderGate(q) {
  state.q = q;
  const g = q.lesson_gate[0];
  // A segmented KP arrives one concept at a time, so the heading names the
  // concept and the subtitle says which of how many. Without the counter the
  // learner meets the same KP three times over and has no way to tell that it
  // is progress rather than the tutor repeating itself.
  const total = Number(g.segment_total) || 1;
  $("gate-title").textContent =
    (total > 1 && g.segment_title) || g.kp_title || g.kc_title || g.kc;
  const where = total > 1
    ? `${g.kp_title || g.kc_title || g.kc} — concept ${(Number(g.segment_index) || 0) + 1} of ${total}`
    : g.lesson_title || g.lesson_id || "";
  $("gate-sub").textContent = `${where} — you haven't met this concept yet, so the tutor is teaching it before it asks anything.`;
  $("gate-status").textContent = "";
  $("gate-status").className = "status small";

  const target = ddNotebooks.forGate(g);
  navPaintNotebook("gate", target, null);
  refreshOpenNotebook("gate", target);

  show("gate");
}

/**
 * Fill in the "open / switch" badge once the Colab tab answers.
 *
 * Guarded by a render sequence: identifying a tab takes a round trip, and the
 * student can be on the next problem by the time it returns.
 */
async function refreshOpenNotebook(prefix, target) {
  const seq = ++state.nav;
  const open = await navWhoIsOpen();
  if (seq !== state.nav) return;
  navPaintNotebook(prefix, target, open);
  if (prefix === "p") {
    const same = !target || (open.ok && open.lessonId === target.id);
    $("btn-goto").textContent = same ? "Go to the cell ↓" : "Open the notebook & go ↗";
  }
}

/* ── the loop ────────────────────────────────────────────────── */

async function advance() {
  clearTimers();
  show("loading", "Asking the tutor what's next…");
  try {
    const q = await ddApi.nextQuestion();
    setConn(true);
    if (q.lesson_gate && q.lesson_gate.length) renderGate(q);
    else renderProblem(q);
  } catch (err) {
    if (err.status === 401) {
      await ddApi.signOut();
      show("connect");
      $("connect-status").textContent = "Session expired — sign in again.";
      $("connect-status").className = "status err";
      setConn(false);
      return;
    }
    fail(err);
  }
}

async function grade(correct, note) {
  if (state.graded || !state.q) return;
  state.graded = true;
  clearTimers();

  const qid = state.q.question_id;
  $("btn-right").disabled = true;
  $("btn-wrong").disabled = true;

  $("s-mark").textContent = correct ? "✓" : "✕";
  $("s-mark").className = `verdict ${correct ? "ok" : "no"}`;
  $("s-text").textContent = note || (correct ? "Logged as correct." : "Logged as wrong.");
  show("settled");

  try {
    await ddApi.submitLocal(qid, correct);
  } catch (err) {
    fail(err);
    return;
  }

  // Undo window: `/override` flips the attempt the server just recorded, so a
  // misclick (or a timer that expired while the student was still reading) is
  // recoverable without polluting the ladder.
  let left = SETTLE_SECONDS;
  const paint = () => {
    $("s-count").textContent = `Next problem in ${left}s`;
    if (left-- <= 0) {
      clearInterval(state.settle);
      state.settle = null;
      advance();
    }
  };
  paint();
  state.settle = setInterval(paint, 1000);

  $("btn-undo").onclick = async () => {
    clearTimers();
    $("btn-undo").disabled = true;
    try {
      await ddApi.override(qid, !correct);
      $("s-text").textContent = `Flipped to ${!correct ? "correct" : "wrong"}.`;
    } catch (err) {
      fail(err);
      return;
    } finally {
      $("btn-undo").disabled = false;
    }
    setTimeout(advance, 700);
  };
}

/* ── navigation into the notebook ──────────────────────────────
   Resolving which notebook, switching to it and finding the cell all live in
   navigate.js; this is only the wiring from the current view. */

async function goToCurrentCell() {
  const q = state.q;
  if (!q) return;
  const target = ddNotebooks.forQuestion(q);
  const out = await navJumpTo({
    target,
    anchor: `dd-q${q.question_id}`,
    text: `Problem ${q.question_id}`,
    status: $("goto-status"),
    arrived: "There it is — run the cell.",
  });
  if (out.open || out.switched) navPaintNotebook("p", target, out.open);
}

/* ── connect ─────────────────────────────────────────────────── */

async function doLogin() {
  const email = $("in-email").value.trim();
  const pass = $("in-pass").value;
  const status = $("connect-status");
  if (!email || !pass) {
    status.textContent = "Email and password, please.";
    status.className = "status err";
    return;
  }
  status.textContent = "Signing in…";
  status.className = "status";
  $("btn-login").disabled = true;
  try {
    await ddApi.setBase($("in-base").value.trim());
    await ddNotebooks.setRepo($("in-repo").value);
    await ddApi.login(email, pass);
    $("in-pass").value = "";
    advance();
  } catch (err) {
    status.textContent = err.message;
    status.className = "status err";
  } finally {
    $("btn-login").disabled = false;
  }
}

async function useToken() {
  const token = $("in-token").value.trim();
  if (!token) return;
  await ddApi.setBase($("in-base").value.trim());
  await ddNotebooks.setRepo($("in-repo").value);
  await ddApi.setToken(token);
  advance();
}

/* ── wiring ──────────────────────────────────────────────────── */

$("btn-login").onclick = doLogin;
$("btn-token").onclick = useToken;
$("in-pass").addEventListener("keydown", (e) => {
  if (e.key === "Enter") doLogin();
});

$("btn-goto").onclick = goToCurrentCell;
$("btn-right").onclick = () => grade(true);
$("btn-wrong").onclick = () => grade(false);
$("btn-retry").onclick = advance;

$("btn-signout").onclick = async () => {
  await ddApi.signOut();
  setConn(false);
  show("connect");
};
$("settings-btn").onclick = async () => {
  clearTimers();
  const { base } = await ddApi.load();
  await ddNotebooks.load();
  $("in-base").value = base;
  $("in-repo").value = ddNotebooks.repo;
  navRenderNotebookList();
  show("connect");
};

$("in-repo").addEventListener("change", async () => {
  await ddNotebooks.setRepo($("in-repo").value);
  $("in-repo").value = ddNotebooks.repo;
  navRenderNotebookList();
});

$("btn-forget-nb").onclick = async () => {
  await ddNotebooks.forget();
  navRenderNotebookList();
};

$("btn-gate-goto").onclick = async () => {
  const g = state.q?.lesson_gate?.[0];
  if (!g) return;
  const target = ddNotebooks.forGate(g);
  const out = await navJumpTo({
    target,
    // One CONCEPT when the KP teaches several, the whole KP when it does not.
    // The gate already says which concept it is owed (`exposure_key`); opening
    // the KP instead puts all three on screen under a heading that claims to
    // be one of them.
    anchor: ddNotebooks.anchorForGate(g) || `dd-kp-${navSlugKc(g.kc)}`,
    text: g.kp_title || g.kc_title,
    status: $("gate-status"),
    arrived: "Read this, then come back.",
  });
  if (out.open || out.switched) navPaintNotebook("gate", target, out.open);
};

$("btn-gate-done").onclick = async () => {
  const q = state.q;
  const g = q?.lesson_gate?.[0];
  if (!g) return;
  show("loading", "Recording that you've read it…");
  try {
    // The CONCEPT that was read, not the KP it belongs to. `<kc>` in the
    // exposure map means the whole KP is done and every later gate reads it
    // that way, so posting it after one concept of three retires the other two
    // unread — silently, permanently, and with nothing ever asking again.
    await ddApi.markExposed(q.lesson_gate.map((x) => x.exposure_key || x.kc));
    // worked_seen is a separate counter from exposure: exposure fires once and
    // gates the KC's first question, worked_seen tells the ladder the concept
    // has been taught and lets it leave the `worked` rung.
    await ddApi.workedSeen(g.kc, q.question_id);
    renderProblem(q);
  } catch (err) {
    fail(err);
  }
};

/* ── boot ────────────────────────────────────────────────────── */

(async function boot() {
  const { base, token } = await ddApi.load();
  await ddNotebooks.load();
  $("in-base").value = base;
  $("in-repo").value = ddNotebooks.repo;
  navRenderNotebookList();
  if (!token) {
    setConn(false);
    show("connect");
    return;
  }
  advance();
})();
