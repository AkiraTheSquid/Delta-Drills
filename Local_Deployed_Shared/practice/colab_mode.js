/* ================================================================
   COLAB_MODE.JS — the Colab edition switch, and question → notebook.

   WHAT THIS IS
     Delta Drills ships from two Vercel projects off the same code:

       delta-drills.vercel.app        the normal app — you solve in the editor
       delta-drills-colab.vercel.app  the Colab edition — you solve in the
                                      notebook, and the app is the tutor beside it

     Same bank, same Fly backend, same Supabase state, so practice on either
     one moves the same mastery record. The ONLY difference is which surface a
     drill is answered on, and that is decided here.

   WHY IT IS A HOST CHECK AND NOT A BRANCH
     A forked branch drifts. The two deploys carry byte-identical JavaScript and
     ask `location.hostname` who they are, so a fix lands on both the next time
     each is deployed and there is nothing to keep in sync by hand. `?mode=colab`
     forces it on for a local check, `?mode=app` forces it off — neither is
     persisted, because a sticky flag on the main host would be a fork nobody
     could see.

   HOW A QUESTION FINDS ITS NOTEBOOK
     It does NOT come from the question. `problem_notebook_path` /
     `solution_notebook_path` are empty on all 499 rows in the bank — which is
     exactly why the old Colab route was unreachable — so the join runs through
     the generated index:

       question id → colab_notebooks.json `questions` → lesson → <lesson>.ipynb

     That file is written by `scripts/generate_colab_notebooks.py` alongside the
     notebooks themselves, so the map cannot drift from what was published. It
     covers 424 of 499 questions.

     ⚠️ Join on the QUESTION, never on `subtopic_key`. A lesson notebook holds
     only the problems its KPs reference, not every question in the subtopic —
     so a subtopic-level match resolves questions whose cell is not in the file
     it points at. Tried that first: q1 matched `np-3` by subtopic, and `np-3`
     has no `dd-q1` cell. The learner would have been sent to a notebook to work
     a drill that is not in it, with the editor hidden behind them.

   THE ONE RULE
     Never route a question to Colab that has no notebook. Routing hides the
     whole editor panel, the aids and the submit bar, so a question sent to a
     notebook that does not exist is a dead end with no controls and no way
     forward. `colabNotebookHrefFor` returning "" is the guard, and it is the
     same guard the pre-fork code had for a different reason.
   ================================================================ */

(function initColabMode() {
  // The fork's own hosts. Vercel also serves every deploy at a generated
  // `delta-drills-colab-<hash>-<scope>.vercel.app`, and a preview of the fork is
  // still the fork — hence the prefix match rather than an exact one.
  const HOST_RE = /^delta-drills-colab(?:[-.]|$)/i;
  const INDEX_URL = "lessons/colab_notebooks.json";
  const DEFAULT_OWNER = "AkiraTheSquid";

  let forced = null;              // ?mode= wins over the hostname, per load
  try {
    const mode = new URLSearchParams(location.search).get("mode");
    if (mode === "colab") forced = true;
    else if (mode === "app" || mode === "editor") forced = false;
  } catch (_) { /* no URL API / opaque origin — fall through to the host */ }

  const active = forced === null ? HOST_RE.test(location.hostname) : forced;

  // question id → lesson entry. Null until the index lands; the resolver answers
  // "" in the meantime, which reads as "no notebook" and keeps the learner on
  // the editor rather than flashing a Colab card that cannot resolve.
  let byQuestion = null;
  let repo = "";
  let pathPrefix = "";
  const readyWaiters = [];

  // Read directly rather than through `stats/predicted-links.js`'s
  // `_accountGithubUsername`: this file has to work no matter where it sits in
  // the script order, and the coupling would be invisible until it broke.
  function owner() {
    try {
      return (localStorage.getItem("account_github_username") || "").trim() || DEFAULT_OWNER;
    } catch (_) {
      return DEFAULT_OWNER;
    }
  }

  function ingest(data) {
    // "arena-book-colab/ARENA_5.0/ch-1-foundations" — the repo the notebooks
    // were published to, then the path inside it. Split rather than hardcoded so
    // republishing under another repo name needs no code change.
    const dir = String((data && data.dir) || "");
    const cut = dir.indexOf("/");
    repo = cut === -1 ? dir : dir.slice(0, cut);
    pathPrefix = cut === -1 ? "" : dir.slice(cut + 1);
    const lessons = Object.create(null);
    ((data && data.lessons) || []).forEach((lesson) => {
      if (lesson && lesson.id && lesson.file) lessons[lesson.id] = lesson;
    });
    // `questions` is {"<question id>": "<lesson id>"} — the map the generator
    // writes from the cells it actually emitted.
    const map = Object.create(null);
    const pairs = (data && data.questions) || {};
    Object.keys(pairs).forEach((qid) => {
      const lesson = lessons[pairs[qid]];
      if (lesson) map[String(qid)] = lesson;
    });
    byQuestion = map;
  }

  function load() {
    if (!active) return;
    fetch(INDEX_URL, { cache: "no-cache" })
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((data) => {
        ingest(data);
        console.log("[colab-mode] notebook index loaded:", Object.keys(byQuestion).length, "questions");
      })
      .catch((err) => {
        // An unreachable index is not fatal: every question keeps its editor.
        byQuestion = Object.create(null);
        console.warn("[colab-mode] notebook index unavailable — staying on the editor:", err);
      })
      .then(() => {
        while (readyWaiters.length) {
          const fn = readyWaiters.shift();
          try { fn(); } catch (_) { /* one bad waiter must not strand the rest */ }
        }
      });
  }

  function questionId(q) {
    const raw = q && (q.question_id != null ? q.question_id : q.id);
    return raw == null ? "" : String(raw);
  }

  /**
   * The Colab URL for this question's lesson notebook, or "" when there is none.
   *
   * The `#scrollTo=dd-q<id>` fragment is the anchor
   * `generate_colab_notebooks.py` mints on each problem's header cell, so the
   * notebook opens AT the problem rather than at the top of an 84-problem file.
   * A wrong or missing anchor costs nothing — Colab simply does not scroll.
   */
  /**
   * Why did (or didn't) this question route? One line per question, in the
   * console.
   *
   * Not decoration. This decision has five inputs — the edition flag, the
   * index having loaded, the question carrying an id, that id being in the map,
   * and the notebook repo being known — and when it comes out "false" the page
   * just shows an editor, which is also what a question with no notebook looks
   * like, and what the normal app looks like. Reading it off the screen is
   * impossible; reading it off the console is instant.
   */
  let lastExplained = "";
  function explain(id, lesson) {
    if (id === lastExplained) return;
    lastExplained = id;
    if (!byQuestion) {
      console.log(`[colab-mode] q${id}: the notebook index has not loaded yet`);
    } else if (!id) {
      console.log("[colab-mode] a question arrived with no id — cannot route it");
    } else if (lesson) {
      console.log(`[colab-mode] q${id} → ${lesson.id} (${lesson.file})`);
    } else {
      console.log(`[colab-mode] q${id}: no notebook for this question `
        + `(${Object.keys(byQuestion).length} in the index) — staying on the editor`);
    }
  }

  function colabNotebookHrefFor(q) {
    const lesson = colabLessonFor(q);
    const id = questionId(q);
    if (active) explain(id, lesson);
    if (!lesson || !repo || !id) return "";
    const path = [pathPrefix, lesson.file].filter(Boolean).join("/");
    const encoded = path.split("/").map(encodeURIComponent).join("/");
    return `https://colab.research.google.com/github/${owner()}/${repo}/blob/main/${encoded}`
      + `#scrollTo=dd-q${id}`;
  }

  function colabLessonFor(q) {
    if (!active || !byQuestion) return null;
    const id = questionId(q);
    return (id && byQuestion[id]) || null;
  }

  /** Run `fn` once the index has settled (immediately if it already has). */
  function whenColabIndexReady(fn) {
    if (typeof fn !== "function") return;
    if (!active || byQuestion) { fn(); return; }
    readyWaiters.push(fn);
  }

  /**
   * Put the notebook on screen for the learner, rather than a link to it.
   *
   * "It doesn't actually bring you to the Colab page" was the whole complaint:
   * the app rendered a card with an "Open in Colab ↗" anchor and waited to be
   * clicked, so every question cost a click and a fresh tab. The tutor knows
   * which notebook the next problem is in — it should just be there.
   *
   * It cannot do that alone. This page runs cross-origin inside the extension's
   * side panel, so `parent.location` is denied and `window.open` on a render
   * (no user gesture) is blocked. What it CAN do is ask its embedder, which is
   * an extension page holding `tabs` permission: `panel/app.js` receives this
   * and points the Colab tab at `url`. Nothing happens in a plain browser tab —
   * there is no second pane to steer — and the anchor stays visible for exactly
   * that case, and as the way back after wandering off.
   *
   * targetOrigin is "*" because the embedder is `chrome-extension://<id>` and
   * the id is not knowable from here. The payload is a public notebook URL and
   * carries nothing a listener could not already read off the page.
   */
  let lastOpened = "";
  function openNotebook(href, opts) {
    if (!active || !href) return false;
    if (window.parent === window) return false;
    if (href === lastOpened && !(opts && opts.force)) return false;
    try {
      window.parent.postMessage(
        { source: "delta-drills", type: "dd:open-notebook", url: href },
        "*",
      );
      lastOpened = href;
      console.log("[colab-mode] asked the panel to open", href);
      return true;
    } catch (err) {
      console.warn("[colab-mode] could not reach the panel:", err);
      return false;
    }
  }

  window.DDColab = {
    active: () => active,
    hrefFor: colabNotebookHrefFor,
    lessonFor: colabLessonFor,
    whenReady: whenColabIndexReady,
    openNotebook,
    framed: () => window.parent !== window,
    /** Everything the routing decision depends on, for the question on screen. */
    debug: () => {
      const q = (typeof practiceProgress === "object" && practiceProgress)
        ? practiceProgress.currentQuestion
        : null;
      return {
        edition: active,
        indexLoaded: Boolean(byQuestion),
        indexSize: byQuestion ? Object.keys(byQuestion).length : 0,
        questionId: questionId(q),
        lesson: (colabLessonFor(q) || {}).id || null,
        href: colabNotebookHrefFor(q),
        framed: window.parent !== window,
        lastOpened,
        routes: typeof torchNeedsColab === "function" ? torchNeedsColab(q) : "ui.js not loaded",
        isTorch: typeof questionIsTorch === "function" ? questionIsTorch(q) : "ui.js not loaded",
        practiceMode: typeof practiceMode === "undefined" ? "unset" : practiceMode,
      };
    },
  };

  /**
   * The card's copy is written for the normal app, where Colab is the fallback
   * for a runner that cannot import torch — "sign in to grade PyTorch drills
   * right here" is an instruction to LEAVE this route. On the Colab edition the
   * notebook is the point, so the same sentence reads as an apology for the
   * feature. Rewritten in place rather than forked into a second markup block,
   * which would then have to be kept in step with the first.
   */
  /**
   * Say which edition this is, on screen.
   *
   * The two deploys are byte-identical apart from the routing, so they look the
   * same in every way a person can check at a glance — same version tag, same
   * tabs, same everything. That cost an evening: a drill opened its editor on
   * the normal app and read as the Colab routing being broken. A badge and a
   * tab title are the whole fix.
   */
  function markEdition() {
    document.title = `${document.title.replace(/ — Colab edition$/, "")} — Colab edition`;
    const version = document.querySelector(".version-tag");
    if (!version || document.getElementById("dd-edition-badge")) return;
    const badge = document.createElement("span");
    badge.id = "dd-edition-badge";
    badge.textContent = "Colab edition";
    // Inline, because this file must stay droppable into either deploy without
    // a stylesheet following it around.
    badge.style.cssText = "margin-left:8px;padding:1px 7px;border-radius:999px;"
      + "background:rgba(255,127,80,0.18);border:1px solid rgba(255,127,80,0.5);"
      + "color:#ff9c78;font-size:11px;font-weight:600;vertical-align:middle;";
    version.insertAdjacentElement("afterend", badge);
  }

  function retitleNotice() {
    const notice = document.getElementById("torch-colab-notice");
    if (!notice) return;
    const title = notice.querySelector(".torch-colab-title");
    const body = notice.querySelector(".torch-colab-body");
    if (title) title.textContent = "🔦 Work this one in the notebook";
    if (body) {
      body.textContent = "This is the Colab edition — the drill is open in its"
        + " lesson notebook, at this problem. Run your solution there, compare"
        + " what it printed against the expected output, and say which it was.";
    }
    // The same two buttons, told the truth about what they mean HERE.
    //
    // They record a graded attempt — `_rateTorchAndAdvance(correct)` in
    // events.js posts a local eval and advances — but the normal app's wording
    // rates EFFORT ("worked through it" / "just skimmed it"), because there it
    // is a fallback for a runner that could not execute the drill at all. On
    // this deploy the notebook DID run it, so the learner has a real result to
    // report and the buttons are the submit: left is "my output matched",
    // right is "it didn't". Anything vaguer feeds the mastery model a guess.
    const yes = document.getElementById("torch-rate-solved");
    const no = document.getElementById("torch-rate-lookedup");
    // Both stay `.ghost`: they are two halves of one question, not an action
    // and its opt-out, and making the "matched" side primary put a thumb on the
    // scale of a self-report the mastery model then trusts.
    if (yes) {
      yes.textContent = "✓ My output matched";
      yes.title = "The notebook printed what the problem asked for — recorded as correct.";
    }
    if (no) {
      no.textContent = "✗ My output didn't match";
      no.title = "It ran, but the result was wrong or you had to read the answer — recorded as incorrect.";
    }
  }

  if (active) {
    document.documentElement.classList.add("dd-colab-edition");
    console.log("[colab-mode] Colab edition active on", location.hostname);
    const onReady = () => { markEdition(); retitleNotice(); };
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", onReady, { once: true });
    } else {
      onReady();
    }
  }
  load();
})();
