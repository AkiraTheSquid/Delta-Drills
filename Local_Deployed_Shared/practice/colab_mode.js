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
  function colabNotebookHrefFor(q) {
    const lesson = colabLessonFor(q);
    const id = questionId(q);
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

  window.DDColab = {
    active: () => active,
    hrefFor: colabNotebookHrefFor,
    lessonFor: colabLessonFor,
    whenReady: whenColabIndexReady,
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
      body.textContent = "This is the Colab edition — the drill lives in its lesson"
        + " notebook, opened at this problem. Work it through there, then tell us"
        + " how it went and the tutor picks the next one.";
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
