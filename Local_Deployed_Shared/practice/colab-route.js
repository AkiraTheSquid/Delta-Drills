/* ================================================================
   COLAB ROUTE — send the learner to the cell this problem lives in
   ================================================================

   Practice no longer runs code. The learner runs it in Colab and reports back
   whether it ran, so "which notebook, which cell" stopped being a convenience
   link and became the step the whole loop depends on. This module owns it.

   The map is GENERATED. `scripts/generate_colab_notebooks.py` compiles the
   nine notebooks and writes `lessons/colab_notebooks.json` in the same pass,
   so the map and the notebooks can never describe different things. Never edit
   that file by hand — regenerate it.

   WHY A NAMED WINDOW. `window.open(url, TARGET)` with a constant name reuses
   ONE tab for every jump instead of opening a new one per problem. That is
   what makes this work "regardless of what tab you're on": the learner keeps a
   single Colab tab and the app steers it. When consecutive problems live in the
   same notebook the two URLs differ only in the fragment, which is a
   same-document navigation — the notebook does not reload and the learner's
   kernel state and edits survive.

   WHY THE LINK IS STILL VISIBLE. Browsers only allow `window.open` during a
   user gesture. Clicks (Start, Next, Skip) qualify; a timer that auto-advances
   does NOT, and neither does a popup blocker that has decided otherwise. The
   auto-open is therefore best-effort, and `#colab-open-link` is always
   rendered as a real anchor so there is a working way through when it fails.
   Do not "simplify" that away by trusting the popup.
*/

(function () {
  "use strict";

  // One tab, reused. Changing this string starts opening a second tab.
  var TARGET = "delta-drills-colab";

  var INDEX_URL = "lessons/colab_notebooks.json";

  // Where the generated notebooks are published. `publish_colab_notebooks.sh`
  // pushes them to <owner>/<repo>/ARENA_5.0/ch-1-foundations, and prints this
  // exact string when it finishes. The stored override exists so a fork can
  // point at its own copy without a redeploy.
  var DEFAULT_REPO = "AkiraTheSquid/arena-book-colab@main/ARENA_5.0/ch-1-foundations";
  var REPO_KEY = "delta_drills_colab_repo";

  var indexPromise = null;
  var index = null;

  function load() {
    if (!indexPromise) {
      indexPromise = fetch(INDEX_URL, { cache: "no-cache" })
        .then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        })
        .then(function (data) {
          index = data;
          return data;
        })
        .catch(function (err) {
          // Fails OPEN, like the q-matrix filter next door: no map means no
          // Colab link, not a broken practice page.
          console.warn("[colab] notebook map unavailable:", err.message);
          index = null;
          return null;
        });
    }
    return indexPromise;
  }

  function repo() {
    try {
      return localStorage.getItem(REPO_KEY) || DEFAULT_REPO;
    } catch (_) {
      return DEFAULT_REPO;
    }
  }

  function setRepo(value) {
    try {
      if (value) localStorage.setItem(REPO_KEY, value);
      else localStorage.removeItem(REPO_KEY);
    } catch (_) {
      /* private mode — the default still works */
    }
  }

  /* "owner/repo@branch/path" -> its three parts. Branch is optional and
     defaults to main, matching what the publish script creates. */
  function parseRepo(spec) {
    var at = spec.indexOf("@");
    var branch = "main";
    var rest = spec;
    if (at >= 0) {
      var tail = spec.slice(at + 1);
      var slash = tail.indexOf("/");
      branch = slash >= 0 ? tail.slice(0, slash) : tail;
      rest = spec.slice(0, at) + (slash >= 0 ? tail.slice(slash) : "");
    }
    var parts = rest.split("/").filter(Boolean);
    if (parts.length < 2) return null;
    return {
      owner: parts[0],
      name: parts[1],
      branch: branch,
      path: parts.slice(2).join("/"),
    };
  }

  function questionId(q) {
    var raw = q && (q.question_id != null ? q.question_id : q.id);
    return raw == null ? "" : String(raw);
  }

  /* Three lookups, in descending order of certainty, because /next-question
     does not always hand back the same key. `questions` is exact; `subtopics`
     still lands on the right notebook for a bank question added after the map
     was generated; `kcs` is what a lesson gate resolves through. */
  /* Was this question found by its own id, or only inferred?
     Only an exact hit proves the notebook contains a `dd-q<id>` cell — the map
     and the cells come from one generator pass. A subtopic/KC fallback names
     the right LESSON but says nothing about this particular question, so the
     anchor must be dropped for those (see urlFor). */
  function isExactMatch(q) {
    return !!(index && index.questions && index.questions[questionId(q)]);
  }

  function lessonIdFor(q) {
    if (!index || !q) return "";
    var byId = index.questions && index.questions[questionId(q)];
    if (byId) return byId;

    var subs = index.subtopics || {};
    // Subtopic naming differs by mode: the backend sends the composite
    // ("Numpy: Indexing and selection"), the local bank the bare name with the
    // composite in `subtopic_key`. Accept either — see practice/README.md.
    var candidates = [q.subtopic_key, q.subtopic];
    if (q.topic && q.subtopic) candidates.push(q.topic + ": " + q.subtopic);
    for (var i = 0; i < candidates.length; i++) {
      if (candidates[i] && subs[candidates[i]]) return subs[candidates[i]];
    }

    var kcs = index.kcs || {};
    var targets = [].concat(q.target_kcs || [], q.ladder_kc || []);
    for (var j = 0; j < targets.length; j++) {
      if (targets[j] && kcs[targets[j]]) return kcs[targets[j]];
    }
    return "";
  }

  function lessonById(id) {
    if (!id || !index || !Array.isArray(index.lessons)) return null;
    for (var i = 0; i < index.lessons.length; i++) {
      if (index.lessons[i].id === id) return index.lessons[i];
    }
    return null;
  }

  function lessonFor(q) {
    return lessonById(lessonIdFor(q));
  }

  /* The notebook URL for a lesson file, with an optional cell anchor already
     encoded. Both the problem route and the concept route end here, so there
     is one place that knows the Colab URL shape. */
  function notebookUrl(lesson, anchor) {
    if (!lesson) return "";
    var r = parseRepo(repo());
    if (!r) return "";
    var path = (r.path ? r.path + "/" : "") + lesson.file;
    var url =
      "https://colab.research.google.com/github/" +
      encodeURIComponent(r.owner) + "/" + encodeURIComponent(r.name) +
      "/blob/" + encodeURIComponent(r.branch) + "/" +
      path.split("/").map(encodeURIComponent).join("/");
    return anchor ? url + "#scrollTo=" + encodeURIComponent(anchor) : url;
  }

  /* The cell anchor. `metadata.id` on every generated cell is `dd-q<id>`, which
     is exactly what Colab's `#scrollTo=` matches — that is the whole reason the
     generator emits nbformat 4.5 instead of copying ARENA's 4.2 notebooks,
     which have no stable ids and cannot be linked into. */
  function urlFor(q) {
    var url = notebookUrl(lessonFor(q), null);
    if (!url) return "";
    // Anchor ONLY on an exact question -> notebook hit. An inferred lesson
    // (matched by subtopic or KC) is very likely the right notebook but carries
    // no promise that THIS question has a cell in it, and a `#scrollTo=` naming
    // a cell that does not exist is worse than none: Colab silently ignores it,
    // so the learner lands at the top of a ~500-cell notebook believing they
    // were taken to their problem. Without the anchor they at least know they
    // have to look.
    var qid = questionId(q);
    return qid && isExactMatch(q)
      ? url + "#scrollTo=dd-q" + encodeURIComponent(qid)
      : url;
  }

  /* ---------- concepts ------------------------------------------------- */

  /* Where a CONCEPT is taught, as opposed to where a problem is worked.

     The lesson gate used to render the concept and its worked example inside
     the panel. It does not any more: a worked example that is split across
     several code cells only behaves like a worked example in a notebook, where
     each cell runs against the state the one above it left behind. So the gate
     sends the learner to the KP section instead, and this is the address.

     `index.kps` is generated alongside the notebooks, so the anchor here and
     the cell id there are the same string by construction. */
  function kpFor(kc) {
    if (!index || !index.kps || !kc) return null;
    return index.kps[kc] || null;
  }

  function urlForKc(kc) {
    var kp = kpFor(kc);
    if (!kp) return "";
    return notebookUrl(lessonById(kp.lesson), kp.anchor);
  }

  function lessonForKc(kc) {
    var kp = kpFor(kc);
    return kp ? lessonById(kp.lesson) : null;
  }

  /* ---------- opening -------------------------------------------------- */

  function openUrl(url) {
    if (!url) return false;
    var win;
    try {
      win = window.open(url, TARGET);
    } catch (_) {
      return false;
    }
    if (!win) return false;
    // Bring it forward when allowed; a blocked focus() is not a failure.
    try {
      win.focus();
    } catch (_) {
      /* cross-origin, or the browser declined — the tab is still correct */
    }
    return true;
  }

  /* Best-effort. Returns true only when the browser actually gave us a window,
     so the caller can say so plainly instead of pretending it worked. */
  function open(q) {
    return openUrl(urlFor(q));
  }

  function openKc(kc) {
    return openUrl(urlForKc(kc));
  }

  window.ColabRoute = {
    load: load,
    lessonFor: lessonFor,
    lessonForKc: lessonForKc,
    urlFor: urlFor,
    urlForKc: urlForKc,
    open: open,
    openKc: openKc,
    repo: repo,
    setRepo: setRepo,
    TARGET: TARGET,
    DEFAULT_REPO: DEFAULT_REPO,
  };
})();
