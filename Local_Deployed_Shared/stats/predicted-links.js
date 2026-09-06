/* ================================================================
   NOTEBOOK LINK HELPERS

   Resolves `colabUpstreamHref(notebookPath)` — the Google Colab URL
   for an ARENA or Delta-Drills notebook, pointed at the student's
   own fork when the Account tab's "GitHub username" is set.

   Consumers (all load AFTER this file in index.html):
     courses.js, courses-fork-gate.js, practice/ui.js,
     practice/drills-catalog.js, practice/arena-unlock.js,
     targeted-practice/targeted-practice.js

   NOTE: this file lives under stats/ for historical reasons — it was
   written for the Statistics tab's predicted-scores table, which was
   removed 2026-07-31. The table-only helpers (bookHrefForNotebook,
   vsCodeHrefFor, openLinkCell) went with it; only the Colab link
   resolution below is still live.
   ================================================================ */

// encodeURI leaves &, ?, # untouched — which is fine for browser nav to
// our own static HTML but breaks when the path is embedded inside a
// URL that downstream parsers (Colab, github.dev, VS Code) treat as
// query-bearing. Per-segment encodeURIComponent maps `&` to `%26`,
// `#` to `%23`, etc., without touching the `/` separators.
const encodePathSegments = (path) => String(path).split("/").map(encodeURIComponent).join("/");

// ARENA notebooks are opened from Seth's fork, AkiraTheSquid/ARENA_3.0, not
// from callummcdougall/ARENA_3.0 (the project kept the v3 repo name as the
// canonical home for all later versions — see ARENA's README). 2026-09-06:
// the fork carries notebooks upstream does not (0.1 with supplementary
// practice), the study group pulls the fork, and the in-app notebooks are
// compiled from it (scripts/compile_arena_notebooks.py) — the default owner
// here is what keeps the Colab link on the same file as the other two. A
// learner's own GitHub username still overrides it, as before.
const ARENA_UPSTREAM_OWNER = "AkiraTheSquid";
const ARENA_FORK_REPO = "ARENA_3.0";
// Procedural drills live in OUR repo (Delta-Drills), not ARENA's. Students
// fork Delta-Drills the same way they fork ARENA_3.0 — the same "GitHub
// username" account field switches both owners to their fork.
const DRILLS_UPSTREAM_OWNER = "AkiraTheSquid";
const DRILLS_REPO = "Delta-Drills";
const DRILLS_PATH_PREFIX = "arena-procedural-drills/";

// Account → "GitHub username" field. When set, every Colab URL points at
// the student's fork instead of upstream, so their saved notebook state
// (via Colab's File → Save a copy in GitHub) persists across visits.
// Falls back to upstream when empty.
const _accountGithubUsername = () => {
  try {
    return (localStorage.getItem("account_github_username") || "").trim();
  } catch (_) {
    return "";
  }
};
const arenaColabOwner = () => _accountGithubUsername() || ARENA_UPSTREAM_OWNER;
const drillsColabOwner = () => _accountGithubUsername() || DRILLS_UPSTREAM_OWNER;

const upstreamRelFor = (notebookPath) => {
  if (typeof notebookPath !== "string") return "";
  return notebookPath.replace(/^content\/ARENA_5\.0-main\//, "");
};

const colabUpstreamHref = (notebookPath) => {
  if (typeof notebookPath !== "string" || !notebookPath) return "";
  // Procedural drill notebooks live in Delta-Drills, not ARENA_3.0. Route them
  // to the student's Delta-Drills fork (falls back to upstream Delta-Drills).
  if (notebookPath.startsWith(DRILLS_PATH_PREFIX)) {
    return `https://colab.research.google.com/github/${drillsColabOwner()}/${DRILLS_REPO}/blob/main/${encodePathSegments(notebookPath)}`;
  }
  const rel = upstreamRelFor(notebookPath);
  if (!rel) return "";
  return `https://colab.research.google.com/github/${arenaColabOwner()}/${ARENA_FORK_REPO}/blob/main/${encodePathSegments(rel)}`;
};
