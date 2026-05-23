/* ================================================================
   STATS PREDICTED — link helpers

   Builds the four launch URLs surfaced in the predicted-scores
   table Open column:
     - Jupyter Book static HTML (bookHrefForNotebook)
     - Upstream ARENA notebook on Google Colab (colabUpstreamHref)
     - Local repo notebook in VS Code (vsCodeHrefFor)
     - Generic single-link <td> wrapper used by chapter/section/
       problem rows (openLinkCell)

   Shared across stats/predicted.js — loaded BEFORE predicted.js
   in index.html so all of these are defined at render time.
   ================================================================ */

// encodeURI leaves &, ?, # untouched — which is fine for browser nav to
// our own static HTML but breaks when the path is embedded inside a
// URL that downstream parsers (Colab, github.dev, VS Code) treat as
// query-bearing. Per-segment encodeURIComponent maps `&` to `%26`,
// `#` to `%23`, etc., without touching the `/` separators.
const encodePathSegments = (path) => String(path).split("/").map(encodeURIComponent).join("/");

const bookHrefForNotebook = (notebookPath) => {
  if (typeof notebookPath !== "string") return "";
  const remapped = notebookPath
    .replace(/^content\/ARENA_5\.0-main\//, "arena-book/")
    .replace(/\.ipynb$/, ".html");
  return encodePathSegments(remapped);
};

// Upstream ARENA notebooks are hosted at callummcdougall/ARENA_3.0 (the
// project kept the v3 repo name as the canonical home for all later
// versions — see ARENA's README). The path inside their repo matches our
// `content/ARENA_5.0-main/<rest>` layout once that prefix is stripped.
const ARENA_UPSTREAM_OWNER = "callummcdougall";
const ARENA_FORK_REPO = "ARENA_3.0";
const VSCODE_LOCAL_ABS_ROOT = "/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared";

// Account → "GitHub username" field. When set, every Colab URL points at
// the student's fork instead of upstream, so their saved notebook state
// (via Colab's File → Save a copy in GitHub) persists across visits.
// Falls back to upstream when empty.
const arenaColabOwner = () => {
  try {
    const u = (localStorage.getItem("account_github_username") || "").trim();
    return u || ARENA_UPSTREAM_OWNER;
  } catch (_) {
    return ARENA_UPSTREAM_OWNER;
  }
};

const upstreamRelFor = (notebookPath) => {
  if (typeof notebookPath !== "string") return "";
  return notebookPath.replace(/^content\/ARENA_5\.0-main\//, "");
};

const colabUpstreamHref = (notebookPath) => {
  const rel = upstreamRelFor(notebookPath);
  if (!rel) return "";
  return `https://colab.research.google.com/github/${arenaColabOwner()}/${ARENA_FORK_REPO}/blob/main/${encodePathSegments(rel)}`;
};

const vsCodeHrefFor = (notebookPath) => {
  if (typeof notebookPath !== "string") return "";
  return `vscode://file/${encodePathSegments(`${VSCODE_LOCAL_ABS_ROOT}/${notebookPath}`)}`;
};

const openLinkCell = (href, title, label = "Open ↗") => {
  if (!href) return `<td class="stats-col-open"></td>`;
  const safeTitle = (title || "").replace(/"/g, "&quot;");
  return `<td class="stats-col-open"><a class="stats-open-link" href="${href}" target="_blank" rel="noreferrer" title="${safeTitle}">${label}</a></td>`;
};
