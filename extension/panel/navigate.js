/* ================================================================
   NAVIGATE.JS — getting the student to the right cell, in the right notebook.

   Why this is its own file: the tutor selects weakest-first across every
   subtopic and one lesson is one subtopic, so two consecutive problems land in
   two different notebooks as a matter of routine. "Scroll to a cell" is
   therefore never just a scroll — it is resolve the notebook, switch to it if
   the open tab is a different one, wait for Colab to finish mounting, and only
   then jump. That is a job with its own failure modes, and keeping it next to
   the view state machine buried the difference.

   Exposed as `window.DDNav`. Loads after api.js (it uses `notebooks` and
   `tab`) and before panel.js.
   ================================================================ */

const { tab: navTab, notebooks: nb } = window.DD;

const el = (id) => document.getElementById(id);

// Mirrors slug() in scripts/generate_colab_notebooks.py, which is what minted
// the `dd-kp-…` anchors. A KC with a dot in it would otherwise never resolve.
const slugKc = (kc) =>
  String(kc)
    .replace(/[^a-zA-Z0-9\-_]/g, "-")
    .slice(0, 64)
    .replace(/^-+|-+$/g, "");

/** Ask the open tab who it is, and remember the URL if it tells us. */
async function whoIsOpen() {
  const res = await navTab.send({ type: "dd:identify" });
  if (res.ok && res.lessonId && res.url) await nb.learn(res.lessonId, res.url);
  return res;
}

/** The "📓 <lesson> · open / switch" row, in either the problem or gate view. */
function paintNotebook(prefix, target, open) {
  const title = el(`${prefix}-nb-title`);
  const badge = el(`${prefix}-nb-state`);
  if (!target) {
    title.textContent = "Notebook not in the index";
    badge.textContent = "";
    badge.className = "nb-state";
    return;
  }
  title.textContent = target.title;
  if (open && open.ok && open.lessonId === target.id) {
    badge.textContent = "open";
    badge.className = "nb-state here";
  } else if (open && open.ok && open.lessonId) {
    badge.textContent = "switch";
    badge.className = "nb-state away";
  } else {
    badge.textContent = "not open";
    badge.className = "nb-state away";
  }
}

/**
 * Make sure the notebook holding `target` is the one on screen.
 *
 * Returns `{ok, switched}` or a reason. A target we cannot name a URL for is
 * the one genuinely unrecoverable case, and it is a configuration problem
 * rather than a runtime one — hence its own reason rather than a generic fail.
 */
async function ensureNotebook(target, status) {
  // No index entry: nothing to switch to, so let the jump try the open tab and
  // report honestly if the cell is not there.
  if (!target) return { ok: true, switched: false, open: null };

  const open = await whoIsOpen();
  if (open.ok && open.lessonId === target.id) return { ok: true, switched: false, open };

  const url = nb.urlFor(target.id);
  if (!url) return { ok: false, reason: "no-url", target };

  status.textContent = `Opening ${target.title}…`;
  status.className = "status small";

  const tabId = await navTab.navigate(url);
  const landed = await navTab.waitForNotebook(tabId, target.id);
  if (!landed) return { ok: false, reason: "timeout", target };
  if (landed.lessonId && landed.lessonId !== target.id) {
    return { ok: false, reason: "wrong-notebook", target, got: landed.lessonId };
  }
  await nb.learn(target.id, landed.url);
  return { ok: true, switched: true, open: landed };
}

function reportSwitchProblem(status, r) {
  status.className = "status small err";
  if (r.reason === "no-url") {
    status.textContent = `No link for "${r.target.title}" yet — set your notebook repo in ⚙, or open that notebook once and the panel will remember it.`;
  } else if (r.reason === "timeout") {
    status.textContent = `${r.target.title} is still loading. Give Colab a moment, then try again.`;
  } else if (r.reason === "wrong-notebook") {
    status.textContent = `That link opened ${r.got}, not ${r.target.title}. Check the notebook repo in ⚙.`;
  } else {
    status.textContent = "Couldn't get to that notebook.";
  }
}

/** Jump to a cell, switching notebooks first when the target lives elsewhere. */
async function jumpTo({ target, anchor, text, status, arrived }) {
  status.textContent = "Finding the cell…";
  status.className = "status small";

  const ready = await ensureNotebook(target, status);
  if (!ready.ok) {
    reportSwitchProblem(status, ready);
    return ready;
  }

  await navTab.focus();
  const res = await navTab.send({ type: "dd:goto", anchor, text });
  const prefix = ready.switched ? `Switched to ${target.title}. ` : "";

  if (res.ok && res.visible) {
    status.textContent =
      prefix +
      (res.expanded
        ? `Opened ${res.expanded} collapsed section${res.expanded === 1 ? "" : "s"} to get there.`
        : arrived);
    status.className = "status small ok";
  } else if (res.reason === "no-colab-tab") {
    status.textContent = "Open the notebook in a tab first.";
    status.className = "status small err";
  } else if (res.reason === "no-receiver") {
    status.textContent = "Reload the Colab tab — it loaded before the extension did.";
    status.className = "status small err";
  } else if (res.reason === "not-found") {
    status.textContent = target
      ? `${prefix}That cell isn't in ${target.title}. Regenerate the notebooks?`
      : "That cell isn't in this notebook. Wrong notebook open?";
    status.className = "status small err";
  } else {
    status.textContent = prefix + "Scrolled, but couldn't confirm it landed.";
    status.className = "status small err";
  }
  return { ...ready, res };
}

/**
 * Every lesson, and whether the panel can currently open it.
 *
 * This is the thing to look at when a switch fails: "remembered" means a URL
 * proven to work, "repo" means one computed from the owner/repo setting and
 * not yet tried, and "no link" means the panel has no way in at all.
 */
function renderNotebookList() {
  const host = el("nb-list");
  host.textContent = "";
  const lessons = nb.index.lessons || [];
  const known = lessons.filter((l) => nb.urlFor(l.id)).length;
  el("nb-summary").textContent = `Notebooks — ${known}/${lessons.length} openable`;

  lessons.forEach((l) => {
    const url = nb.urlFor(l.id);
    const remembered = Boolean(nb.urls[l.id]);
    const li = document.createElement("li");
    li.className = url ? "nb-item" : "nb-item off";

    const name = document.createElement("span");
    name.className = "nb-name";
    name.textContent = l.title;

    const how = document.createElement("span");
    how.className = "nb-how";
    how.textContent = remembered ? "remembered" : url ? "repo" : "no link";

    li.append(name, how);
    if (url) {
      li.title = `Open ${l.file}`;
      li.onclick = () => navTab.navigate(url);
    }
    host.appendChild(li);
  });
}

window.DDNav = {
  slugKc,
  whoIsOpen,
  paintNotebook,
  ensureNotebook,
  jumpTo,
  renderNotebookList,
};
