/* ================================================================
   APP.JS — the side panel that IS the Delta Drills web app.

   The panel hosts the deployed app in an iframe rather than reimplementing it.
   That is the whole design decision: the app already has the practice loop, the
   lesson pages, the knowledge graph, the courses and the stats, and every one of
   them is a feature this panel would otherwise have to grow a second time and
   then keep in sync. Ship the app; the extension supplies the surface.

   Why an iframe and not `side_panel.default_path: "https://…"` — MV3 only
   accepts an extension-relative path there, so a local page holding a frame is
   the only route to a remote UI.

   What this file is allowed to know: the address of the app, whether it loaded,
   and how to get out of the panel into a full tab. It does not reach into the
   frame. The app is a different origin, so it could not anyway, and the moment
   this file starts scripting the app's DOM the two become one codebase again.
   ================================================================ */

const DEFAULT_APP = "https://delta-drills.vercel.app/";

// How long Colab-grade cold starts get before we call it a failure. Generous on
// purpose: a fallback screen shown over an app that was merely slow is worse
// than a spinner that runs a few seconds long.
const LOAD_TIMEOUT_MS = 20000;

const $ = (id) => document.getElementById(id);

const state = {
  url: DEFAULT_APP,
  compact: false,
  timer: null,
};

/* ── the address ─────────────────────────────────────────────── */

/** Accepts what a human types: bare hosts, missing scheme, stray whitespace. */
function normalize(raw) {
  const s = String(raw || "").trim();
  if (!s) return DEFAULT_APP;
  const withScheme = /^https?:\/\//i.test(s) ? s : `https://${s}`;
  try {
    return new URL(withScheme).href;
  } catch (_) {
    return DEFAULT_APP;
  }
}

/**
 * The URL to actually put in the frame.
 *
 * `?embed=1` is the app's own flag — it hides the tab bar and banners for the
 * knowledge-graph practice overlay. In a 400px panel that reads as a focused
 * practice view, but it also removes the navigation, so it stays opt-in.
 */
function srcFor() {
  const u = new URL(state.url);
  if (state.compact) u.searchParams.set("embed", "1");
  else u.searchParams.delete("embed");
  return u.href;
}

function label() {
  try {
    $("where").textContent = new URL(state.url).host;
  } catch (_) {
    $("where").textContent = "Delta Drills";
  }
}

/* ── loading ─────────────────────────────────────────────────── */

function stopWaiting() {
  if (state.timer) clearTimeout(state.timer);
  state.timer = null;
  $("spinner").classList.remove("on");
}

function showBlocked(why) {
  stopWaiting();
  $("blocked-why").textContent = why;
  $("blocked").classList.add("on");
}

/**
 * Point the frame at the app.
 *
 * The `about:blank` hop is not superstition: assigning the identical URL to
 * `src` is a no-op in Chrome, so a plain reload would do nothing. Going via a
 * blank document forces the load and also clears the previous page before the
 * spinner covers it.
 */
function loadApp(force) {
  const frame = $("app");
  const next = srcFor();
  $("blocked").classList.remove("on");
  $("spinner").classList.add("on");

  if (state.timer) clearTimeout(state.timer);
  state.timer = setTimeout(() => {
    showBlocked(
      "It's been 20 seconds with no response. The app may be down, the address may be wrong, or Chrome may be refusing to embed it.",
    );
  }, LOAD_TIMEOUT_MS);

  if (force && frame.src && frame.src !== "about:blank") {
    frame.src = "about:blank";
    // One turn of the event loop so the blank document commits first.
    setTimeout(() => {
      frame.src = next;
    }, 0);
  } else {
    frame.src = next;
  }
  label();
}

// A cross-origin frame tells us almost nothing, but `load` firing at all means
// Chrome committed a document rather than refusing the frame outright.
$("app").addEventListener("load", () => {
  if ($("app").src && $("app").src !== "about:blank") stopWaiting();
});

/* ── settings ────────────────────────────────────────────────── */

function toggleSettings(on) {
  const sheet = $("settings");
  const show = on === undefined ? !sheet.classList.contains("on") : on;
  sheet.classList.toggle("on", show);
  $("btn-settings").classList.toggle("on", show);
  if (show) {
    $("in-url").value = state.url;
    $("in-compact").checked = state.compact;
    $("settings-status").textContent = "";
    $("settings-status").className = "status small";
  }
}

async function save() {
  const url = normalize($("in-url").value);
  const compact = $("in-compact").checked;
  const changed = url !== state.url || compact !== state.compact;
  state.url = url;
  state.compact = compact;
  await chrome.storage.local.set({ dd_app_url: url, dd_app_compact: compact });
  $("in-url").value = url;
  $("settings-status").textContent = changed ? "Saved." : "No change.";
  $("settings-status").className = "status small ok";
  toggleSettings(false);
  loadApp(true);
}

/* ── wiring ──────────────────────────────────────────────────── */

$("btn-home").onclick = () => {
  toggleSettings(false);
  loadApp(true);
};
$("btn-reload").onclick = () => {
  toggleSettings(false);
  loadApp(true);
};
$("btn-settings").onclick = () => toggleSettings();
$("btn-save").onclick = save;

$("btn-reset").onclick = async () => {
  $("in-url").value = DEFAULT_APP;
  $("in-compact").checked = false;
  await save();
};

const inTab = () => chrome.tabs.create({ url: srcFor(), active: true });
$("btn-tab").onclick = inTab;
$("btn-blocked-tab").onclick = inTab;
$("btn-retry").onclick = () => loadApp(true);

// Swaps the panel to the hand-built Colab tutor UI. Close and reopen the panel
// to come back — the manifest's default_path is this page.
$("btn-native").onclick = () => {
  location.href = "panel.html";
};

$("in-url").addEventListener("keydown", (e) => {
  if (e.key === "Enter") save();
});

/* ── boot ────────────────────────────────────────────────────── */

(async function boot() {
  const { dd_app_url, dd_app_compact } = await chrome.storage.local.get([
    "dd_app_url",
    "dd_app_compact",
  ]);
  state.url = normalize(dd_app_url || DEFAULT_APP);
  state.compact = Boolean(dd_app_compact);
  loadApp(false);
})();
