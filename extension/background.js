/* ================================================================
   BACKGROUND.JS — service worker.

   Two jobs, and the second one is the reason this file is no longer almost
   empty:

   1. The STUDY LAYOUT. Clicking the toolbar button tiles two real Chrome
      windows — the Colab edition of the app on the left half of the screen, a
      Colab notebook on the right half. Two windows rather than a side panel
      because Colab cannot be framed (it sends X-Frame-Options), and because
      Chrome's side panel is ~400px and its side is a USER setting an extension
      cannot touch. Tiled windows are the only arrangement this code can
      actually produce and verify.

   2. KEEPING Colab on the right. "Open in Colab ↗" in the app is an ordinary
      link, so Chrome opens it wherever it likes — usually a new tab beside the
      app, which lands the notebook on the LEFT and undoes the layout a second
      after it was made. Every tab that turns out to be Colab is therefore moved
      into the right-hand window.

   The side panel still exists (`side_panel.default_path`), reachable from
   Chrome's own side-panel menu. It just is not what the toolbar button does any
   more — `openPanelOnActionClick` has to be false for `action.onClicked` to
   fire at all, so this is one or the other, not both.
   ================================================================ */

// The Colab edition, not the normal app: on this deploy a drill routes to its
// lesson notebook instead of the in-page editor, which is the whole point of
// putting a notebook next to it. See Local_Deployed_Shared/practice/colab_mode.js.
const APP_URL = "https://delta-drills-colab.vercel.app/";

// Colab's GitHub browser for the published lesson notebooks — it lists all nine
// so the student picks one, which beats guessing a lesson the tutor has not
// chosen yet. Upstream owner: the extension holds no account state, and a
// student working in their own fork will already have that tab open, which the
// reuse path below prefers over this.
const COLAB_HOME = "https://colab.research.google.com/github/AkiraTheSquid/arena-book-colab";

const COLAB_MATCH = "https://colab.research.google.com/*";
const LAYOUT_KEY = "dd_layout";

chrome.runtime.onInstalled.addListener(() => {
  // False on purpose: with the default (true) Chrome swallows the click to open
  // the side panel and `action.onClicked` never fires.
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: false })
    .catch((err) => console.warn("[dd] setPanelBehavior failed", err));
});

chrome.runtime.onStartup.addListener(() => {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: false }).catch(() => {});
});

// ── Where the two windows go ─────────────────────────────────────────

/**
 * The usable screen rectangle — the display minus taskbars/docks.
 *
 * `system.display` is the only API that reports it. If it is unavailable the
 * focused window's own bounds are the fallback: not the work area, but the last
 * rectangle Chrome itself chose, which is close enough to tile against and far
 * better than a hardcoded 1920×1080 on a laptop.
 */
async function workArea() {
  try {
    const displays = await chrome.system.display.getInfo();
    const primary = displays.find((d) => d.isPrimary) || displays[0];
    if (primary && primary.workArea) return primary.workArea;
  } catch (err) {
    console.warn("[dd] system.display unavailable, tiling against the focused window", err);
  }
  const win = await chrome.windows.getLastFocused().catch(() => null);
  if (win && Number.isFinite(win.width)) {
    return { left: win.left || 0, top: win.top || 0, width: win.width, height: win.height };
  }
  return { left: 0, top: 0, width: 1440, height: 900 };
}

function halves(area) {
  const leftWidth = Math.floor(area.width / 2);
  return {
    left: { left: area.left, top: area.top, width: leftWidth, height: area.height },
    right: {
      left: area.left + leftWidth,
      top: area.top,
      width: area.width - leftWidth,
      height: area.height,
    },
  };
}

// ── Remembering the pair ─────────────────────────────────────────────
// Session storage, not local: a window id is meaningless once the browser has
// restarted, and a stale id read back after one would repoint the layout at
// whatever window happened to inherit the number.

async function readLayout() {
  const stored = await chrome.storage.session.get(LAYOUT_KEY);
  return stored[LAYOUT_KEY] || {};
}

async function writeLayout(patch) {
  const next = { ...(await readLayout()), ...patch };
  await chrome.storage.session.set({ [LAYOUT_KEY]: next });
  return next;
}

/** The window under this id, or null if it has since been closed. */
async function liveWindow(id) {
  if (!Number.isFinite(id)) return null;
  return chrome.windows.get(id).catch(() => null);
}

// ── Building the layout ──────────────────────────────────────────────

/**
 * The app pane. A `popup` window: no tab strip and no omnibox, so it reads as an
 * app rather than a browser someone left open. Nothing is ever moved INTO it —
 * Chrome refuses to move tabs into a popup — which is fine, because the only
 * thing that belongs here is the app.
 */
async function ensureAppWindow(bounds) {
  const layout = await readLayout();
  const existing = await liveWindow(layout.appWindowId);
  if (existing) {
    await chrome.windows.update(existing.id, { ...bounds, state: "normal", focused: true });
    return existing.id;
  }
  const created = await chrome.windows.create({ url: APP_URL, type: "popup", ...bounds });
  await writeLayout({ appWindowId: created.id });
  return created.id;
}

/**
 * The notebook pane. A NORMAL window, because Colab tabs get moved into it and
 * a popup cannot receive them.
 *
 * An already-open Colab tab is preferred over a fresh one: the student may be
 * mid-notebook, and opening the picker over the top of that would lose their
 * place. Only when there is none does the picker open.
 */
async function ensureColabWindow(bounds) {
  const layout = await readLayout();
  const existing = await liveWindow(layout.colabWindowId);
  if (existing) {
    await chrome.windows.update(existing.id, { ...bounds, state: "normal" });
    return existing.id;
  }

  const openColab = await chrome.tabs.query({ url: COLAB_MATCH });
  if (openColab.length) {
    const tab = openColab[0];
    const host = await liveWindow(tab.windowId);
    // Its own window already, and nothing else in it → just move that window.
    if (host && host.type === "normal") {
      const tabsInHost = await chrome.tabs.query({ windowId: tab.windowId });
      if (tabsInHost.length === 1) {
        await chrome.windows.update(host.id, { ...bounds, state: "normal" });
        await writeLayout({ colabWindowId: host.id });
        return host.id;
      }
    }
    // Otherwise pull the tab out into a window of its own on the right.
    const created = await chrome.windows.create({ tabId: tab.id, ...bounds });
    await writeLayout({ colabWindowId: created.id });
    return created.id;
  }

  const created = await chrome.windows.create({ url: COLAB_HOME, ...bounds });
  await writeLayout({ colabWindowId: created.id });
  return created.id;
}

async function openStudyLayout() {
  const area = await workArea();
  const { left, right } = halves(area);
  // Right first, then left: the app takes focus last, because that is where the
  // student reads the next problem.
  await ensureColabWindow(right);
  await ensureAppWindow(left);
}

chrome.action.onClicked.addListener(() => {
  openStudyLayout().catch((err) => console.warn("[dd] study layout failed", err));
});

// ── Keeping Colab on the right ───────────────────────────────────────

function isColabUrl(url) {
  return typeof url === "string" && url.startsWith("https://colab.research.google.com/");
}

/**
 * Move a stray Colab tab into the notebook window.
 *
 * Guarded three ways, because this listener sees every tab in the browser:
 *   - the layout must be live (a `colabWindowId` that still resolves), so this
 *     does nothing at all for anyone not using the study layout;
 *   - the tab must already be somewhere else;
 *   - a failure is swallowed. A tab can be dragged, closed or navigated between
 *     the check and the move, and none of that is worth an error.
 */
async function captureColabTab(tab) {
  if (!tab || !isColabUrl(tab.url)) return;
  const layout = await readLayout();
  const target = await liveWindow(layout.colabWindowId);
  if (!target || tab.windowId === target.id) return;
  try {
    await chrome.tabs.move(tab.id, { windowId: target.id, index: -1 });
    await chrome.tabs.update(tab.id, { active: true });
  } catch (err) {
    console.warn("[dd] could not move a Colab tab into the notebook window", err);
  }
}

// A tab created by a link click usually has no URL yet, so onUpdated is the one
// that actually fires with an address. onCreated covers the case where it does.
chrome.tabs.onCreated.addListener((tab) => { void captureColabTab(tab); });
chrome.tabs.onUpdated.addListener((_id, changed, tab) => {
  if (changed.url) void captureColabTab(tab);
});

// Forget a pane the student closed, so the next click rebuilds it rather than
// updating an id that no longer exists.
chrome.windows.onRemoved.addListener(async (windowId) => {
  const layout = await readLayout();
  const patch = {};
  if (layout.appWindowId === windowId) patch.appWindowId = null;
  if (layout.colabWindowId === windowId) patch.colabWindowId = null;
  if (Object.keys(patch).length) await writeLayout(patch);
});
