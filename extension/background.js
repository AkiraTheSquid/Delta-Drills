/* ================================================================
   BACKGROUND.JS — service worker.

   Deliberately almost empty. The panel is an extension page, so it can
   `fetch` the backend directly (host_permissions covers it) and can
   `chrome.tabs.sendMessage` the content script itself. The only thing that
   genuinely needs the worker is the toolbar-click → open-panel behaviour,
   which cannot be declared in the manifest.

   ⚠️ DO NOT MAKE THE BUTTON OPEN WINDOWS. Tried on 2026-07-31: the toolbar
   click tiled two `chrome.windows` (app left, Colab right) to get "app on the
   left". It works and it is the wrong thing — separate windows are not a side
   pane, and they pop out of the browser instead of splitting it. The side panel
   IS the design.

   The panel's SIDE is a Chrome setting, not an extension one. There is no API
   for it: `chrome.sidePanel` exposes `setOptions` and `setPanelBehavior` and
   nothing about position or width. The user moves it with
   Settings → Appearance → "Side panel position: Show on left" (or by
   right-clicking the panel), and widens it by dragging its inner edge. Anything
   here claiming to do that for them is a workaround for a setting that already
   exists.
   ================================================================ */

chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch((err) => console.warn("[dd] setPanelBehavior failed", err));
});

// The worker is torn down and restarted freely; re-assert on every startup so
// the behaviour survives a browser restart without a reinstall.
chrome.runtime.onStartup.addListener(() => {
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch(() => {});
});
