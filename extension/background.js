/* ================================================================
   BACKGROUND.JS — service worker.

   Deliberately almost empty. The panel is an extension page, so it can
   `fetch` the backend directly (host_permissions covers it) and can
   `chrome.tabs.sendMessage` the content script itself. The only thing that
   genuinely needs the worker is the toolbar-click → open-panel behaviour,
   which cannot be declared in the manifest.
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
