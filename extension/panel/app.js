/* ================================================================
   APP.JS — the one thing the framed site cannot do for itself.

   The side panel is the live Colab edition in an iframe (see app.html). That
   framing is deliberate and this file is not the beginning of a second front
   end: it renders nothing, owns no state and touches no DOM. It exists because
   of a single hard boundary.

   The site knows which notebook the next problem is in — `practice/colab_mode.js`
   resolves question id → lesson notebook → `#scrollTo=dd-q<id>`. It cannot open
   it. It is cross-origin inside this frame, so `parent.location` is denied, and
   a question renders without a user gesture, so `window.open` is blocked as a
   popup. The result was the thing Seth reported: "it doesn't actually bring you
   to the Google Collaboratory page… it just shows you the problem on the pane
   itself" — a card with a link on it, once per question.

   This page is an extension page and holds `tabs`, so it can. The site posts
   `{source:"delta-drills", type:"dd:open-notebook", url}` and the Colab tab goes
   there. That is the entire contract.

   ⚠️ The origin check is the security boundary. `window.message` is reachable
   by any frame, and this listener drives tab navigation, so an unchecked
   handler is an open redirect wearing an extension's permissions. Both halves
   matter: the sender must BE the framed site (`event.source`), and it must be
   ON the expected origin.
   ================================================================ */

const APP_ORIGIN = "https://delta-drills-colab.vercel.app";
const COLAB_MATCH = "https://colab.research.google.com/*";

/**
 * Reuse the open notebook tab rather than piling up tabs.
 *
 * A session crosses notebooks constantly — the tutor picks weakest-first across
 * every subtopic and one lesson is one subtopic — so a tab per switch would
 * leave a dozen stale kernels behind within the hour. Same reasoning, and same
 * behaviour, as `tab.navigate` in api.js; duplicated rather than imported
 * because that file pulls in the whole panel stack and this page loads one
 * script on purpose.
 */
async function colabTab() {
  const tabs = await chrome.tabs.query({ url: COLAB_MATCH });
  if (!tabs.length) return null;
  return tabs.find((t) => t.active) || tabs[0];
}

// Two questions in the same notebook differ only by fragment, and re-issuing an
// identical URL makes Chrome RELOAD the tab — which drops the kernel and any
// work in progress. Remembering the last one turns that into a no-op.
let lastUrl = "";

async function openNotebook(url) {
  if (!url || url === lastUrl) return;
  lastUrl = url;
  const t = await colabTab();
  if (t) {
    await chrome.tabs.update(t.id, { url, active: true });
    return;
  }
  await chrome.tabs.create({ url, active: true });
}

window.addEventListener("message", (event) => {
  if (event.origin !== APP_ORIGIN) return;
  const frame = document.querySelector("iframe");
  if (!frame || event.source !== frame.contentWindow) return;
  const msg = event.data;
  if (!msg || msg.source !== "delta-drills" || msg.type !== "dd:open-notebook") return;
  // Never navigate to anything but a notebook. The site only ever sends Colab
  // URLs; this makes a compromised or mistaken sender unable to send anything
  // else, which is the difference between a bridge and an open redirect.
  const url = String(msg.url || "");
  if (!/^https:\/\/colab\.research\.google\.com\//.test(url)) {
    console.warn("[dd] refused a non-Colab navigation:", url);
    return;
  }
  openNotebook(url).catch((err) => console.warn("[dd] could not open the notebook", err));
});
