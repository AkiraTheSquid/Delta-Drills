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

/**
 * Unlock one problem's answer cell in the notebook.
 *
 * The solution cell is hidden by `content/colab_dd.css` until this arrives, so
 * the answer is not sitting under the code the learner is trying to write. The
 * notebook cannot unhide it on its own — Colab renders cell output in a
 * sandboxed iframe, so nothing a cell emits can reach a sibling cell.
 *
 * Best-effort by design: on a tab where the content script has not loaded (a
 * stock ARENA notebook, a tab opened before the extension was reloaded) this
 * throws and is swallowed. Nothing about the recorded attempt depends on it.
 */
async function revealSolution(problem) {
  const t = await colabTab();
  if (!t) return;
  await chrome.tabs.sendMessage(t.id, { type: "dd:reveal-solution", problem });
}

/**
 * The other direction: a check that finished in the notebook, handed to the app.
 *
 * `content/colab_focus.js` reads the line `dd_check` prints — the notebook has
 * no other way to reach us, since a cell's rich output is sandboxed and a
 * beacon would need a token pasted into the notebook. The app treats it as the
 * verdict click the learner would otherwise have made.
 *
 * Targeted at APP_ORIGIN, never "*": this says the learner got a problem right
 * or wrong, and the frame is the only thing entitled to hear it.
 */
chrome.runtime.onMessage.addListener((msg) => {
  if (!msg || msg.type !== "dd:check-result") return;
  const frame = document.querySelector("iframe");
  if (!frame || !frame.contentWindow) return;
  frame.contentWindow.postMessage(
    {
      source: "delta-drills-panel",
      type: "dd:check-result",
      problem: String(msg.problem || ""),
      correct: Boolean(msg.correct),
    },
    APP_ORIGIN,
  );
});

window.addEventListener("message", (event) => {
  if (event.origin !== APP_ORIGIN) return;
  const frame = document.querySelector("iframe");
  if (!frame || event.source !== frame.contentWindow) return;
  const msg = event.data;
  if (!msg || msg.source !== "delta-drills") return;

  if (msg.type === "dd:reveal-solution") {
    // A problem NUMBER, never a selector or a URL — the content script re-checks
    // it against /^\d+$/ before it touches the page.
    revealSolution(String(msg.problem || ""))
      .catch((err) => console.warn("[dd] could not reveal the solution", err));
    return;
  }

  if (msg.type !== "dd:open-notebook") return;
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
