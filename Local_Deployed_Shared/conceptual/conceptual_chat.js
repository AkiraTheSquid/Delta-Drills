/* ================================================================
   CONCEPTUAL CHAT — the "why does this work" tab (Seth, 2026-09-25).

   The Practice tab drills procedure. This tab is the other half: a
   ChatGPT/Claude-style conversation for conceptual learning, answered by
   /api/conceptual/chat (backend app/conceptual/), which can run on Seth's
   ChatGPT subscription through openai-oauth.

   The UI is Deep Chat (vendor/deep-chat/, MIT) — a framework-free web
   component. This file owns three things and nothing else:
     1. loading the 387 KB bundle LAZILY, the first time the tab opens
        (same pattern as groups/groups_checklist.js → Tiptap);
     2. the `connect.handler` that streams our SSE into Deep Chat;
     3. mounting/remounting one <deep-chat> into #concept-chat-root.
   How it looks is conceptual_chat_theme.js.

   🔴 THE SERVER OWNS THE PROMPT, THE MODEL AND THE LIMITS. This file sends
   only the visible thread. Never put a system prompt, model name or key
   here — anything in this file is readable by every visitor.

   `apiFetch` is read as `window.apiFetch`, which app.js assigns explicitly
   (`window.apiFetch = apiFetch`, next to its definition) — the bare const
   alone would NOT be a window property. It adds the Bearer token (guests
   have one too) and retries a 401 after a refresh.
   ================================================================ */
(() => {
  const SCRIPT_SRC = document.currentScript?.src || location.href;
  const BUNDLE_URL = new URL("../vendor/deep-chat/deepChat.bundle.js", SCRIPT_SRC).href;
  const ENDPOINT = "/api/conceptual/chat";
  const MAX_SENT_TURNS = 24; // the server keeps 24 too; sending more is waste

  const SUGGESTIONS = [
    "Why does softmax subtract the max before exponentiating?",
    "What is einsum actually doing when an index disappears?",
    "Explain attention like I know matrix multiplication",
    "Why do residual connections help deep networks train?",
  ];

  let bundle = null;
  let chat = null;
  let mountedFor = null; // identity the current thread belongs to
  let inflight = null; // { ctrl, closed } of the answer streaming right now

  const loadBundle = () => {
    if (!bundle) {
      bundle = import(BUNDLE_URL).catch((err) => {
        bundle = null; // let the next open retry instead of caching a failure
        throw err;
      });
    }
    return bundle;
  };

  const identity = () => window.DDIdentity?.email?.() || "";

  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  const introHtml = () =>
    '<div class="cc-intro">' +
    '<div class="cc-intro-title">What do you want to understand?</div>' +
    '<div class="cc-intro-sub">Ask about any idea behind the drills — intuition first, then the maths.</div>' +
    '<div class="cc-suggestions">' +
    SUGGESTIONS.map((s) => `<button class="deep-chat-suggestion-button" type="button">${esc(s)}</button>`).join("") +
    "</div></div>";

  /* Deep Chat's renderer knows `$`/`$$` only. Models still emit \( \) and
     \[ \] now and then; rewrite the whole accumulated text each chunk (the
     stream is re-rendered with `overwrite`, so a split `\` + `(` across two
     chunks still lands).

     🔴 NEVER INSIDE CODE. `re.compile(r"\(")` in a fenced block is code, and
     rewriting it would hand the learner a broken regex. Fenced blocks (closed
     or still streaming) and inline `code` spans pass through untouched. */
  const CODE = /(```[\s\S]*?(?:```|$)|`[^`\n]*`)/;
  const normaliseMath = (t) =>
    t.split(CODE).map((part, i) => (i % 2 ? part : part
      .replace(/\\\[([\s\S]*?)\\\]/g, (_, m) => `$$${m}$$`)
      .replace(/\\\(([\s\S]*?)\\\)/g, (_, m) => `$${m}$`))).join("");

  const thread = () =>
    (chat?.getMessages?.() || [])
      .filter((m) => typeof m.text === "string" && m.text.trim() && (m.role === "user" || m.role === "ai"))
      .map((m) => ({ role: m.role === "ai" ? "assistant" : "user", content: m.text }))
      .slice(-MAX_SENT_TURNS);

  const errorText = async (res) => {
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") return body.detail;
    } catch (_) { /* not JSON */ }
    return res.status === 401 || res.status === 403
      ? "Sign in (or refresh the page) to use the tutor."
      : "The tutor is unavailable right now.";
  };

  /* Deep Chat stream handler: onOpen → onResponse* → onClose. Our SSE is
     `data: {"delta"}` … optional `data: {"error"}` … `data: [DONE]`.

     🔴 AN ERROR BEFORE THE FIRST TOKEN CANNOT GO THROUGH onResponse. Deep
     Chat's streamError finalises the streamed message first, and with no
     text streamed that throws "No valid stream events were sent" — the error
     is never shown and the page gets an uncaught exception. So a failure
     with nothing streamed yet closes the stream (which also clears the
     loading dots) and adds the error as its own message. */
  const handler = async (_body, signals) => {
    let text = "";
    const fail = async (msg) => {
      if (text) {
        await signals.onResponse({ error: msg });
      } else {
        signals.onClose();
        chat?.addMessage({ error: msg });
      }
    };
    const fetcher = window.apiFetch;
    const ctrl = new AbortController();
    let closed;
    const turn = { ctrl, closed: new Promise((r) => { closed = r; }) };
    inflight = turn;
    signals.stopClicked.listener = () => ctrl.abort();
    try {
      if (typeof fetcher !== "function") throw new Error("app not ready");
      const res = await fetcher(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ messages: thread() }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        await fail(await errorText(res));
        return;
      }
      signals.onOpen();
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let cut;
        while ((cut = buf.indexOf("\n\n")) >= 0) {
          const line = buf.slice(0, cut).replace(/^data: ?/, "");
          buf = buf.slice(cut + 2);
          if (line === "[DONE]") continue;
          let ev;
          try { ev = JSON.parse(line); } catch (_) { continue; }
          if (ev.delta) {
            text += ev.delta;
            await signals.onResponse({ text: normaliseMath(text), overwrite: true });
          } else if (ev.error) {
            await fail(ev.error);
          }
        }
      }
    } catch (err) {
      if (err?.name !== "AbortError") {
        console.warn("[concept-chat]", err);
        await fail("Couldn't reach the tutor. Check your connection and try again.");
      }
    } finally {
      signals.onClose();
      if (inflight === turn) inflight = null;
      closed();
    }
  };

  const build = () => {
    const el = document.createElement("deep-chat");
    el.id = "concept-chat";
    window.DDConceptualChatTheme?.apply(el);
    el.introMessage = { html: introHtml() };
    el.connect = { stream: true, handler };
    el.remarkable = { math: true, linkTarget: "_blank", typographer: true };
    // Per account; no identity = no persistence, so two signed-out visitors
    // on one browser never see each other's thread.
    if (identity()) el.browserStorage = { key: `dd-concept-chat:${identity()}`, maxMessages: 200 };
    el.focusMode = false;
    el.displayLoadingBubble = true;
    el.errorMessages = { displayServiceErrorMessages: true };
    return el;
  };

  /* Size the page to the viewport below wherever it starts (under the
     topbar AND the guest banner, which is in flow), so the composer sits on
     the bottom edge without the window scrolling. */
  const fit = () => {
    const page = document.getElementById("page-concept-chat");
    if (!page || page.classList.contains("hidden")) return;
    const top = page.getBoundingClientRect().top + window.scrollY;
    page.style.height = `${Math.max(320, window.innerHeight - top)}px`;
  };
  window.addEventListener("resize", fit);

  const mount = async () => {
    const root = document.getElementById("concept-chat-root");
    if (!root) return;
    fit();
    const who = identity();
    if (chat && mountedFor === who && root.contains(chat)) return;
    root.setAttribute("aria-busy", "true");
    try {
      await loadBundle();
    } catch (err) {
      console.warn("[concept-chat] bundle failed to load", err);
      root.innerHTML = '<p class="cc-fail">The chat failed to load. Refresh the page to try again.</p>';
      return;
    } finally {
      root.removeAttribute("aria-busy");
    }
    // Replacing the component mid-answer: stop that answer first, or it keeps
    // spending the subscription into a detached element nobody can see.
    inflight?.ctrl.abort();
    chat = build();
    mountedFor = who;
    root.replaceChildren(chat);
  };

  const newChat = async () => {
    if (!chat) return;
    // 🔴 ABORT, WAIT FOR THE CLOSE, THEN CLEAR. A still-streaming answer
    // keeps writing into a cleared thread, and even once aborted, Deep Chat
    // commits the half-streamed message on onClose — so clearing before the
    // close lands leaves that fragment as the new thread's first message.
    const turn = inflight;
    if (turn) {
      turn.ctrl.abort();
      await turn.closed;
    }
    chat.clearMessages(true);
    chat.focusInput?.();
  };
  document.getElementById("concept-chat-new")?.addEventListener("click", newChat);

  // Sign-in / guest adoption without a reload: re-key the thread if the tab
  // is open now; otherwise the next open() sees the new identity.
  window.addEventListener("delta:auth-state-changed", () => {
    const page = document.getElementById("page-concept-chat");
    if (chat && page && !page.classList.contains("hidden")) mount();
  });

  window.DDConceptualChat = { open: mount, newChat };
})();
