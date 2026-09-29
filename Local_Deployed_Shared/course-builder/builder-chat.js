/* course-builder/builder-chat.js — the course builder's left pane: a chat
   about the course being planned, with concepts from the graph attached.

   The chat is the Concept Chat tab's, not a second one: the same Deep Chat
   bundle, theme and SSE stream (conceptual/conceptual_chat.js exports
   `createStream` and `loadBundle`), answered by the same
   /api/conceptual/chat — so the same server-side prompt, model and limits,
   including running on Seth's ChatGPT subscription through openai-oauth.
   🔴 Nothing here names a model, key or system prompt; the server owns them.

   What this file adds is the ATTACHMENTS tray above the composer. A concept
   sent over from the graph card (builder-graph.js) sits there as a chip
   until it is removed, and while it does, every question sent carries a
   short plain-text note describing it — title, id, prerequisites, what it
   unlocks, the learner's mastery — prepended to that question only. The
   visible thread stays what the learner typed. The first attached concept
   also goes up as the request's `concept`, which the server's prompt
   builder already accepts. */
(function () {
  "use strict";

  const MAX_ATTACHED = 8;
  const SUGGESTIONS = [
    "Help me plan a course: I'll tell you the subject and who it's for",
    "What should a learner know before the concepts I've attached?",
    "Break the attached concept into smaller ideas a learner could fail separately",
    "Turn the attached concepts into a lesson sequence",
  ];

  let chat = null;
  let root = null;
  let tray = null;
  let stream = null;
  let mountedFor = null; // identity the thread belongs to
  let courseLabel = () => "";
  const attached = []; // kc ids, in the order they were added
  let onDetach = () => {};

  const esc = (v) => String(v == null ? "" : v).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const D = () => window.DDBuilderData;
  const title = (id) => (D().kc(id) || {}).title || id;

  const describe = (id) => {
    const names = (ids) => ids.slice(0, 6).map(title).join("; ") + (ids.length > 6 ? `; +${ids.length - 6} more` : "");
    const info = typeof window.deltaKcReadinessInfo === "function" ? window.deltaKcReadinessInfo(id) : null;
    const r = info && Number.isFinite(info.r) ? `${Math.round(info.r * 100)}%` : "not estimated yet";
    const p = D().parents(id), c = D().children(id);
    return `- ${title(id)} (${id}). Prerequisites: ${p.length ? names(p) : "none"}. Unlocks: ${c.length ? names(c) : "nothing yet"}. My mastery: ${r}.`;
  };

  const contextNote = () => {
    const lines = [`[Course builder — I'm planning a course on the concept graph shown beside this chat (currently showing: ${courseLabel() || "every concept"}).`];
    if (attached.length) {
      lines.push("Concepts I've attached from the graph:");
      attached.forEach((id) => lines.push(describe(id)));
    }
    return lines.join("\n") + "]\n\n";
  };

  // The note rides on the LAST user turn only — the one being asked now.
  const body = (messages) => {
    const out = messages.map((m) => ({ ...m }));
    for (let i = out.length - 1; i >= 0; i--) {
      if (out[i].role === "user") { out[i].content = contextNote() + out[i].content; break; }
    }
    const req = { messages: out };
    if (attached.length) req.concept = title(attached[0]).slice(0, 200);
    return req;
  };

  const renderTray = () => {
    if (!tray) return;
    tray.hidden = !attached.length;
    tray.innerHTML =
      `<span class="cb-tray-label">Attached</span>` +
      attached.map((id) =>
        `<span class="cb-chip" data-kc="${esc(id)}"><span class="cb-chip-dot" style="background:${esc(
          typeof window.deltaKcMasteryColor === "function"
            ? window.deltaKcMasteryColor((window.deltaKcReadinessInfo?.(id) || {}).r) : "currentColor")}"></span>` +
        `<span class="cb-chip-text">${esc(title(id))}</span>` +
        `<button type="button" class="cb-chip-x" aria-label="Remove ${esc(title(id))}">×</button></span>`).join("");
  };

  const attach = (id) => {
    if (!id || attached.includes(id)) return;
    if (attached.length >= MAX_ATTACHED) detach(attached[0]);
    attached.push(id);
    renderTray();
    chat?.focusInput?.();
  };
  const detach = (id) => {
    const i = attached.indexOf(id);
    if (i < 0) return;
    attached.splice(i, 1);
    renderTray();
    onDetach(id);
  };

  const introHtml = () =>
    '<div class="cc-intro">' +
    '<div class="cc-intro-title">What course do you want to build?</div>' +
    '<div class="cc-intro-sub">Plan it here with the AI. Click a concept on the graph to see where it sits, then attach it to the conversation.</div>' +
    '<div class="cc-suggestions">' +
    SUGGESTIONS.map((s) => `<button class="deep-chat-suggestion-button" type="button">${esc(s)}</button>`).join("") +
    "</div></div>";

  const identity = () => window.DDIdentity?.email?.() || "";

  const build = () => {
    const el = document.createElement("deep-chat");
    el.className = "cb-deep-chat";
    window.DDConceptualChatTheme?.apply(el);
    if (el.textInput && el.textInput.placeholder) {
      el.textInput = { ...el.textInput, placeholder: { ...el.textInput.placeholder, text: "Describe the course you want to build…" } };
    }
    el.introMessage = { html: introHtml() };
    el.connect = { stream: true, handler: stream.handler };
    el.remarkable = { math: true, linkTarget: "_blank", typographer: true };
    mountedFor = identity();
    // Per account; no identity = no persistence, same rule as Concept Chat.
    if (mountedFor) el.browserStorage = { key: `dd-course-builder-chat:${mountedFor}`, maxMessages: 200 };
    el.focusMode = false;
    el.displayLoadingBubble = true;
    el.errorMessages = { displayServiceErrorMessages: true };
    return el;
  };

  const mount = async (host, trayEl, opts) => {
    root = host;
    tray = trayEl;
    courseLabel = (opts && opts.courseLabel) || courseLabel;
    onDetach = (opts && opts.onDetach) || onDetach;
    tray.addEventListener("click", (e) => {
      const chip = e.target.closest(".cb-chip");
      if (chip && e.target.closest(".cb-chip-x")) detach(chip.dataset.kc);
    });
    renderTray();
    const cc = window.DDConceptualChat;
    if (!cc || !cc.createStream) {
      root.innerHTML = '<p class="cc-fail">The chat is unavailable. Refresh the page to try again.</p>';
      return;
    }
    stream = cc.createStream({ getChat: () => chat, body });
    root.setAttribute("aria-busy", "true");
    try {
      await cc.loadBundle();
    } catch (err) {
      console.warn("[course-builder] chat bundle failed to load", err);
      root.innerHTML = '<p class="cc-fail">The chat failed to load. Refresh the page to try again.</p>';
      return;
    } finally {
      root.removeAttribute("aria-busy");
    }
    chat = build();
    root.replaceChildren(chat);
  };

  const newChat = () => (stream ? window.DDConceptualChat.resetChat(stream, chat) : Promise.resolve());

  /* Sign-in, sign-out or a guest adopted without a reload: the thread and
     its browserStorage key belong to the identity it was built for, so the
     next person at this browser must not see it. Stop the answer in flight
     and rebuild for the new identity (conceptual_chat.js does the same). */
  window.addEventListener("delta:auth-state-changed", () => {
    if (!chat || mountedFor === identity()) return;
    stream.inflight()?.ctrl.abort();
    attached.slice().forEach(detach); // their picks go with them
    chat = build();
    root.replaceChildren(chat);
  });

  window.DDBuilderChat = { mount, attach, detach, attached: () => attached.slice(), newChat, refreshTray: renderTray };
})();
