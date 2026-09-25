# conceptual

## Purpose
- The Concept Chat tab: a ChatGPT/Claude-style conversation for CONCEPTUAL learning ("why does this work"), the counterpart to the Practice tab's procedural drills.
- Built on Deep Chat (`../vendor/deep-chat/`), a framework-free web component, so it stays plain JS with no build step.

## Owns
- `conceptual_chat.js`: lazy `import()` of the Deep Chat bundle on first open, the `connect.handler` that streams `/api/conceptual/chat` SSE into Deep Chat, mount/remount into `#concept-chat-root`, viewport fit, "New chat". Publishes `window.DDConceptualChat = { open, newChat }`.
- `conceptual_chat_theme.js`: every visual decision inside the chat — Deep Chat style objects + the `auxiliaryStyle` CSS string injected into its shadow root. Publishes `window.DDConceptualChatTheme = { apply }`.
- The thread's persistence in THIS browser: Deep Chat `browserStorage`, key `dd-concept-chat:<email|anon>`, last 200 messages.

## Does NOT own
- The prompt, model, provider (openai-oauth vs API) and spend limits — backend `This-Directory-Only/backend/app/conceptual/`.
- The page frame (header, height, root flex) — `../styles/conceptual/conceptual-chat.css`.
- Tab routing — `../app.js` `switchTab` calls `DDConceptualChat.open()` on arrival; markup in `../index.html` (`#page-concept-chat`).
- The Deep Chat code itself — `../vendor/deep-chat/` (never edited).

## Key Files
- `conceptual_chat.js`: handler + mount. Start here for behaviour.
- `conceptual_chat_theme.js`: look. Start here for styling.
- `watch.py`: wiring (tab/page/root/scripts/switchTab), tokens exist in all 3 themes, no raw colours, svg `xmlns`, no `@import` in auxiliaryStyle, no model/prompt/key on the client.

## Data & External Dependencies
- `window.apiFetch` (app.js): Bearer token + 401 refresh. Guests hold a token too.
- `window.DDIdentity.email()` (app.js): picks the storage key.
- `window.katex` (vendored, already on the page): Deep Chat renders maths with it, `output: "mathml"`.
- Backend `POST /api/conceptual/chat` — `{messages:[{role,content}]}` in; SSE `data: {"delta"}` / `{"error"}` / `[DONE]` out; 429/400/401/403 JSON `detail` before the stream.

## How It Works (Flow)
1. Tab click → `switchTab("concept-chat")` → `DDConceptualChat.open()` → fit height → `import()` bundle (once) → build `<deep-chat>` with theme, intro (welcome + 4 suggestion buttons), `connect: {stream: true, handler}`.
2. Send → Deep Chat calls `handler` → we POST the visible thread (last 24, from `getMessages()`, `ai`→`assistant`) → read SSE → `signals.onResponse({text: accumulated, overwrite: true})` per delta.
3. Stop → `stopClicked.listener` aborts the fetch. Errors → `fail()` (see invariant below).

## Invariants & Constraints
- 🔴 No system prompt, model id or key in this folder — every visitor can read it. `watch.py` greps for it.
- 🔴 Every colour is `var(--token)`; custom properties inherit into the shadow root, so the three themes work with no code.
- 🔴 `svg.content` must carry `xmlns` — Deep Chat parses it as `image/svg+xml`; without it no icon and a throw on every hover.
- 🔴 No `@import` in `auxiliaryStyle` (constructable stylesheet; refused).
- 🔴 An error before the first token must NOT go through `signals.onResponse({error})`: Deep Chat throws "No valid stream events were sent" and shows nothing. `fail()` closes the stream and `addMessage({error})` instead.
- Bundle is `import()`-ed, never `<script>`-tagged (387 KB).
- `#input` needs `boxSizing: border-box` or its padding overflows the host and clips the send button on a phone.

## Extension Points
- Context from the app (concept the learner is on): send `concept` in the POST body; the backend already threads it into the system prompt.
- Launch from elsewhere with a prefilled question: `switchTab("concept-chat")` then `document.querySelector("deep-chat").submitUserMessage({text})`.
- Server-side thread persistence: replace `browserStorage` with `loadHistory` against a new GET endpoint.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Deep Chat stream error before first token throws** — `RESOLVED`
  - When it happens: backend down, 429, proxy failure before any delta.
  - Symptom: no error bubble; uncaught "No valid stream events were sent".
  - Root cause: Deep Chat `streamError` finalises an empty streamed message.
  - Prevention/fix: `fail()` in `conceptual_chat.js`.
  - Status: `RESOLVED`

- **New chat mid-answer kept the fragment** — `RESOLVED`
  - When it happens: New chat (or a sign-in remount) while an answer streams.
  - Symptom: the cleared thread starts with the half-written answer.
  - Root cause: Deep Chat commits the streamed message on `onClose`, which lands AFTER an immediate `clearMessages`.
  - Prevention/fix: `newChat` aborts `inflight.ctrl`, awaits `inflight.closed`, then clears; `mount` aborts before replacing.
  - Status: `RESOLVED`

- **`\( \)` rewrite inside code** — `RESOLVED`
  - `normaliseMath` rewrites maths delimiters only OUTSIDE ``` fences and inline `code` (a regex `r"\("` must survive).

## Recent Changes
- 2026-09-25: Critic fixes — New chat/remount abort the live stream (await close before clear); maths rewrite skips code; no `browserStorage` without an email.
- 2026-09-25: Created. Concept Chat tab on Deep Chat 2.5.1: lazy bundle, SSE handler, ChatGPT-style theme (3 themes, phone), suggestions, New chat, Stop, error bubbles.
