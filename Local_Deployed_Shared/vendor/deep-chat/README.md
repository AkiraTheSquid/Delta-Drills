# vendor/deep-chat

## Purpose
- Deep Chat 2.5.1 (OvidijusParsiunas/deep-chat, MIT): the chat UI web component behind the conceptual-learning chat.

## Owns
- `deepChat.bundle.js` — the npm tarball's `dist/deepChat.bundle.js`, byte for byte (387 KB, sha256 pinned in `watch.py`). Self-contained ESM: remarkable (markdown), fetch-event-source and speech-to-element are inside it; no imports, no framework.
- `LICENSE` — MIT, from the same tarball.

## Does NOT own
- How the chat is configured, styled or connected — `../../conceptual/`.
- The model call — the component never talks to a model directly here. `conceptual/` uses Deep Chat's `connect.handler` to stream from our own backend (`/api/conceptual/chat`). The vendor URLs baked into the bundle (api.openai.com etc.) are only used by `directConnection`, which we never set.

## Key Files
- `deepChat.bundle.js`: `customElements.define("deep-chat", …)` + `export { DeepChat }`.
- `watch.py`: present, LICENSE, hash, still registers `<deep-chat>`, never script-tagged or CDN-loaded.

## Data & External Dependencies
- None at runtime. Math uses `window.katex` (vendored `../katex/`), already on the page.

## How It Works (Flow)
1. `conceptual/conceptual_chat.js` calls `import("../vendor/deep-chat/deepChat.bundle.js")` the first time the chat tab opens (same pattern as Groups → Tiptap).
2. The import registers `<deep-chat>`; the chat module creates one and mounts it.

## Invariants & Constraints
- 🔴 Imported lazily, never `<script>`-tagged in index.html: 387 KB on every boot for a tab most visits never open.
- 🔴 No CDN copy. Offline must work (vendor policy, 2026-09-09).
- Do not edit the bundle. Upgrade = re-copy from the tarball (`npm pack deep-chat@X`), update the hash in `watch.py` and the version here.

## Extension Points
- Upgrade: `npm pack deep-chat@<ver>`, copy `package/dist/deepChat.bundle.js` + `package/LICENSE`, update the hash, re-run the conceptual chat by hand (the shadow-DOM style keys are the likely break).

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)
- None yet.

## Recent Changes
- 2026-09-25: Vendored deep-chat 2.5.1 for the conceptual chat.
