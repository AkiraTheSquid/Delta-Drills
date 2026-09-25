# styles/conceptual

## Purpose
- The page frame around the Concept Chat: a full-height column with a slim header and the chat filling the rest.

## Owns
- `conceptual-chat.css`: `#page-concept-chat` layout, `.cc-head` / `.cc-head-title` / `.cc-new`, `#concept-chat-root` flex fill, loading/fail placeholders.

## Does NOT own
- Anything inside `<deep-chat>` (bubbles, composer, markdown) — shadow DOM; styled by `../../conceptual/conceptual_chat_theme.js`.
- The page HEIGHT — set inline by `conceptual_chat.js` (viewport minus whatever sits above, guest banner included).

## Key Files
- `conceptual-chat.css`: the frame.
- `watch.py`: tokens exist in every theme, no raw colours, selectors still in index.html.

## Data & External Dependencies
- Tokens from `../variables.css`.

## How It Works (Flow)
1. `.page` gives 60px top padding app-wide; this page zeroes it and lays out as a flex column.
2. The header clears the placement-timer notch (30px top padding).

## Invariants & Constraints
- 🔴 `#page-concept-chat .cc-head-title` needs the id: `.page h1` (layout.css) is more specific than a bare class and would make it a 32px title.
- Only tokens; no raw colours.

## Extension Points
- Header actions (e.g. thread list) go in `.cc-head`.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)
- None yet.

## Recent Changes
- 2026-09-25: Created with the Concept Chat tab.
