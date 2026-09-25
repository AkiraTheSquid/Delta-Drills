# conceptual

## Purpose
- The backend of the conceptual-learning chat: a ChatGPT-style "why does this work" conversation, as opposed to the Practice tab's procedural drills.
- Designed so Seth's ChatGPT subscription (via an openai-oauth proxy) can answer every visitor, with spend limits in front of it.

## Owns
- `POST /api/conceptual/chat` — one streamed assistant turn over SSE (`router.py`).
- Which model and which door answers (`provider.py`): openai-oauth proxy when `CONCEPTUAL_CHAT_BASE_URL` is set, the OpenAI API key otherwise.
- The system prompt (`prompts.py`).
- Per-learner and global spend limits (`limits.py`).

## Does NOT own
- The chat UI — `Local_Deployed_Shared/conceptual/` (Deep Chat web component).
- The drill tutor — `app/practice/ai_router.py` `/ai-tutor` is a different, non-streaming surface about one drill.
- Running the openai-oauth proxy itself — see `~/Applications/openai-oauth`; deployment of it is not in this repo yet.
- API-key resolution — reused from `app/practice/chatgpt_helpers.load_chatgpt_api_key`.

## Key Files
- `router.py`: request schema, trim (24 turns, 8000 chars/message), limit check, SSE generator.
- `provider.py`: `resolve_provider()` → `ChatProvider(client, model, name)`.
- `limits.py`: `check_and_spend(user_id)` → refusal string or None.
- `prompts.py`: `build_system_prompt(concept)`.
- `watch.py`: route mounted, auth required, no sampling controls, limits behaviour.

## Data & External Dependencies
- `openai` Python client (Chat Completions, `stream=True`).
- Settings (`app/config.py`): `CONCEPTUAL_CHAT_BASE_URL`, `CONCEPTUAL_CHAT_MODEL` (default `gpt-6-luna` on the proxy — must be in the plan's `/v1/models` list — `gpt-4o-mini` on the API), `CONCEPTUAL_CHAT_PER_USER_HOUR` (40), `CONCEPTUAL_CHAT_DAILY_CAP` (1500). Add them to `.env` / Fly secrets — `Settings` rejects unknown keys, so they had to be declared there.
- `app.auth.get_current_user` — guest tokens count, so any visitor can chat.

## How It Works (Flow)
1. Frontend replays the visible thread as `{messages:[{role,content}], concept?}` with the learner's Bearer token.
2. Router drops empty messages, keeps the last 24, requires the last to be `user`, then spends one unit from `limits` (429 with a human message on refusal).
3. `resolve_provider()` picks the door; the model streams; each text delta becomes `data: {"delta": ...}`.
4. A failure mid-stream becomes one `data: {"error": ...}`; the stream always ends with `data: [DONE]`.

## Invariants & Constraints
- 🔴 Never send `temperature`/`top_p` — the ChatGPT/Codex backend behind openai-oauth rejects them. `watch.py` greps for it.
- 🔴 The openai-oauth proxy has NO auth. `CONCEPTUAL_CHAT_BASE_URL` must be a private address (Fly `.internal`, 127.0.0.1), never public.
- 🔴 The endpoint must keep `Depends(get_current_user)`; without it any curl spends the subscription.
- The proxy's refresh token rotates: give the server its OWN login (`npx openai-oauth login --oauth-file ...`), never a copy of `~/.codex/auth.json` (a rotated copy invalidates the original and logs Seth's codex out).
- Limits are in-process memory — correct for the one Fly machine; move to the DB before scaling out.
- ToS: openai-oauth's README says each person must use their own ChatGPT account. Serving the public from one subscription risks the account; the provider switch exists so moving to an API key is a secret flip.

## Extension Points
- New context (the concept a learner is on, their mastery): add a field to `ChatRequest`, thread it through `build_system_prompt`.
- Tools/function calling: pass `tools=` in `_stream` and handle `delta.tool_calls`; openai-oauth supports tool calls.
- Persistence of threads: new table + GET endpoint; the frontend currently keeps history in browser storage.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Limits reset on restart** — `ACTIVE`
  - When it happens: any Fly deploy/restart.
  - Symptom: a learner who hit the hourly limit can chat again.
  - Root cause: in-memory counters.
  - Prevention/fix: acceptable (it only forgives); move to DB if abuse appears.
  - Status: `ACTIVE`

- **No request BYTE cap** — `ACTIVE`
  - `ChatRequest` field bounds (64 messages, 16k chars each, concept 200) 422 an oversized thread before quota/model, but FastAPI has already read the whole body. A byte cap belongs in ASGI middleware or the proxy, app-wide.
  - Status: `ACTIVE`

## Recent Changes
- 2026-09-25: Critic fixes — body field bounds; idle learners swept past 2000 tracked; `per_user_hour=0` = off (was IndexError); one cached `OpenAI` client per door; watch checks the mounted route's real dependency graph.
- 2026-09-25: Created. `/api/conceptual/chat` SSE endpoint, provider switch (openai-oauth / OpenAI API), per-user + daily limits, system prompt.
