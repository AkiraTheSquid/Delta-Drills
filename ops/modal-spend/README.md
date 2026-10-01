# modal-spend

## Purpose
- Tell Seth, off the app, when the notebook kernels (Modal sandboxes) are spending faster than the $30/month budget. September 2026 ran $56 before anyone looked.
- The in-app ceiling (`app/modal_spend.py`) stops new kernels at the cap; this is the early warning that comes before it.

## Owns
- The daily check: yesterday's cost and the month's pace against two limits.
- The alert file every Claude Code session prints at start, and the Telegram push.
- The systemd user unit + timer that run it on Greed (symlinked into `~/.config/systemd/user/`).

## Does NOT own
- The ceiling that refuses kernels — `This-Directory-Only/backend/app/modal_spend.py`, which runs on Fly.
- The phone transport — it reuses the Delta Note alert relay's Telegram bot config (`~/.config/delta-note/alert-relay.env`), read-only.
- Sandbox lifetimes and the idle reaper — `app/modal_kernel.py`.

## Key Files
- `dd_modal_spend_check.py`: the check. Reads Modal billing, writes/clears the alert, pushes. Exit 0 ok, 2 alert raised, 1 billing unreadable.
- `dd_modal_spend_alert`: SessionStart hook (symlinked as `~/.local/bin/dd_modal_spend_alert`); prints the alert or a stale-check warning.
- `dd-modal-spend-check.service` / `.timer`: daily at 09:30 local, `Persistent=true`.

## Data & External Dependencies
- Modal billing API (`Workspace.billing.report` for yesterday, `.billing.summary()` for the month) via `~/.modal.toml`.
- The backend venv's interpreter — the only one here with `modal`.
- Telegram Bot API, token read at runtime from the file the relay env names. Never printed.
- State in `~/.local/state/delta-drills/`: `modal-spend-alert.md`, `modal-spend-last-run`.

## How It Works (Flow)
1. Timer fires → read yesterday (UTC) and the month's metered cost.
2. Pace = month so far ÷ max(days elapsed, 3) × days in month. The 3-day floor stops one busy first morning projecting $40.
3. Yesterday > `DD_MODAL_DAILY_ALERT_USD` (1.50) or pace > `DD_MODAL_PACE_ALERT_USD` (30) → write the alert file, push, exit 2.
4. Otherwise, if an alert was open, delete it and push "back under limits".

## Invariants & Constraints
- **Only a run that read the bill may clear the alert.** A billing error exits 1 and touches nothing, and does not refresh `modal-spend-last-run` — so a dead credential shows up as a stale check, not a quiet month.
- Metered cost, not billed: the plan's $30 of free credits make "billed" read $0 until it is too late. Same number the in-app cap uses.
- Never print or log the Telegram URL; it carries the bot token.

## Extension Points
- New limit → a line in `judge()` plus a flag; keep the test override (`--daily-threshold 0`) working, it is how the alert path is verified.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Greed off = no alert** — `ACTIVE`
  - When it happens: the machine is asleep or off at 09:30 for days.
  - Symptom: no push while spend climbs.
  - Root cause: the check runs on Greed by design (spec D5); `Persistent=true` only catches up on wake.
  - Prevention/fix: the in-app cap still stops kernels at `DD_MODAL_CAP_USD`; a Modal dashboard spend limit, if the workspace offers one, is the second backstop.
  - Status: `ACTIVE` (accepted trade-off).

## Recent Changes
- 2026-09-30: Folder created (Modal budget pass 1, `SPEC_MODAL_COMPUTE_BUDGET.md` D). Verified: forced $0 limits → Telegram delivered + alert file; real limits → cleared.
