# ops/gap-watch

## Purpose
- Carries a content gap — a learner ran OUT of unseen drills on a rung — into the
  live Claude session the moment it happens, so the session writes the drills
  instead of Seth having to read the banner and paste it in.
- The domain concern is the last hop of the content-exhaustion loop:
  `app/content_gaps.py` records the gap on the Fly volume; this reads it and
  interrupts the attended session; `/drill-gaps` (a Claude skill) writes the
  drills.

## Owns
- `dd_gap_watch.py`: the poll loop. Fetches `/data/user_data/content-gaps.json`
  over `flyctl ssh console` every 60 s, prints one line per NEW hit, re-nags a
  wall the learner keeps hitting after 20 min, announces its own blindness after
  5 failed fetches.
- `SESSION_JOB`: the marker that says this is started by a session, never by a
  timer, and the exact command.

## Does NOT own
- Recording gaps (`This-Directory-Only/backend/app/content_gaps.py`).
- Writing drills — that is the session, via the `drill-gaps` skill, into the
  normal authoring pipeline.
- Any decision. It observes and interrupts. It must not grow toward spawning a
  Claude process or changing permission modes (Delta Note removed exactly that
  loop, `ops/autofix/`, as too risky).

## Data & External Dependencies
- `~/.fly/bin/flyctl` authenticated as Seth (`FLY_API_TOKEN` or `~/.fly/config.yml`).
- Machine-local state in `~/.local/state/delta-drills/`: `gap-watch.json`
  (`--source fly`) or `gap-watch-local.json` (`--source local`) — one announced-set
  per source (which hits have already been announced), so a local diagnostic run
  cannot mark a prod hit as already told — and `gap-watch.lock` (one watcher per
  machine, `flock`).
- No model API key, no backend import.

## How It Works (Flow)
1. Session starts it with the Monitor tool, `persistent: true`, `python3 -u`.
2. Arming pass: gaps hit in the last 2 h are announced (that is the wall Seth is
   at); older ones adopted silently — they are history, `/drill-gaps` still
   lists them.
3. Every pass: a gap whose `last_seen` advanced since it was last announced is
   printed as one line. Same key within 20 min: silent. After 20 min and still
   advancing: printed again, marked STILL DRY.
4. The session reads the line and runs `/drill-gaps` for that concept.

## Invariants & Constraints
- **A failed fetch is not an empty file.** `None` (could not look) keeps the
  previous view; only a parsed `{}` means no gaps. Five failures in a row are
  announced — silence must never read as "no gaps".
- **One line per event**, stdout only, flushed. Monitor turns each line into a
  message; anything on stderr never reaches the session.
- **Not deployed, not scheduled.** `SESSION_JOB` marker; no unit.

## Recent Changes
- 2026-09-13: Created. Seth: "any time I run out of problems, it automatically
  gets sent to this claude discussion for you to specifically create more
  problems … a shell with a command that's running that's listening".
- 2026-09-13 (critic): a `null` row in the gaps file no longer kills the loop
  (rows are filtered before the sort); the announced-set is namespaced per
  `--source`. Known: a hit that lands inside the 20-min re-nag window is
  announced when the window ends, not when it happens.
