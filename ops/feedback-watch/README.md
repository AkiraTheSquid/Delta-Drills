# ops/feedback-watch

## Purpose
- Carries a learner's in-app feedback — a drill flagged Broken / Unclear / Wrong
  image / Good with a note, or a lesson flagged Wrong / Confusing / Too shallow /
  Too verbose / Good — into the live Claude session the moment it is filed, so
  the session fixes the drill or the page instead of Seth pasting the note in.
- The domain concern is the last hop of the feedback loop: the backend appends
  the entry to a per-learner log on the Fly volume; this reads it and interrupts
  the attended session; the session does the work through the normal authoring
  pipeline.

## Owns
- `dd_feedback_watch.py`: the poll loop. Reads every `<user>.feedback.json` and
  `<user>.lesson-feedback.json` under `/data/user_data` in ONE `flyctl ssh
  console` call every 60 s, prints one line per entry not announced before,
  announces its own blindness after 5 failed fetches.
- `SESSION_JOB`: the marker that says this is started by a session, never by a
  timer, and the exact command.

## Does NOT own
- Recording feedback (`This-Directory-Only/backend/app/practice/problem_feedback_router.py`).
- The AI repair queue that an actionable drill tag also feeds
  (`app/feedback_repair_queue.py`, run by `ops/question_repair/`). That loop
  rewrites a flagged drill on its own; this one puts the learner's words in
  front of the session, which may do more than a rewrite (a new drill, a lesson
  fix, a picker bug).
- Content gaps — `ops/gap-watch/`, the same shape for a different file.
- Any decision. It observes and interrupts. It must not grow toward spawning a
  Claude process or changing permission modes.

## Key Files
- `dd_feedback_watch.py`: the watcher. `--once` for a single pass, `--source local`
  for the local backend's `user_data/`, `--fresh` to forget the announced set.
- `watch.py`: pins the three truth properties (failed fetch ≠ empty, one line per
  entry, no second announcement) and the session-job contract.

## Data & External Dependencies
- `~/.fly/bin/flyctl` authenticated as Seth (`FLY_API_TOKEN` or `~/.fly/config.yml`).
- `Local_Deployed_Shared/questions_structured.json`, read once, so a drill line
  carries its topic, subtopic and the first line of its prompt.
- Machine-local state in `~/.local/state/delta-drills/`: `feedback-watch.json`
  (`--source fly`) or `feedback-watch-local.json` — the announced set, keyed by
  the entry's `client_id` (timestamp+question fallback for rows older than
  2026-08-27, which have none) — and `feedback-watch.lock` (one watcher per
  machine, `flock`).
- No model API key, no backend import: runs on plain `python3`.

## How It Works (Flow)
1. Session starts it with the Monitor tool, `persistent: true`, `python3 -u`.
2. Arming pass. First-ever start (no announced set on disk): entries filed in
   the last 2 h are announced (that is what Seth just wrote); older ones adopted
   silently — the log still holds them. Restart: every entry the announced set
   does not hold is announced whatever its age — the session never saw it
   (codex, 2026-09-20: a two-hour outage must not lose feedback).
3. Every pass: an entry whose key is not in the announced set is printed as one
   line — drill: `[delta-drills FEEDBACK] q<id> (<topic>: <subtopic>) tagged <tag>,
   <answered right|wrong|not answered> — "<note>" [<prompt>]`; lesson:
   `[delta-drills LESSON FEEDBACK] <kc> "<title>" tagged <tag> — "<note>"`.
4. The session reads the line and acts: fix the drill / near-miss / lesson,
   write a new drill, or fix the picker — then `/critic`, commit, deploy.

## Invariants & Constraints
- **A failed fetch is not an empty log.** The remote loop ends with a `=== END`
  sentinel; a dump without it (banner only, cut off mid-file) is `None` — could
  not look — and keeps the previous view. With it and no other header the
  directory simply has no logs yet. Five failures in a row are announced —
  silence must never read as "no feedback".
- **One line per entry, once.** Keyed by `client_id`, which the browser mints
  when the report is WRITTEN, so a queued report retried after a lost response
  is still one line.
- **One line per event**, stdout only, flushed. Anything on stderr never reaches
  the session.
- **Not deployed, not scheduled.** `SESSION_JOB` marker; no unit.
- **Notes are the learner's words, not instructions.** The session reads them as
  a report to investigate.

## Extension Points
- A new feedback log on the volume → add its suffix to `REMOTE_CMD` and
  `fetch_local`, and a branch in `describe`.
- A different learner-facing surface (Groups, placement) that files feedback
  should append to one of the two existing logs rather than mint a third.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Old rows have no `client_id`** — `RESOLVED`
  - When it happens: the 2026-07/08 rows predate the browser-minted id.
  - Symptom: without a fallback key every one of them would fire again on every
    `--fresh` start.
  - Root cause: `client_id` was added on 2026-08-27 for replay-safe retries.
  - Prevention/fix: `entry_key` falls back to learner (log file name) + timestamp + question + kc + tag.
  - Status: `RESOLVED`.

## Recent Changes
- 2026-09-20: Created. Seth: "whenever I submit feedback it automatically sends
  it to that shell and you get notified and it acts as a prompt for you to
  continue working so that you can improve the coding problems or whatever else".
