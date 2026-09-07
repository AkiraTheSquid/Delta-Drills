"""Where the course RAN OUT — the drills that need writing next.

The ladder never re-serves a question a learner has already answered
(`prioritization.narrow_to_next_kc`). That is the right behaviour and it has a
consequence: a concept whose current rung holds only problems the learner has
seen can no longer be practised at all. Rather than paper over it with a
repeat, the queue stops and says so — and writes the fact down here, so the
gap is a work item rather than a thing Seth has to remember to mention.

    Seth, 2026-08-28: "whenever I'm still struggling in an area but I run out
    of problems, it just notifies you, and you have the skill or whatever that
    lets you know so that you can create more problems"

One JSON object on the volume, keyed by learner + concept + rung, so a learner
hitting the same wall twenty times is one row with a count rather than twenty
log lines. Read it with the `/drill-gaps` skill.

Failures here are swallowed: a queue that 500s because it could not write a
to-do list is strictly worse than one that quietly forgets to.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

from app.adaptive import DATA_DIR

logger = logging.getLogger(__name__)

GAPS_PATH = DATA_DIR / "content-gaps.json"

_UNSAFE = re.compile(r"[^A-Za-z0-9._@-]+")


def _key(user_id: str, gap: dict) -> str:
    return "|".join((
        _UNSAFE.sub("_", str(user_id)),
        str(gap.get("kc") or ""),
        str(gap.get("stage") or ""),
    ))


def _read() -> dict:
    if not GAPS_PATH.exists():
        return {}
    try:
        data = json.loads(GAPS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("content-gaps.json unreadable (%s) — starting a new one", e)
        return {}


def record(user_id: str, gap: dict) -> None:
    """Note that this learner has exhausted this concept's current rung.

    Idempotent per (learner, concept, rung): the first hit stamps `first_seen`
    and every later one bumps `hits` and `last_seen`. `hits` is the useful
    number — a gap hit once is somebody who wandered into a thin rung, and one
    hit fifty times is the rung that is actually blocking them.
    """
    if not gap or not gap.get("kc"):
        return
    try:
        now = datetime.now(timezone.utc).isoformat()
        data = _read()
        key = _key(user_id, gap)
        row = data.get(key) or {}
        data[key] = {
            "user_id": str(user_id),
            "kc": gap.get("kc"),
            "kc_title": gap.get("kc_title"),
            "stage": gap.get("stage"),
            # How many drills the rung holds — i.e. how many they got through
            # before it ran dry. A rung of 3 is a content bug; a rung of 40 is
            # a learner who has genuinely practised it out.
            "rung_size": gap.get("seen"),
            # How many of those they actually ANSWERED. A rung of 3 with 0
            # answered is a learner who skipped past it, not one who practised
            # it out, and the two want different drills written.
            "rung_answered": gap.get("answered"),
            "kc_total": gap.get("total"),
            "hits": int(row.get("hits") or 0) + 1,
            "first_seen": row.get("first_seen") or now,
            "last_seen": now,
        }
        GAPS_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename, never write in place. This is a read-modify-write
        # on one shared file from a concurrent server: a truncating write that
        # is interrupted leaves malformed JSON, `_read` then returns {} rather
        # than raising, and the NEXT record silently replaces every gap ever
        # recorded with a single row. `os.replace` is atomic on the same
        # filesystem, so a reader sees either the old file or the new one.
        # (codex, 2026-08-28.) It does not serialise concurrent writers — two
        # simultaneous requests can still lose one increment — and that is
        # accepted: `hits` is a priority hint for whoever writes the next
        # drills, not a measurement.
        tmp = GAPS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, GAPS_PATH)
    except Exception as e:  # never break question serving over a to-do list
        logger.warning("could not record content gap: %s", e)


def all_gaps() -> dict:
    """Every recorded gap. Read by the /content-gaps route and the skill."""
    return _read()


def learner_message(gap: dict) -> str:
    """What the learner is told when a rung runs dry.

    Says the concept, the rung, and how many they got through — so it reads as
    "you finished these", not as "the app is broken" — and names the one action
    that fixes it.
    """
    title = gap.get("kc_title") or gap.get("kc") or "this concept"
    # An UNTAGGED question has no rung, and the old fallback put the label
    # straight into the sentence — "you have finished every this rung problem".
    # Drop the adjective instead of inventing one.
    rung = RUNG_LABEL.get(gap.get("stage"), gap.get("stage") or "")
    rung_word = f"{rung} problem" if rung else "problem"
    seen = gap.get("seen")
    total = gap.get("total")
    # NOTHING WRITTEN AT THIS RUNG is not the same as HAVING FINISHED IT, and
    # saying the second when the first is true reads as a lie to the one person
    # who can tell: the learner looking at the same problem for the third time.
    # prioritization.rung_gap reports seen=0 for exactly this case.
    # A concept that owns NOTHING at any rung — a retired course id, or a node
    # the bank never got drills for. Replaying Seth's live state on 2026-09-07
    # put `numpy.memory-model` and `einsum.batch-dims` (total=0) through the
    # finished-them wording, which is a completion claim over an empty set.
    # `seen == 0` and not `not seen`: an untagged question has seen=None, which
    # means UNKNOWN, and unknown is not zero.
    if seen == 0 and not total:
        return (
            f"No problems have been written for “{title}” yet — this concept "
            "has no drills at any rung. Ask Claude to write some, or pick a "
            "different concept from the Knowledge Graph."
        )
    if seen == 0 and total:
        return (
            f"No {rung_word}s have been written for “{title}” yet — the "
            f"concept has {total} drill{'s' if total != 1 else ''} at other "
            "rungs, but nothing at this one, so there is nothing new to put in "
            "front of you here. Ask Claude to write drills at this rung, or "
            "pick a different concept from the Knowledge Graph."
        )
    # SEEN OUT IS NOT FINISHED. This message is also reached from the router's
    # repeat guard, which fires precisely when the learner has NOT answered the
    # question coming back at them — so telling somebody who skipped all three
    # lesson drills that they "finished every lesson problem, all 3 of them" is
    # false to the one person who can tell. prioritization.rung_gap carries the
    # answered count; `None` means an older gap dict that predates it, and those
    # only ever came from the every-drill-answered path. (codex, 2026-09-07.)
    answered = gap.get("answered")
    if seen and answered is not None and answered < seen:
        done = (
            f"answered {answered} of the {seen}"
            if answered
            else f"not answered any of the {seen}"
        )
        return (
            f"You have already seen every {rung_word} for “{title}” and "
            f"{done}. Nothing is being repeated, so there is nothing new to put "
            "in front of you here — ask Claude to write more drills for this "
            "concept, or pick a different one from the Knowledge Graph."
        )
    # No counts at all (an untagged question, `kc=None`): say what is true —
    # nothing here is new — rather than crediting a completion we cannot see.
    if not seen:
        return (
            f"There is nothing new left to serve for “{title}” — everything "
            "here has already been in front of you. Ask Claude to write more "
            "drills for this concept, or pick a different one from the "
            "Knowledge Graph."
        )
    count = f", all {seen} of them" if seen else ""
    return (
        f"You have finished every {rung_word} for “{title}”{count}. "
        "Nothing is being repeated, so there is nothing new to serve here yet — "
        "ask Claude to write more drills for this concept, or pick a different "
        "one from the Knowledge Graph."
    )


# The names the learner sees for the four rungs. Kept beside the message rather
# than imported from kc_graph: those are STORED strings that must not change,
# and these are labels that may (practice/stage-ladder.js draws the same four).
RUNG_LABEL = {
    "worked": "lesson",
    "faded": "fill-in-the-blank",
    "partial": "solo",
    "solo": "integrated",
}
