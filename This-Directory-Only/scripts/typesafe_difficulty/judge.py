"""Pairwise "which is harder" judgments from TypeSafe's Jev, with a cache.

One request carries one champion and a batch of challengers in `state`, and
two Noul questions per challenger — "challenger harder than champion" and
"champion harder than challenger" — so a framing bias averages out the way
the legacy rater's forward + reversed prompts did. Both directions come back
from a single call because Jev answers every question over the same state in
parallel.

Every judgment is appended to `cache.jsonl` keyed by (model, champion,
challenger, framing, fingerprint) where the fingerprint hashes everything the
model actually saw — both drills' text, the context paragraph, the criteria
and the framing wording — so an edited prompt or rubric re-asks instead of
reusing a judgment made against old inputs. A rerun on unchanged inputs
re-reads instead of re-billing, and `scale.py` rescales offline.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from bank import DATA_DIR, Problem

MODEL = os.environ.get("TYPESAFE_MODEL", "jev-1.13.0")
BATCH = int(os.environ.get("TYPESAFE_BATCH", "8"))
CACHE = DATA_DIR / "cache.jsonl"

CONTEXT = (
    "These are two coding drills from a PyTorch / tensor-programming course "
    "(einops, einsum, indexing, broadcasting, CNNs). Each has a prompt and a "
    "reference solution. A drill is HARDER when a learner at this level would "
    "need more distinct ideas, less familiar API calls, trickier shape "
    "reasoning, or more steps to reach the solution unaided. Longer prompt "
    "text alone does not make a drill harder."
)
FRAMINGS = {
    "fwd": "Is `challengers[{i}]` a harder drill than `champion`?",
    "rev": "Is `champion` a harder drill than `challengers[{i}]`?",
}
CRITERIA = {
    "true": ("The first-named drill needs more distinct ideas, less familiar calls, "
             "trickier shape or index reasoning, or more solution steps than the other"),
    "false": ("The first-named drill is about as hard as, or easier than, the other; "
              "a longer prompt or solution on its own is not harder"),
}


def logit(p: float) -> float:
    eps = 1e-6
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def fingerprint(champion: Problem, challenger: Problem) -> str:
    """Hash of everything the model sees for this pair (both directions share it)."""
    h = hashlib.sha1()
    for part in (CONTEXT, json.dumps(CRITERIA, sort_keys=True), json.dumps(FRAMINGS, sort_keys=True),
                 json.dumps(champion.as_state(), sort_keys=True), json.dumps(challenger.as_state(), sort_keys=True)):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()[:12]


def load_cache() -> dict[tuple, float]:
    cache: dict[tuple, float] = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cache[(r["model"], r["champion"], r["challenger"], r["framing"], r.get("fp", ""))] = r["noul"]
    return cache


def _append_cache(rows: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


class Judge:
    """`p_harder(champion, challengers)` → {challenger id: P(challenger harder)}."""

    def __init__(self) -> None:
        self.cache = load_cache()
        self._client = None
        self.calls = 0
        self.input_tokens = 0

    def _get_client(self):
        if self._client is None:
            from typesafe_sdk import TypeSafeClient
            if not os.environ.get("TYPESAFE_API_KEY"):
                raise SystemExit("TYPESAFE_API_KEY is not set")
            self._client = TypeSafeClient()
        return self._client

    def p_harder(self, champion: Problem, challengers: list[Problem]) -> dict[int, float]:
        out: dict[int, float] = {}
        todo = []
        for c in challengers:
            if c.id == champion.id:
                out[c.id] = 0.5
                continue
            fp = fingerprint(champion, c)
            keys = {f: (MODEL, champion.id, c.id, f, fp) for f in FRAMINGS}
            if all(k in self.cache for k in keys.values()):
                out[c.id] = self._combine(self.cache[keys["fwd"]], self.cache[keys["rev"]])
            else:
                todo.append(c)
        for start in range(0, len(todo), BATCH):
            batch = todo[start:start + BATCH]
            out.update(self._ask(champion, batch))
        return out

    @staticmethod
    def _combine(p_fwd: float, p_rev: float) -> float:
        """Average the two framings in logit space; `rev` asks the opposite."""
        return sigmoid((logit(p_fwd) + logit(1 - p_rev)) / 2)

    def _ask(self, champion: Problem, batch: list[Problem]) -> dict[int, float]:
        from typesafe_sdk import Noul, NoulCriteria
        criteria = NoulCriteria(**CRITERIA)
        state = {
            "context": CONTEXT,
            "champion": champion.as_state(),
            "challengers": [c.as_state() for c in batch],
        }
        questions = {}
        for i in range(len(batch)):
            for framing, text in FRAMINGS.items():
                questions[f"{framing}_{i}"] = Noul(instructions=text.format(i=i), criteria=criteria)
        resp = self._get_client().system_one(model=MODEL, state=state, questions=questions)
        self.calls += 1
        self.input_tokens += int(getattr(resp.usage, "input_tokens", 0) or 0)
        rows, out = [], {}
        for i, c in enumerate(batch):
            got = {f: float(resp.answers[f"{f}_{i}"].noul) for f in FRAMINGS}
            fp = fingerprint(champion, c)
            for f, noul in got.items():
                self.cache[(MODEL, champion.id, c.id, f, fp)] = noul
                rows.append({"model": MODEL, "champion": champion.id, "challenger": c.id,
                             "framing": f, "fp": fp, "noul": noul})
            out[c.id] = self._combine(got["fwd"], got["rev"])
        _append_cache(rows)
        return out
