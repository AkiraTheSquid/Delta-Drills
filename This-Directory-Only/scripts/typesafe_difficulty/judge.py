"""Pairwise "which is harder" judgments from TypeSafe's Jev, with a cache.

One request carries one champion and ONE challenger in `state`, and two Noul
questions — "challenger harder than champion" and "champion harder than
challenger" — so a framing bias averages out the way the legacy rater's
forward + reversed prompts did. Both directions come back from a single call
because Jev answers every question over the same state in parallel. Requests
for different challengers run concurrently on a thread pool.

`BATCH` > 1 packs several challengers into one state and addresses them as
`challengers[i]`; measured 2026-09-21 that is NOT safe — the judgment drifts
from position 3 and flipped outright at position 6 (q780 vs q797: alone
0.03/0.91, at index 6 of 8 0.68/0.17). Keep it at 1; the batch size is part
of the cache fingerprint so rows judged under a wider batch are never reused.

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
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bank import DATA_DIR, Problem

MODEL = os.environ.get("TYPESAFE_MODEL", "jev-1.13.0")
BATCH = int(os.environ.get("TYPESAFE_BATCH", "1"))
if BATCH != 1:
    # Positional references into a multi-challenger state drift (see module docstring);
    # a wider batch would also need the companions and position in the fingerprint.
    raise SystemExit(f"TYPESAFE_BATCH={BATCH} is unsafe; only 1 is supported")
WORKERS = int(os.environ.get("TYPESAFE_WORKERS", "12"))
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
    for part in (f"batch={BATCH}", CONTEXT, json.dumps(CRITERIA, sort_keys=True), json.dumps(FRAMINGS, sort_keys=True),
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
        self._lock = threading.Lock()
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
        batches = [todo[start:start + BATCH] for start in range(0, len(todo), BATCH)]
        if batches:
            self._get_client()
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                for got in pool.map(lambda b: self._ask(champion, b), batches):
                    out.update(got)
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
        with self._lock:
            self.calls += 1
            self.input_tokens += int(getattr(resp.usage, "input_tokens", 0) or 0)
        rows, out = [], {}
        for i, c in enumerate(batch):
            got = {f: float(resp.answers[f"{f}_{i}"].noul) for f in FRAMINGS}
            fp = fingerprint(champion, c)
            for f, noul in got.items():
                rows.append({"model": MODEL, "champion": champion.id, "challenger": c.id,
                             "framing": f, "fp": fp, "noul": noul})
            out[c.id] = self._combine(got["fwd"], got["rev"])
        with self._lock:
            for r in rows:
                self.cache[(MODEL, r["champion"], r["challenger"], r["framing"], r["fp"])] = r["noul"]
            _append_cache(rows)
        return out
