"""Tag every kept problem with its pattern(s) using TypeSafe's Jev.

One request per problem: the state is the problem (title, statement, reference
solution, source tags), and there is one Noul question per concept in
patterns.json ("is this at heart a <pattern> problem?") plus two clean-up
questions (`self_contained`, `needs_figure`). Jev answers every question over
the same state in parallel, so a problem costs one request. Positional
`challengers[i]` drift (typesafe_difficulty/judge.py) does not apply: there is
one problem per state and questions are keyed by name.

Primary pattern = the highest-probability concept; secondaries = the others at
or above SECONDARY. Cached append-only in RAW_DIR/tags_cache.jsonl keyed by a
fingerprint of the state + questions, so a rerun re-asks only changed problems.
Writes This-Directory-Only/leetcode_course/tags.csv.

Needs TYPESAFE_API_KEY (This-Directory-Only/typesafe_difficulty/.env).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from build_pool import RAW_DIR

MODEL = os.environ.get("TYPESAFE_MODEL", "jev-1.13.0")
WORKERS = int(os.environ.get("TYPESAFE_WORKERS", "12"))
SECONDARY = 0.6
# --fallback also tags the verified LeetCode fallback tier (big sheet minus core);
# select_bank.py draws from it only to top up concepts the core bank leaves thin.
FALLBACK = "--fallback" in sys.argv
COURSE_DIR = Path(__file__).resolve().parents[2] / "leetcode_course"
CACHE = RAW_DIR / "tags_cache.jsonl"

CONTEXT = ("A coding-interview practice problem with its reference solution. The course "
           "groups problems by the algorithmic PATTERN the intended solution is built on. "
           "Judge by the technique the reference solution relies on, not by surface words "
           "in the title.")
QUALITY = {
    "self_contained": ("Is the statement self-contained: a reader can write the function from the text "
                       "alone, with inputs, output and rules all stated?",
                       "Every input, the expected output and the rules are stated in the text",
                       "Something needed is missing, e.g. an image, an undefined term or an unstated format"),
    "needs_figure": ("Does understanding the statement depend on a figure or image that is not in the text?",
                     "An example or rule refers to a picture the text does not reproduce",
                     "The text alone is enough"),
}


def load_problems() -> list[dict]:
    """Core LeetCode rows that passed verify.py + generated rows kept by gen_missing.py."""
    ok = {json.loads(l)["id"] for l in open(RAW_DIR / "verified.jsonl") if json.loads(l)["ok"]}
    out = []
    for line in open(RAW_DIR / "pool.jsonl"):
        p = json.loads(line)
        wanted = p["tier"] == "core" or (FALLBACK and p["source"] == "leetcode")
        if wanted and p["id"] in ok:
            out.append({"id": p["id"], "tier": p["tier"], "title": p["title"], "statement": p["statement"],
                        "solution": p["solution"], "hints": ", ".join(p["source_tags"] + [p["sheet_topic"]])})
    gen = RAW_DIR / "generated.jsonl"
    if gen.exists():
        for line in open(gen):
            g = json.loads(line)
            if g["ok"]:
                out.append({"id": g["key"], "tier": "core", "title": g["title"], "statement": g["statement"], "solution": g["solution"],
                            "hints": g["meta"]["topic"]})
    return out


def questions(kcs: list[dict]) -> dict[str, tuple[str, str, str]]:
    qs = {k["id"]: (f"Is this problem, at heart, a '{k['title']}' problem? {k['describe']}",
                    "The reference solution's central idea is this pattern",
                    "This pattern is absent or only incidental to the solution") for k in kcs}
    qs.update(QUALITY)
    return qs


def state_of(p: dict) -> dict:
    return {"context": CONTEXT, "title": p["title"], "statement": p["statement"][:3500],
            "reference_solution": p["solution"][:2500], "source_topic_hints": p["hints"]}


def fingerprint(state: dict, qs: dict) -> str:
    return hashlib.sha1(json.dumps([MODEL, state, qs], sort_keys=True).encode()).hexdigest()[:12]


def main() -> None:
    kcs = json.loads((COURSE_DIR / "patterns.json").read_text())["kcs"]
    qs = questions(kcs)
    probs = load_problems()
    if "--limit" in sys.argv:
        probs = probs[: int(sys.argv[sys.argv.index("--limit") + 1])]
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            r = json.loads(line)
            cache[r["fp"]] = r["answers"]
    from typesafe_sdk import Noul, NoulCriteria, TypeSafeClient
    client, lock = None, threading.Lock()

    def ask(p: dict) -> tuple[str, dict]:
        nonlocal client
        state = state_of(p)
        fp = fingerprint(state, qs)
        if fp in cache:
            return p["id"], cache[fp]
        with lock:
            if client is None:
                client = TypeSafeClient()
        nouls = {k: Noul(instructions=text, criteria=NoulCriteria(true=t, false=f)) for k, (text, t, f) in qs.items()}
        resp = client.system_one(model=MODEL, state=state, questions=nouls)
        answers = {k: float(resp.answers[k].noul) for k in qs}
        with lock:
            cache[fp] = answers
            with CACHE.open("a") as fh:
                fh.write(json.dumps({"fp": fp, "id": p["id"], "answers": answers}) + "\n")
        return p["id"], answers

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(ask, probs))
    kc_ids = [k["id"] for k in kcs]
    tier = {p["id"]: p["tier"] for p in probs}
    rows = []
    for pid, ans in results:
        ranked = sorted(kc_ids, key=lambda k: -ans[k])
        rows.append({"id": pid, "tier": tier[pid], "primary": ranked[0], "p_primary": round(ans[ranked[0]], 3),
                     "secondary": ";".join(k for k in ranked[1:] if ans[k] >= SECONDARY),
                     "self_contained": round(ans["self_contained"], 3), "needs_figure": round(ans["needs_figure"], 3)})
    out = COURSE_DIR / "tags.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for r in rows:
        if r["tier"] == "core":
            counts[r["primary"]] = counts.get(r["primary"], 0) + 1
    print(f"{len(rows)} tagged -> {out}; core counts per concept:")
    for k in kc_ids:
        print(f"  {counts.get(k, 0):4d}  {k}")


if __name__ == "__main__":
    main()
