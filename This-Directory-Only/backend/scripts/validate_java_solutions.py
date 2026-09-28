"""Reference Java answers for the LeetCode drills (Show answer in Java).

The answers live in Local_Deployed_Shared/lessons/leetcode/solutions_java.json
as {question_id: code}. Every one must pass every case of its drill through
the same grader a learner's Java goes through (app/leetcode_java.py), so the
"💡 Solution (Java)" cell never shows an answer that would be marked wrong.

Run: PATH=<a JDK 17-23>/bin:$PATH .venv/bin/python scripts/validate_java_solutions.py
       [--merge FILE]   add/replace answers from FILE ("### <id>" then the
                        code, repeated); only answers that pass are merged
       [--ids 1,2,3]    check only these ids
       [--require-all]  also fail if a Java-capable drill has no answer

Exit 1 on any failing answer (or a missing one under --require-all).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import leetcode_java as J  # noqa: E402
from app.questions import get_all_questions, get_question_by_id  # noqa: E402

STORE = Path(__file__).resolve().parents[3] / "Local_Deployed_Shared" / "lessons" / "leetcode" / "solutions_java.json"
WORKERS = 6  # each case runs a JVM; keep the machine under 80% CPU


def load() -> dict[str, str]:
    return json.loads(STORE.read_text()) if STORE.exists() else {}


def save(rows: dict[str, str]) -> None:
    ordered = dict(sorted(rows.items(), key=lambda kv: int(kv[0])))
    STORE.write_text(json.dumps(ordered, indent=1, ensure_ascii=False) + "\n")


def parse_blocks(text: str) -> dict[str, str]:
    """`### <id>` headers, each followed by that drill's Java answer."""
    out: dict[str, str] = {}
    for chunk in re.split(r"^### ", text, flags=re.M)[1:]:
        head, _, body = chunk.partition("\n")
        qid = head.split()[0]
        if not qid.isdigit():
            raise SystemExit(f"bad block header: ### {head}")
        out[qid] = body.strip("\n") + "\n"
    return out


def check(qid: str, code: str) -> str | None:
    """None when the answer passes every case, else why not."""
    question = get_question_by_id(int(qid))
    if question is None:
        return "no such question"
    if not J.supports_java(question):
        return "drill has no Java form"
    rows, execution, _ = J.run_java_tests(code, question)
    bad = [r for r in rows if not r.passed]
    if not bad:
        return None
    first = bad[0]
    detail = first.error or f"got {first.actual!r}, expected {first.expected!r}"
    return f"{len(rows) - len(bad)}/{len(rows)} passed; {detail.strip()[:400]}"


def run(pairs: dict[str, str]) -> dict[str, str | None]:
    with ThreadPoolExecutor(WORKERS) as pool:
        verdicts = pool.map(lambda kv: check(*kv), pairs.items())
    return dict(zip(pairs, verdicts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge")
    ap.add_argument("--ids")
    ap.add_argument("--require-all", action="store_true")
    args = ap.parse_args()

    stored = load()
    if args.merge:
        fresh = parse_blocks(Path(args.merge).read_text())
        verdicts = run(fresh)
        good = {q: fresh[q] for q, v in verdicts.items() if v is None}
        for q, v in verdicts.items():
            if v:
                print(f"FAIL {q}: {v}")
        stored.update(good)
        save(stored)
        print(f"merged {len(good)}/{len(fresh)}; store now {len(stored)}")
        return 0 if len(good) == len(fresh) else 1

    pairs = stored
    if args.ids:
        wanted = [q.strip() for q in args.ids.split(",") if q.strip()]
        pairs = {q: stored[q] for q in wanted if q in stored}
    failures = {q: v for q, v in run(pairs).items() if v}
    if args.ids:
        # An asked-for id with no stored answer is a failure, not a silent skip.
        failures.update({q: "no stored answer" for q in wanted if q not in stored})
    for q, v in failures.items():
        print(f"FAIL {q}: {v}")
    missing: list[int] = []
    if args.require_all:
        capable = [q.id for q in get_all_questions() if J.is_leetcode(q) and J.supports_java(q)]
        missing = [q for q in capable if str(q) not in stored]
        if missing:
            print(f"MISSING {len(missing)}: {missing[:40]}{' …' if len(missing) > 40 else ''}")
    checked = len(pairs) + sum(v == "no stored answer" for v in failures.values())
    print(f"{checked - len(failures)}/{checked} Java answers pass")
    return 1 if failures or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
