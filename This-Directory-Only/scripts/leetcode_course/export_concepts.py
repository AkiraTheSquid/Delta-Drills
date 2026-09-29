"""Write the LeetCode concept pages the Knowledge Graph shows.

The course has no lesson pages, so clicking a `leetcode.*` bubble used to open
nothing. This joins the authored notes (leetcode_course/concept_notes.md, one
`## <kc id>` section per concept) with the exported drills and writes
Local_Deployed_Shared/lessons/leetcode/concepts.json:

    {"version": 1, "kcs": {"<kc>": {"title", "description_markdown",
                                    "drills", "by_difficulty", "examples"}}}

`examples` are the concept's easiest drills (id, problem title, difficulty
label), so the pane can say what practising it looks like. Run on its own after
editing the notes; export_app.py also calls it so a re-export never leaves the
pages describing a stale drill set. Fails if a registered concept has no notes.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
COURSE_DIR = REPO / "This-Directory-Only" / "leetcode_course"
OUT_DIR = REPO / "Local_Deployed_Shared" / "lessons" / "leetcode"
NOTES = COURSE_DIR / "concept_notes.md"
N_EXAMPLES = 6

_HEAD = re.compile(r"^## (leetcode\.[a-z0-9-]+)\s*$", re.M)
_TITLE = re.compile(r"^\*\*(.+?)\*\*")


def parse_notes(text: str) -> dict[str, str]:
    """`## leetcode.x` section bodies, keyed by concept id."""
    parts = _HEAD.split(text)
    # parts = [preamble, id1, body1, id2, body2, ...]
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def _problem_title(row: dict) -> str:
    m = _TITLE.match(row.get("question_text") or "")
    return m.group(1).strip() if m else f"Problem {row['id']}"


def build(graph: list[dict], rows: list[dict], notes: dict[str, str]) -> dict:
    by_kc: dict[str, list[dict]] = {}
    for r in rows:
        by_kc.setdefault(r["leetcode_kc"], []).append(r)
    out = {}
    for k in graph:
        kc = k["id"]
        drills = sorted(by_kc.get(kc, []), key=lambda r: (int(r.get("difficulty_score") or 0), int(r["id"])))
        counts: dict[str, int] = {}
        for r in drills:
            label = r.get("difficulty_label") or "unrated"
            counts[label] = counts.get(label, 0) + 1
        out[kc] = {
            "title": k["title"],
            "description_markdown": notes.get(kc, ""),
            "drills": len(drills),
            "by_difficulty": counts,
            "examples": [
                {"id": int(r["id"]), "title": _problem_title(r), "difficulty": r.get("difficulty_label") or ""}
                for r in drills[:N_EXAMPLES]
            ],
        }
    return {"version": 1, "kcs": out}


def notes_problem(graph: list[dict]) -> str | None:
    """Why concept_notes.md can't be exported against `graph`, else None.
    Every pattern needs notes, drills or not, so a concept that gains its
    first drill already has a page. export_app.py asks BEFORE it writes."""
    notes = parse_notes(NOTES.read_text(encoding="utf-8"))
    missing = [k["id"] for k in graph if not notes.get(k["id"])]
    stray = sorted(set(notes) - {k["id"] for k in graph})
    return f"concept_notes.md: missing {missing}, unknown {stray}" if missing or stray else None


def main() -> int:
    graph = json.loads((COURSE_DIR / "patterns.json").read_text())["kcs"]
    problem = notes_problem(graph)
    if problem:
        print(problem, file=sys.stderr)
        return 1
    rows = json.loads((OUT_DIR / "problems.json").read_text())
    notes = parse_notes(NOTES.read_text(encoding="utf-8"))
    doc = build(graph, rows, notes)
    (OUT_DIR / "concepts.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(doc['kcs'])} concept pages -> {OUT_DIR / 'concepts.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
