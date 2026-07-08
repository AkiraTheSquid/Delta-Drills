#!/usr/bin/env python3
"""Report which questions still need the parameterized-regen rewrite.

The rewrite (docs/fable-exercise-generation-harness.md, Seth's 2026-07-06
directive): every question gets a real-parameter `solve(<params>)` stub graded
against 3-6 test cases including edge cases. Done ids live in
chatgpt/parameterized_regen_overrides.jsonl.

In-scope = non-torch, non-visual, not the intentional-error demo (id 65).
Demoted ex-visual questions (einops_visual_fixes.jsonl) count as in-scope.

METHOD (mandatory): INLINE authoring only — no Workflow tool, no agent
fan-outs (2026-07-07 session-limit incident). Per batch: author candidate
JSONL -> mech_gate_candidate.py --batch -> hand-review -> append to
parameterized_regen_overrides.jsonl + authored/parameterized_regen.jsonl +
refresh dd_questions.json texts -> build_solution_colabs.py -> export ->
audit --gate -> commit -> deploy -> backup. Starter convention: `return None`
(never raise). See memory delta-drills-parameterized-regen for gotchas.

Usage:  python3 regen_status.py [--ids N]   (print first N remaining ids)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_sys_path_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_sys_path_root))

from delta_paths import THIS_DIR_ONLY, get_chatgpt_runtime_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=int, default=30, help="how many remaining ids to print")
    args = parser.parse_args()

    questions = json.loads((THIS_DIR_ONLY / "questions_full.json").read_text())
    done_path = get_chatgpt_runtime_dir() / "parameterized_regen_overrides.jsonl"
    done: set[int] = set()
    if done_path.exists():
        done = {int(json.loads(l)["id"]) for l in done_path.read_text().splitlines() if l.strip()}

    remaining: list[dict] = []
    for q in questions:
        code = (q.get("answer_code") or "") + (q.get("starter_code") or "")
        if "torch" in code or q.get("supports_visual_output") or q["id"] == 65:
            continue
        if q["id"] in done:
            continue
        remaining.append(q)

    by_topic: dict[str, int] = {}
    for q in remaining:
        by_topic[q["topic"]] = by_topic.get(q["topic"], 0) + 1
    print(f"done: {len(done)}   remaining in-scope: {len(remaining)}")
    print("by topic:", dict(sorted(by_topic.items(), key=lambda kv: -kv[1])))
    ids = [q["id"] for q in remaining]
    print(f"next {args.ids} ids: {ids[:args.ids]}")


if __name__ == "__main__":
    main()
