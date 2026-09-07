#!/usr/bin/env python3
"""Content critic launcher: ONE agentic GPT-6 Astra review of drills + lesson
pages against scripts/CONTENT_RUBRIC.md.

Why this shape (Seth, 2026-09-07): the first version chunked the scope and
launched 24 model processes at once ("did you run 24 calls at once? holy
shit"). The replacement is at most TWO model processes per review, one per
model, each reading the whole scope with its own tools:

- GPT-6 Astra: this script writes a task file and runs `codex exec` once,
  read-only sandbox, reasoning effort PINNED on the command line (the code
  critic sat silently at `medium` for weeks because effort was inherited from
  a config file).
- Claude Fable 5.1: the in-session author reads the same task file and writes
  its own report by hand — no second process, and it does not review its own
  fixes without saying so.

Reviewers never edit content. The author reconciles the two reports
(agreed / fable-only / astra-only), verifies single-reviewer findings against
the data, then fixes and re-runs `audit_ladder_pairing.py --strict`.

Usage
-----
    python3 scripts/content_critic.py --qids 847-940 \
        --pages Local_Deployed_Shared/lessons/einops --label einops
    python3 scripts/content_critic.py --qids 847-940 --pages ... --dry-run   # write task, don't run

Output: scripts/content_review/<YYYYMMDD>-<label>-astra.md (and the task file
next to it as <YYYYMMDD>-<label>-task.md). ~20 min, ~170k tokens for 94 drills
+ 12 pages.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUBRIC = REPO / "scripts" / "CONTENT_RUBRIC.md"
BANK = REPO / "This-Directory-Only" / "questions_full.json"
REVIEW_DIR = REPO / "scripts" / "content_review"
MODEL = "gpt-6-astra"
EFFORT = "xhigh"

TASK = """You are a content reviewer for Delta Drills, a coding-drill tutor. Review AGAINST THE RUBRIC ONLY.

Read first, with your own tools:
1. {rubric}  (the standard; every finding must cite a CODE from it)
2. {pages}  (lesson pages; frontmatter lists which drill ids each page owns under faded/guided/independent/integrated)
3. {bank} — ONLY rows with id {lo}..{hi} (fields: question_text, starter_code, answer_code, test_cases, wrong_examples, difficulty_label, submission_mode, expected_artifact_type)

Do NOT edit anything. Do NOT run code that writes.

Grade every drill {lo}..{hi} and every lesson page listed. Be concrete: for each issue give code, severity, the exact quote, and a rewritten replacement sentence (not "be clearer"). Group repeated defects (e.g. the same cross-reference pattern) under one issue listing all ids. Then list the 10 worst items overall and what a LeetCode-quality version of the prompt would say.

Output: a single markdown report. Sections: (1) Drills — per-id findings, pass items listed compactly by id; (2) Lesson pages — per page, per rubric code; (3) Systemic patterns; (4) Top 10 fixes with rewrites.
"""


def parse_range(spec: str) -> tuple[int, int]:
    lo, _, hi = spec.partition("-")
    return int(lo), int(hi or lo)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--qids", required=True, help="inclusive id range, e.g. 847-940")
    ap.add_argument("--pages", required=True, help="lesson dir or glob, e.g. Local_Deployed_Shared/lessons/einops")
    ap.add_argument("--label", required=True, help="scope label for the report name")
    ap.add_argument("--dry-run", action="store_true", help="write the task file only")
    args = ap.parse_args(argv)

    if os.environ.get("MODULARIO_CRITIC_ACTIVE") == "1":
        print("refusing: nested critic (MODULARIO_CRITIC_ACTIVE=1)", file=sys.stderr)
        return 2
    lo, hi = parse_range(args.qids)
    pages = Path(args.pages)
    page_spec = f"{pages.relative_to(REPO) if pages.is_absolute() else pages}/kp-*.md" if pages.is_dir() else str(pages)
    stamp = dt.date.today().strftime("%Y%m%d")
    task_path = REVIEW_DIR / f"{stamp}-{args.label}-task.md"
    out_path = REVIEW_DIR / f"{stamp}-{args.label}-astra.md"
    task_path.write_text(TASK.format(
        rubric=RUBRIC.relative_to(REPO), pages=page_spec, bank=BANK.relative_to(REPO), lo=lo, hi=hi))
    print(f"task -> {task_path.relative_to(REPO)}")
    if args.dry_run:
        return 0
    cmd = ["codex", "exec", "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
           "-m", MODEL, "-c", f"model_reasoning_effort={EFFORT}", "-o", str(out_path), "-"]
    env = {**os.environ, "MODULARIO_CRITIC_ACTIVE": "1"}
    with task_path.open() as fh:
        rc = subprocess.run(cmd, stdin=fh, cwd=REPO, env=env).returncode
    print(f"report -> {out_path.relative_to(REPO)} (exit {rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
